"""
Flask Backend API for Doctor Interface
Handles image upload, segmentation, XAI, and report generation
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
import numpy as np
import cv2
from PIL import Image
import io
import base64
import os
from pathlib import Path
import logging

from fl_model import MedSAM_FL
from xai_explainability import MedSAMExplainer
from cancer_staging import BreastCancerStaging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create necessary directories
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(RESULTS_FOLDER).mkdir(exist_ok=True)

# Global model instance (loaded once at startup)
MODEL_CHECKPOINT = "work_dir/MedSAM/medsam_vit_b.pth"
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Loading MedSAM model on {device}...")
try:
    model = MedSAM_FL(checkpoint_path=MODEL_CHECKPOINT, device=device)
    model.sam.eval()
    explainer = MedSAMExplainer(model, device=device)
    staging = BreastCancerStaging()
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    explainer = None
    staging = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def numpy_to_base64(image_array):
    """Convert numpy array to base64 encoded string."""
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    
    img = Image.fromarray(image_array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    status = "healthy" if model is not None else "unhealthy"
    return jsonify({
        'status': status,
        'model_loaded': model is not None,
        'device': device
    })


@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Upload endpoint for medical images.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"File uploaded: {filename}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'File uploaded successfully'
        })
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint: performs segmentation + XAI.
    
    Request JSON:
        {
            "filename": "image.jpg",
            "bbox": [x1, y1, x2, y2]  // optional, defaults to full image
        }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        filename = data.get('filename')
        bbox = data.get('bbox', [0, 0, 1024, 1024])
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        # Load image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        logger.info(f"Processing image: {filename}")
        
        # Generate attention map and segmentation
        attention_map, mask = explainer.generate_attention_map(image, np.array(bbox))
        
        # Calculate mask coverage for confidence
        mask_coverage = mask.sum() / (mask.shape[0] * mask.shape[1])
        confidence = min(0.95, 0.7 + mask_coverage * 0.3)  # Heuristic confidence
        
        # Generate XAI explanation
        xai_report = explainer.generate_explanation_report(
            image=image,
            box=np.array(bbox),
            confidence=confidence
        )
        
        # Analyze mask for staging
        mask_features = staging.analyze_mask(mask)
        stage, stage_confidence = staging.estimate_stage(mask_features)
        
        # Create visualizations
        overlay = xai_report['visualization']
        
        # Convert images to base64 for JSON response
        response = {
            'success': True,
            'segmentation_mask': numpy_to_base64(mask),
            'attention_map': numpy_to_base64((attention_map * 255).astype(np.uint8)),
            'xai_overlay': numpy_to_base64(overlay),
            'confidence': float(confidence),
            'stage': stage,
            'stage_confidence': float(stage_confidence),
            'mask_coverage': float(mask_coverage),
            'xai_description': xai_report['description'],
            'feature_importance': xai_report['feature_importance'],
            'mask_features': mask_features
        }
        
        logger.info(f"Prediction complete: Stage={stage}, Confidence={confidence:.2f}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/report', methods=['POST'])
def generate_report():
    """
    Generate comprehensive medical report.
    
    Request JSON:
        {
            "filename": "image.jpg",
            "mask_features": {...},
            "stage": "Stage I",
            "confidence": 0.85,
            "xai_data": {...},
            "patient_info": {
                "patient_id": "12345",
                "age": 45,
                "name": "Jane Doe"
            }
        }
    """
    try:
        data = request.get_json()
        filename = data.get('filename')
        mask_features = data.get('mask_features')
        stage = data.get('stage')
        confidence = data.get('confidence', 0.0)
        xai_data = data.get('xai_data', {})
        patient_info = data.get('patient_info', {})
        
        # Load mask from previous prediction
        # For now, we'll generate a dummy mask or require it in the request
        # In production, you'd store the mask temporarily
        
        # Generate clinical report
        report = staging.generate_clinical_report(
            mask=np.zeros((1024, 1024)),  # Placeholder
            mask_features=mask_features,
            stage=stage,
            confidence=confidence,
            xai_data=xai_data,
            patient_info=patient_info
        )
        
        # Save report as JSON
        report_filename = f"report_{report['report_id']}.json"
        report_path = os.path.join(app.config['RESULTS_FOLDER'], report_filename)
        staging.export_report_json(report, report_path)
        
        # Also save as text
        text_filename = f"report_{report['report_id']}.txt"
        text_path = os.path.join(app.config['RESULTS_FOLDER'], text_filename)
        staging.export_report_text(report, text_path)
        
        logger.info(f"Report generated: {report['report_id']}")
        
        return jsonify({
            'success': True,
            'report': report,
            'report_id': report['report_id'],
            'json_file': report_filename,
            'text_file': text_filename
        })
    
    except Exception as e:
        logger.error(f"Report generation error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/download_report/<report_id>', methods=['GET'])
def download_report(report_id):
    """Download report file."""
    file_format = request.args.get('format', 'json')
    
    if file_format == 'json':
        filename = f"report_{report_id}.json"
    elif file_format == 'txt':
        filename = f"report_{report_id}.txt"
    else:
        return jsonify({'error': 'Invalid format'}), 400
    
    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Report not found'}), 404
    
    return send_file(filepath, as_attachment=True)


@app.route('/fl_status', methods=['GET'])
def fl_status():
    """
    Get federated learning status and metrics.
    This would connect to the FL server/logs in production.
    """
    # Mock FL metrics for demo
    fl_metrics = {
        'status': 'completed',
        'current_round': 5,
        'total_rounds': 5,
        'global_accuracy': 0.912,
        'fairness_score': 92,
        'participating_hospitals': 3,
        'last_updated': '2025-11-15T07:30:00Z'
    }
    
    return jsonify(fl_metrics)


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size limit exceeded."""
    return jsonify({'error': 'File too large (max 16MB)'}), 413


if __name__ == '__main__':
    logger.info("Starting Flask server...")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Results folder: {RESULTS_FOLDER}")
    
    # Run app
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )
