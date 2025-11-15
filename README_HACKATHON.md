# 🏥 MedSAM Federated Learning + XAI for Breast Cancer Detection

## Hackathon Project: Privacy-Preserving AI for Medical Diagnostics

This project implements a complete federated learning system for breast cancer detection using MedSAM, with explainable AI (XAI) and a doctor-facing interface.

---

## 🎯 Project Overview

### Key Features

1. **Federated Learning with Flower**
   - Privacy-preserving training across multiple hospital nodes
   - No patient data sharing between hospitals
   - FedAvg aggregation strategy
   - FLED metrics: Fairness, Latency, Explainability, Drift

2. **MedSAM Segmentation**
   - Segment Anything Model fine-tuned for medical images
   - Breast cancer histopathology analysis
   - Automatic lesion detection and segmentation

3. **Explainable AI (XAI)**
   - Attention map visualization
   - Feature importance analysis
   - Human-interpretable explanations
   - Trust-building for medical professionals

4. **Cancer Staging & Reports**
   - Automated cancer stage estimation (Stage 0-IV)
   - Comprehensive clinical reports
   - Quantitative metrics and recommendations

5. **Doctor Interface**
   - Simple drag-and-drop image upload
   - Real-time AI analysis
   - Visual explanations with overlays
   - Downloadable reports

---

## 📁 Project Structure

```
Medsam_modified/
├── fl_server.py                 # Flower FL server
├── fl_client.py                 # Flower FL client (hospital nodes)
├── fl_model.py                  # MedSAM FL model wrapper
├── fl_data_loader.py            # Data loading for FL
├── run_fl_simulation.sh         # FL training orchestration script
│
├── xai_explainability.py        # XAI module with attention maps
├── cancer_staging.py            # Staging & report generation
├── app_backend.py               # Flask API backend
├── doctor_interface.html        # Doctor-facing web UI
│
├── requirements.txt             # Python dependencies
├── README_HACKATHON.md          # This file
└── work_dir/MedSAM/             # Model checkpoints
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
conda create -n medsam python=3.10 -y
conda activate medsam

# Install requirements
pip install -r requirements.txt

# Note: If segment-anything install fails, install manually:
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 2. Download MedSAM Model

Download the pretrained MedSAM checkpoint:
```bash
# Create directory
mkdir -p work_dir/MedSAM

# Download from Google Drive (or your checkpoint location)
# Place medsam_vit_b.pth in work_dir/MedSAM/
```

### 3. Prepare Data

Organize your breast cancer dataset:
```
data/breast_cancer/
├── imgs/
│   ├── image1.npy
│   ├── image2.npy
│   └── ...
└── gts/
    ├── mask1.npy
    ├── mask2.npy
    └── ...
```

---

## 🏥 Federated Learning

### Run FL Simulation (All-in-One)

```bash
# Make script executable
chmod +x run_fl_simulation.sh

# Run FL with 3 hospital clients
./run_fl_simulation.sh
```

### Manual FL Setup

**Terminal 1 - Start Server:**
```bash
python fl_server.py --rounds 5 --clients 3 --address 127.0.0.1:8080
```

**Terminal 2 - Start Hospital 1:**
```bash
python fl_client.py --client-id 0 --server 127.0.0.1:8080
```

**Terminal 3 - Start Hospital 2:**
```bash
python fl_client.py --client-id 1 --server 127.0.0.1:8080
```

**Terminal 4 - Start Hospital 3:**
```bash
python fl_client.py --client-id 2 --server 127.0.0.1:8080
```

### FL Checkpoints

Global model checkpoints are saved in `fl_checkpoints/`:
- `round_1_global_model.pt`
- `round_2_global_model.pt`
- ...

---

## 🩺 Doctor Interface

### 1. Start Backend API

```bash
python app_backend.py
```

Server runs on: `http://localhost:5000`

### 2. Open Doctor Interface

```bash
# Open in browser
open doctor_interface.html

# Or use Python HTTP server
python -m http.server 8000
# Then visit: http://localhost:8000/doctor_interface.html
```

### 3. Upload & Analyze

1. Drag & drop or select a breast cancer histopathology image
2. AI automatically segments and analyzes the image
3. View:
   - Cancer stage estimation
   - Confidence scores
   - XAI attention maps
   - Segmentation masks
4. Generate and download comprehensive report

---

## 🔍 API Endpoints

### Health Check
```bash
GET http://localhost:5000/health
```

### Upload Image
```bash
POST http://localhost:5000/upload
Content-Type: multipart/form-data

file: <image_file>
```

### Get Prediction + XAI
```bash
POST http://localhost:5000/predict
Content-Type: application/json

{
  "filename": "image.jpg",
  "bbox": [0, 0, 1024, 1024]
}
```

### Generate Report
```bash
POST http://localhost:5000/report
Content-Type: application/json

{
  "filename": "image.jpg",
  "mask_features": {...},
  "stage": "Stage I",
  "confidence": 0.85,
  "xai_data": {...}
}
```

### Download Report
```bash
GET http://localhost:5000/download_report/<report_id>?format=txt
```

---

## 📊 FLED Metrics

The system tracks **FLED** metrics for trustworthy FL:

- **F**airness: Standard deviation of client accuracies
- **L**atency: Communication time per round
- **E**xplainability: XAI attention maps & feature importance
- **D**rift: Model robustness across rounds

Check metrics in FL logs:
```bash
grep -E 'fairness|accuracy|dice' fl_logs/server.log
```

---

## 🧪 Testing

### Test Backend API
```bash
# Test health endpoint
curl http://localhost:5000/health

# Test with sample image
curl -X POST -F "file=@sample_image.jpg" http://localhost:5000/upload
```

### Test XAI Module
```python
from xai_explainability import MedSAMExplainer
from fl_model import MedSAM_FL
import cv2

# Load model
model = MedSAM_FL("work_dir/MedSAM/medsam_vit_b.pth")
explainer = MedSAMExplainer(model)

# Load image
image = cv2.imread("test.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate explanation
report = explainer.generate_explanation_report(
    image=image,
    box=[0, 0, 1024, 1024]
)

print(report['description'])
```

---

## 🎨 Visualizations

The system generates multiple visualizations:

1. **Original Image**: Uploaded histopathology slide
2. **Segmentation Mask**: Binary mask of detected cancer tissue
3. **Attention Map**: Heatmap showing where AI focused
4. **XAI Overlay**: Combined visualization with explanations

---

## 📈 Demo Workflow

### Complete Pipeline Demo

1. **Train with FL:**
   ```bash
   ./run_fl_simulation.sh
   ```

2. **Start Backend:**
   ```bash
   python app_backend.py
   ```

3. **Open Doctor UI:**
   Open `doctor_interface.html` in browser

4. **Analyze Sample:**
   - Upload a breast cancer slide image
   - View AI analysis with stage estimation
   - Download comprehensive report

---

## 🔐 Privacy Features

- **Federated Learning**: No raw data leaves hospital premises
- **Differential Privacy**: Optional noise addition (via Opacus)
- **Secure Aggregation**: Only model updates shared
- **Local Training**: Each hospital trains on private data

---

## 🛠️ Troubleshooting

### Model Not Loading
```bash
# Check if checkpoint exists
ls work_dir/MedSAM/medsam_vit_b.pth

# If missing, download or use your trained model
```

### FL Clients Won't Connect
```bash
# Check server is running
curl http://localhost:8080

# Ensure correct server address in clients
python fl_client.py --server 127.0.0.1:8080 --client-id 0
```

### Backend API Errors
```bash
# Check uploads directory exists
mkdir -p uploads results

# Verify model path in app_backend.py (line 46)
MODEL_CHECKPOINT = "work_dir/MedSAM/medsam_vit_b.pth"
```

---

## 📝 Hackathon Presentation Points

### 1. Problem Statement
- Medical data is sensitive and siloed across hospitals
- Traditional ML requires centralized data (privacy risk)
- Doctors need explainable AI, not black boxes

### 2. Our Solution
- **Federated Learning**: Train collaboratively without data sharing
- **MedSAM**: State-of-the-art medical image segmentation
- **XAI**: Transparent, interpretable AI decisions
- **Staging & Reports**: Actionable clinical insights

### 3. Technical Highlights
- Flower framework for production-ready FL
- Attention-based XAI for trustworthiness
- Automated cancer staging with confidence scores
- Real-time doctor interface with drag & drop

### 4. Impact
- **Privacy**: Patient data never leaves hospital
- **Collaboration**: Hospitals benefit from collective learning
- **Trust**: Doctors understand AI reasoning via XAI
- **Efficiency**: Automated analysis saves time

### 5. Demo Flow
1. Show FL training across 3 hospitals
2. Display FLED fairness metrics
3. Upload test image to doctor interface
4. Highlight XAI attention maps
5. Generate and download clinical report

---

## 🎯 Future Enhancements

- [ ] Support for DICOM medical image format
- [ ] Real-time FL model updates
- [ ] Integration with hospital PACS systems
- [ ] Multi-cancer type support
- [ ] Mobile app for doctors
- [ ] Blockchain for audit trails

---

## 📚 References

- [MedSAM Paper](https://www.nature.com/articles/s41467-024-44824-z)
- [Flower Federated Learning](https://flower.dev/)
- [Segment Anything Model](https://segment-anything.com/)

---

## 👥 Team & Contact

**Hackathon Project**  
Built with ❤️ for privacy-preserving medical AI

For questions or issues, check the repository documentation.

---

## 📜 License

This project is for educational/hackathon purposes. Please ensure compliance with medical data regulations (HIPAA, GDPR) in production use.
