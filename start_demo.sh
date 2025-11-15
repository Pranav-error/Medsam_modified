#!/bin/bash
# Quick Start Script for MedSAM Hackathon Demo
# This script starts the doctor interface (backend only)

set -e

echo "🏥 MedSAM Breast Cancer Detection System"
echo "========================================="
echo ""

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Check if model exists
if [ ! -f "work_dir/MedSAM/medsam_vit_b.pth" ]; then
    echo "⚠️  Warning: MedSAM model not found!"
    echo "Please place medsam_vit_b.pth in work_dir/MedSAM/"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create necessary directories
mkdir -p uploads results fl_logs fl_checkpoints

echo "📦 Setting up directories..."
echo "  ✓ uploads/"
echo "  ✓ results/"
echo "  ✓ fl_logs/"
echo "  ✓ fl_checkpoints/"
echo ""

# Check if requirements are installed
echo "🔍 Checking dependencies..."
python -c "import flask" 2>/dev/null || { echo "⚠️  Flask not installed. Run: pip install -r requirements.txt"; exit 1; }
python -c "import torch" 2>/dev/null || { echo "⚠️  PyTorch not installed. Run: pip install -r requirements.txt"; exit 1; }
echo "  ✓ All dependencies found"
echo ""

# Start backend server
echo "🚀 Starting Flask Backend API..."
echo "  Server will run on: http://localhost:5000"
echo ""
echo "📋 Available endpoints:"
echo "  - Health check: http://localhost:5000/health"
echo "  - Upload image: POST /upload"
echo "  - Get prediction: POST /predict"
echo "  - Generate report: POST /report"
echo ""
echo "🌐 To use the Doctor Interface:"
echo "  1. Open doctor_interface.html in your browser"
echo "  2. Or run: open doctor_interface.html"
echo "  3. Or serve via: python -m http.server 8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================="
echo ""

# Start Flask app
python app_backend.py
