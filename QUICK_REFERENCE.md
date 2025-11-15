# 🚀 Quick Reference Guide

## One-Command Starts

### Start Doctor Interface
```bash
./start_demo.sh
# Then open: doctor_interface.html in browser
```

### Start Federated Learning
```bash
./run_fl_simulation.sh
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Federated Learning Layer                  │
│  Hospital 1 ──┐                                              │
│  Hospital 2 ──┼─→ FL Server (Flower) → Global Model         │
│  Hospital 3 ──┘                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      MedSAM Model Layer                      │
│  • Segment Anything (SAM) Architecture                       │
│  • Fine-tuned for Medical Images                             │
│  • Breast Cancer Histopathology                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     XAI + Analysis Layer                     │
│  • Attention Maps           • Feature Importance             │
│  • Cancer Staging           • Report Generation              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Doctor Interface                        │
│  Upload Image → AI Analysis → View Results → Download Report │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Files & What They Do

| File | Purpose |
|------|---------|
| `fl_server.py` | Federated learning server (aggregates models) |
| `fl_client.py` | Hospital node (trains locally) |
| `app_backend.py` | REST API for doctor interface |
| `doctor_interface.html` | Web UI for doctors |
| `xai_explainability.py` | Attention maps & explanations |
| `cancer_staging.py` | Stage estimation & reports |
| `fl_model.py` | MedSAM wrapper for FL |

---

## Common Commands

### Install Everything
```bash
pip install -r requirements.txt
```

### Test Backend
```bash
python app_backend.py
curl http://localhost:5000/health
```

### Run FL (Manual)
```bash
# Terminal 1: Server
python fl_server.py --rounds 5 --clients 3

# Terminal 2-4: Clients
python fl_client.py --client-id 0
python fl_client.py --client-id 1
python fl_client.py --client-id 2
```

### View FL Logs
```bash
tail -f fl_logs/server.log
grep -E 'fairness|accuracy' fl_logs/server.log
```

---

## Data Flow

### 1. Doctor Uploads Image
```
doctor_interface.html 
  ↓ (POST /upload)
app_backend.py → saves to uploads/
```

### 2. AI Analysis
```
doctor_interface.html
  ↓ (POST /predict)
app_backend.py
  ↓
MedSAM_FL (segmentation)
  ↓
XAI Explainer (attention maps)
  ↓
Cancer Staging (stage estimation)
  ↓ (JSON response)
doctor_interface.html (displays results)
```

### 3. Report Generation
```
doctor_interface.html
  ↓ (POST /report)
app_backend.py
  ↓
BreastCancerStaging.generate_clinical_report()
  ↓
Saves report_{id}.txt and report_{id}.json
  ↓
Returns download link
```

---

## API Quick Reference

### Upload Image
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/upload
```

### Get Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"filename":"image.jpg","bbox":[0,0,1024,1024]}'
```

### Generate Report
```bash
curl -X POST http://localhost:5000/report \
  -H "Content-Type: application/json" \
  -d '{"filename":"image.jpg","stage":"Stage I","confidence":0.85,...}'
```

---

## Troubleshooting

### Model Not Found
```bash
# Check if model exists
ls work_dir/MedSAM/medsam_vit_b.pth

# If not, place your trained model there
mkdir -p work_dir/MedSAM
# Copy your medsam_vit_b.pth to this directory
```

### Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000

# Kill it
kill -9 <PID>

# Or use different port
python app_backend.py --port 5001
```

### CORS Errors
- Make sure Flask-CORS is installed: `pip install flask-cors`
- Backend must be running when opening HTML file

### Image Not Processing
1. Check file format (JPG, PNG supported)
2. Check file size (max 16MB)
3. View browser console for errors (F12)
4. Check backend logs for errors

---

## Demo Checklist

- [ ] Dependencies installed
- [ ] MedSAM model in `work_dir/MedSAM/`
- [ ] Test data available (breast cancer images)
- [ ] Backend starts without errors
- [ ] Can open doctor interface
- [ ] Can upload test image
- [ ] Results display correctly
- [ ] Can generate report

---

## Presentation Flow

### 1. Introduction (2 min)
- Show problem: privacy in medical AI
- Explain federated learning concept

### 2. FL Demo (3 min)
```bash
./run_fl_simulation.sh
# Show terminal output
# Highlight fairness metrics
```

### 3. Doctor Interface Demo (4 min)
1. Open `doctor_interface.html`
2. Upload breast cancer slide
3. Show real-time analysis
4. Highlight XAI attention maps
5. Generate and download report

### 4. Technical Deep Dive (3 min)
- Flower FL architecture
- MedSAM segmentation
- XAI transparency
- Cancer staging logic

### 5. Impact & Future (1 min)
- Privacy preservation
- Multi-hospital collaboration
- Trust through XAI
- Future enhancements

---

## Keyboard Shortcuts (for Demo)

- **Cmd+R**: Refresh browser
- **Cmd+Shift+R**: Hard refresh
- **Cmd+Option+I**: Open browser console
- **Ctrl+C**: Stop backend server

---

## Emergency Fixes

### Quick Backend Restart
```bash
pkill -f app_backend.py
python app_backend.py
```

### Clear Uploads
```bash
rm -rf uploads/* results/*
```

### Reset FL
```bash
rm -rf fl_checkpoints/* fl_logs/*
```

---

## Performance Tips

- Use `mps` device on Mac M1/M2: Auto-detected
- Reduce batch size if out of memory
- Use smaller images for faster demo
- Pre-load sample images

---

## Contact & Support

If something breaks during the hackathon:
1. Check this guide first
2. Read error messages carefully
3. Check file paths in code
4. Restart backend/FL as needed

Good luck! 🚀
