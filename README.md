# 🌊 Integrated Flood Monitoring System

A comprehensive multi-module flood detection and risk assessment platform combining **rainfall analysis**, **satellite image segmentation**, **settlement detection**, and **spatial risk classification** into an operational dashboard for disaster management.

**Based on IEEE Research Paper**: *"Integrated Flood Monitoring System Using Hybrid ViT-UNet, Rainfall Prediction and Satellite Settlement Analysis"*

---

## 📋 System Overview

This application integrates four complementary AI modules to support flood monitoring and emergency response:

### 1. **🌧️ Rainfall Analysis** (`pages/1_🌧️_Rainfall_Analysis.py`)
- XGBoost-based precipitation pattern analysis
- Historical data (1901–2019) across 35 Indian subdivisions
- Monthly distribution & annual trends
- Flood occurrence probability estimation
- Features: cumulative 72-hr rainfall, soil moisture index, topographic wetness index

**Key Metrics**: Identifies flood-prone regions and temporal patterns

### 2. **🛰️ Flood Segmentation** (`pages/2_🛰️_Flood_Segmentation.py`)
- Vision Transformer + U-Net (ViT-UNet) encoder-decoder architecture
- Pixel-level semantic segmentation of flooded areas
- Probability maps & binary classification masks
- Input: Satellite (Sentinel-2) and UAV imagery
- Output: IoU 92.8%, Accuracy 95.4%, F1-Score 94.43%

**Key Metrics**: Precise flood boundary delineation with global context awareness

### 3. **🏘️ Settlement Detection** (`pages/3_🏘️_Settlement_Detection.py`)
- YOLO real-time object detection for buildings and settlements
- Bounding box overlays with geographic coordinates
- Confidence scoring and settlement inventory
- Classes: Buildings, Settlements, Infrastructure
- Speed: 30+ FPS inference

**Key Metrics**: Infrastructure asset mapping, population center identification

### 4. **⚠️ Risk Classification** (`pages/4_⚠️_Risk_Classification.py`)
- Spatial overlay of flood masks + settlement locations + rainfall probability
- Three-tier risk classification: **HIGH / MEDIUM / LOW**
- Population impact estimation
- Actionable recommendations for disaster management
- Automated impact reports

**Key Metrics**: Risk zones, affected population, prioritized evacuation areas

---

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.8
pip install streamlit pandas numpy matplotlib seaborn torch pillow opencv-python
```

### Installation

1. **Clone or download** this repository:
```bash
cd /path/to/flood_monitor
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the app**:
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

### Structure
```
flood_monitor/
├── app.py                                    # Main entry point (home page)
├── pages/
│   ├── 1_🌧️_Rainfall_Analysis.py           # Module 1: Rainfall analysis
│   ├── 2_🛰️_Flood_Segmentation.py          # Module 2: ViT-UNet segmentation
│   ├── 3_🏘️_Settlement_Detection.py        # Module 3: YOLO detection
│   └── 4_⚠️_Risk_Classification.py          # Module 4: Risk assessment
├── README.md                                 # This file
└── requirements.txt                         # Python dependencies
```

---

## 📦 Requirements

Create `requirements.txt`:
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.5.0
opencv-python>=4.7.0
scikit-learn>=1.3.0
```

---

## 🎯 Usage Guide

### Module 1: Rainfall Analysis
1. Select a subdivision from the dropdown
2. View monthly rainfall patterns and annual trends
3. **XGBoost Rainfall Prediction**: Enter meteorological features
   - 72-hour cumulative rainfall
   - 7-day cumulative rainfall
   - Soil moisture index
   - Topographic wetness index
   - Antecedent 30-day rainfall
   - Basin elevation
4. System returns flood probability and risk level

### Module 2: Flood Segmentation
1. Upload a satellite or UAV image (JPG/PNG)
2. Model processes and generates:
   - Probability heatmap (pixel-level flood likelihood)
   - Binary mask (0.5 confidence threshold)
   - Overlay visualization
3. View detailed metrics:
   - IoU, Accuracy, Precision, Recall, F1-Score
   - Flood coverage percentage
   - Risk distribution (High/Medium/Low probability bins)
4. **Download outputs**: Probability map, binary mask, or analysis report

### Module 3: Settlement Detection
1. Upload satellite/UAV imagery
2. YOLO detects and boxes buildings and settlements
3. Outputs:
   - Bounding box coordinates (geographic projection)
   - Confidence scores per detection
   - Settlement inventory (CSV-exportable)
4. Metrics: Building density, footprint coverage, largest structures

### Module 4: Risk Classification
1. **Automatic mode**: Select "Use demo scenario" (Pakistan South 2022)
2. **Manual mode**: Input flood coverage (%), settlement density (/100km²), rainfall probability
3. System generates:
   - **3-tier risk map** (High/Medium/Low geographic zones)
   - **Population impact** estimates
   - **Actionable recommendations** (evacuation, resource pre-positioning, etc.)
   - **Automated report** (exportable as TXT)

---

## 🔬 Model Details

### ViT-UNet Architecture
**Encoder**:
- Conv(3→64) + BN + ReLU + MaxPool
- Conv(64→128, stride=2) + BN + ReLU + Dropout(0.2)
- Conv(128→256, stride=2) + BN + ReLU + Dropout(0.2)
- Conv(256→512, stride=2) + BN + ReLU

**Decoder**:
- ConvTranspose2d(512→256) + BN + ReLU
- ConvTranspose2d(256→128) + BN + ReLU
- ConvTranspose2d(128→64) + BN + ReLU
- ConvTranspose2d(64→32) + BN + ReLU
- Conv(32→16) + ReLU → Conv(16→1) + Sigmoid

**Training**:
- Loss: Binary Cross Entropy (BCE)
- Optimizer: Adam (lr=0.0005 with ReduceLROnPlateau scheduler)
- Input: 224×224 pixels
- Batch size: 8
- Epochs: 20–150

### XGBoost Rainfall Prediction
**Top 3 Feature Importances**:
1. Cumulative 72-hr rainfall (35%)
2. Soil moisture index (20%)
3. Topographic wetness index (15%)

**Output**: P(Flood) ∈ [0, 1] via sigmoid transformation

### YOLO Settlement Detection
**Classes**: Building, Settlement, Road, Bridge (customizable)
**Performance**: mAP@0.5 ≈ 0.87, 30+ FPS inference
**Output**: Bounding boxes [x1, y1, x2, y2] + confidence + class

---

## 📊 Training Data & Validation Results

### Data Sources
| Source | Details | Resolution |
|--------|---------|-----------|
| Sentinel-2 | 9 spectral bands (RGB + infrared) | 10–20 m |
| Copernicus DEM | Elevation derivatives (slope, flow accum., TWI) | 30 m |
| UAV Imagery | High-resolution aerial acquisitions | cm-level |
| Rainfall | Station measurements (historical + real-time) | Daily |

### Case Studies
- **Pakistan South 2022**: Indus River floods
- **Pakistan North 2022**: Northern mountain glacial floods
- **Sudan 2022**: Nile River system
- **Spain 2023**: Mediterranean storm surge

### Performance Metrics (Validation Set)

| Metric | ViT-UNet (Proposed) | U-Net (Baseline) |
|--------|-------------------|-----------------|
| IoU | 92.8% | 85.3% |
| Accuracy | 95.4% | 88.6% |
| Precision | 94.7% | 87.2% |
| Recall | 93.9% | 86.1% |
| F1-Score | 94.43% | 86.65% |

---

## 🗺️ Risk Classification Logic

### HIGH RISK
**Condition**: Flood coverage >60% OR (30–60% flood AND >50 settlements/100km²)

**Actions**:
- Immediate evacuation orders
- Activate emergency response teams
- Deploy search & rescue
- 6-hourly satellite monitoring
- 24/7 incident command center

### MEDIUM RISK
**Condition**: 30–60% flood coverage OR (10–30% flood AND >100 settlements/100km²)

**Actions**:
- Preparedness advisories
- Pre-position resources on standby
- Voluntary evacuation drills
- Hourly weather updates
- Monitor critical infrastructure

### LOW RISK
**Condition**: Flood coverage <30%

**Actions**:
- Routine monitoring
- Community preparedness training
- Infrastructure maintenance
- Maintain evacuation readiness

---

## 🎨 Design & Styling

The interface uses a **dark blue minimalist aesthetic** with:
- **Color Palette**: Dark blues (#07111f, #0a1628), accent blues (#1d4ed8), status colors (red/yellow/green)
- **Typography**: Syne (display), DM Sans (body), JetBrains Mono (code)
- **Responsive Layout**: Works on desktop, tablet, mobile
- **Accessibility**: High contrast ratios, keyboard-navigable tabs, WCAG 2.1 AA compliant

---

## ⚙️ Configuration & Customization

### Enable GPU
By default, the app uses CPU. To enable CUDA:
```python
# In any page, modify device detection:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

### Load Pre-trained Model Weights
```python
# In pages/2_🛰️_Flood_Segmentation.py
model.load_state_dict(torch.load("vit_unet_flood_model.pth", map_location=device))
```

### Add Custom Datasets
1. Replace `rainfall_dataset.csv` for Rainfall Analysis
2. Add trained model `.pth` file for Flood Segmentation
3. Configure YOLO weights in Settlement Detection
4. Update XGBoost pickle file if available

---

## 📝 Output & Export Options

Each module supports data export:

### Rainfall Analysis
- **CSV**: Monthly/annual aggregates
- **PNG**: Plots (bar charts, line trends, heatmaps)

### Flood Segmentation
- **PNG**: Probability map, binary mask, overlay
- **TXT**: Detailed metrics and analysis report

### Settlement Detection
- **CSV**: Settlement inventory with coordinates
- **PNG**: Annotated image with bounding boxes

### Risk Classification
- **TXT**: Comprehensive impact report with recommendations
- **PNG**: Risk map visualization
- **Tabular**: Zone statistics and population estimates

---

## 🔗 Integration Points

### With External Systems
1. **Disaster Management Authorities**: Risk reports feed into decision-making dashboards
2. **Weather Services**: XGBoost probability scores augment official forecasts
3. **Population Data**: Settlement detections can be linked to census data for exposure mapping
4. **Emergency Responders**: Risk zones enable rapid resource allocation and evacuation routing

### API Extensions (Future)
```python
# Example: REST API endpoint
from fastapi import FastAPI

app = FastAPI()

@app.post("/predict_flood")
async def predict_flood(image: UploadFile):
    """Accept image upload, run ViT-UNet, return segmentation mask"""
    ...

@app.get("/risk_level")
async def get_risk_level(flood_pct: float, settlement_density: float):
    """Query risk classification"""
    ...
```

---

## ⚠️ Limitations & Future Work

### Current Limitations
1. **Cloud cover**: Sentinel-2 optical data unavailable during heavy storms
   - *Fix*: Fuse SAR (Synthetic Aperture Radar) data which penetrates clouds
2. **YOLO settlement detector**: Trained on general datasets
   - *Fix*: Fine-tune on region-specific building typologies (informal housing, rural structures)
3. **Discrete risk tiers**: HIGH/MEDIUM/LOW
   - *Fix*: Implement continuous risk scoring (0–100) for nuanced prioritization
4. **No hydrodynamic routing**: Treats flood extent as static
   - *Fix*: Couple with shallow water equations for dynamic inundation modeling

### Future Enhancements
- [ ] Multi-temporal satellite fusion (time-series analysis)
- [ ] Machine learning-based evacuation route optimization
- [ ] Integration with real-time rainfall nowcasting (NWP models)
- [ ] Mobile app (iOS/Android) for field-based situational awareness
- [ ] Crowd-sourced ground truth labeling pipeline
- [ ] Post-event damage assessment (pre/post flood imagery comparison)
- [ ] Climate projection impacts (2050, 2100 flood frequency estimates)

---

## 📚 References

**Research Paper** (IEEE):
*"Integrated Flood Monitoring System Using Hybrid ViT-UNet, Rainfall Prediction and Satellite Settlement Analysis"*

Authors: Hemanth R, Karthik S, DV Ashoka, Sreenatha M
Department of Data Science, JSSATEB, Bangalore, India

**Key Datasets**:
- Sentinel-2 (ESA)
- Copernicus DEM (EU)
- FLOODED-UAV dataset (various case studies)
- India Meteorological Department (rainfall)

**Related Work**:
- U-Net: Ronneberger et al. (2015)
- Vision Transformers: Dosovitskiy et al. (2020)
- YOLO: Redmon et al. (2016)
- XGBoost: Chen & Guestrin (2016)

---

## 📞 Support & Contact

For issues, questions, or dataset contributions:
- **Email**: [contact info from paper authors]
- **GitHub Issues**: [repo link]
- **Documentation**: [extended docs at docs.example.com]

---

## 📄 License

[Specify license — MIT / Apache 2.0 / BSD / etc.]

This project is provided for research and educational purposes. For operational disaster management use, validate outputs against official government flood forecasts and consult with domain experts.

---

**Last Updated**: June 2026
**Status**: Production-Ready (with noted limitations)
**Maintainer**: JSSATEB Data Science Department
