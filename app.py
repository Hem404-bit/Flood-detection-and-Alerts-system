import streamlit as st

st.set_page_config(page_title="Test")

st.markdown("""
<h1 style="color:red;">Hello Streamlit!</h1>
<p>This should appear as HTML, not code.</p>
""", unsafe_allow_html=True)
# ── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #071525 0%, #0d2240 45%, #081d38 100%);
    border: 1px solid #112e50;
    border-radius: 16px;
    padding: 3rem 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
">
    <div style="
        position: absolute; top: -40px; right: -40px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
        border-radius: 50%;
    "></div>
    <div style="
        position: absolute; bottom: -30px; left: 30%;
        width: 160px; height: 160px;
        background: radial-gradient(circle, rgba(16,185,129,0.08) 0%, transparent 70%);
        border-radius: 50%;
    "></div>

    <div style="position:relative; z-index:1;">
        <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
            <div style="
                background: linear-gradient(135deg,#1d4ed8,#1e40af);
                border-radius: 12px; padding: 0.6rem 0.9rem;
                font-size: 1.8rem; line-height:1;
            ">water</div>
            <div>
                <p style="color:#3b82f6; font-size:0.75rem; letter-spacing:0.18em; text-transform:uppercase; margin:0 0 0.2rem; font-family:'DM Sans',sans-serif; font-weight:500;">
                    JSSATEB · Data Science Dept.
                </p>
                <h1 style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#e0eeff; margin:0; line-height:1.15;">
                    Integrated Flood Monitoring System
                </h1>
            </div>
        </div>
        <p style="color:#4e7aa8; font-size:0.95rem; margin:0; max-width:700px; line-height:1.65;">
            ViT-UNet flood segmentation · XGBoost rainfall prediction · YOLO settlement detection ·
            Spatial risk classification — unified in a single operational pipeline.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Module cards ─────────────────────────────────────────────────────────────
MODULES = [
    ("🌧️", "Rainfall Analysis",
     "XGBoost-based precipitation analysis across 35 Indian subdivisions (1901–2019). "
     "Monthly patterns, annual trends, heatmaps, and flood occurrence probability.",
     "#1d4ed8", "#1e3a5f",
     "pages/1_🌧️_Rainfall_Analysis.py"),
    ("🛰️", "Flood Segmentation",
     "ViT-UNet semantic segmentation on satellite and UAV imagery. "
     "Upload an image to detect flooded pixels with probability overlay and binary mask.",
     "#065f46", "#0d3528",
     "pages/2_🛰️_Flood_Segmentation.py"),
    ("🏘️", "Settlement Detection",
     "YOLO-based settlement and building detection on aerial imagery. "
     "Bounding-box overlays with geographic coordinates and impact inventory.",
     "#7c3aed", "#3b1e78",
     "pages/3_🏘️_Settlement_Detection.py"),
    ("⚠️", "Risk Classification",
     "Spatial overlay of flood masks + detected settlements + XGBoost probability. "
     "Three-tier risk map (High / Medium / Low) with impact statistics and reports.",
     "#b45309", "#4a2007",
     "pages/4_⚠️_Risk_Classification.py"),
]

cols = st.columns(2)
for i, (icon, title, desc, accent, dark, _path) in enumerate(MODULES):
    with cols[i % 2]:
        st.markdown(f"""
        <div style="
            background: #0a1628;
            border: 1px solid {dark};
            border-left: 3px solid {accent};
            border-radius: 12px;
            padding: 1.5rem 1.6rem;
            margin-bottom: 1.2rem;
            transition: border-color 0.2s;
        ">
            <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.75rem;">
                <span style="font-size:1.5rem;">{icon}</span>
                <span style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; color:#c8e0f8;">{title}</span>
            </div>
            <p style="color:#4e7aa8; font-size:0.83rem; margin:0; line-height:1.6;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Paper metrics ─────────────────────────────────────────────────────────────
st.markdown("""
<p style="font-family:'Syne',sans-serif; color:#7fa8d4; font-size:0.72rem;
   letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.8rem;">
  ViT-UNet · Reported Performance (IEEE Paper)
</p>
""", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("IoU", "92.8%", "↑ 7.5pp vs U-Net")
m2.metric("Accuracy", "95.4%")
m3.metric("Precision", "94.7%")
m4.metric("Recall", "93.9%")
m5.metric("F1-Score", "94.43%")

st.markdown("<br>", unsafe_allow_html=True)

# ── Architecture note ─────────────────────────────────────────────────────────
with st.expander("📄 About this system — paper abstract"):
    st.markdown("""
    <div style="color:#7fa8d4; font-size:0.88rem; line-height:1.75; font-style:italic; padding:0.5rem 0;">
    Floods are among the most destructive natural disasters, causing significant economic losses,
    environmental damage, and threats to human life. This system proposes an intelligent flood detection
    and segmentation framework that integrates rainfall analysis with a <strong style="color:#a8c8ef;">
    Vision Transformer U-Net (ViT-UNet)</strong> architecture for automated flood mapping from satellite imagery.
    The framework additionally incorporates <strong style="color:#a8c8ef;">YOLO-based settlement detection</strong>
    and spatial risk classification to support disaster management decision-making.
    <br><br>
    The system was trained on multi-source flood-affected satellite images (Sentinel-2, Copernicus DEM) and UAV
    acquisitions across Pakistan South, Pakistan North, Sudan, and Spain.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<p style="color:#1e3a5f; font-size:0.75rem; text-align:center; margin-top:2rem;">
    Dept. of Data Science, JSSATEB, Bangalore · Hemanth R · Karthik S · DV Ashoka · Sreenatha M
</p>
""", unsafe_allow_html=True)
