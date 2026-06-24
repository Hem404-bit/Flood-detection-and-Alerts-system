import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

st.set_page_config(page_title="Settlement Detection", page_icon="🏘️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #07111f; }
[data-testid="stSidebar"] { background: #050e1a !important; border-right: 1px solid #112240 !important; }
[data-testid="stSidebar"] * { color: #7fa8d4 !important; }
.main .block-container { background: #07111f; }
[data-testid="metric-container"] { background:#0a1a2e; border:1px solid #112240; border-radius:10px; padding:1rem 1.2rem; }
[data-testid="metric-container"] [data-testid="stMetricLabel"] { color:#4e7aa8 !important; font-size:0.78rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#e0eeff !important; font-size:1.5rem !important; font-weight:700 !important; }
</style>
""", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #071525 0%, #0d2240 60%, #071d35 100%);
    border: 1px solid #112e50; border-left: 3px solid #7c3aed;
    border-radius: 14px; padding: 1.8rem 2.2rem; margin-bottom: 2rem;
">
    <h2 style="font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:700; color:#e0eeff; margin:0 0 0.3rem;">
        🏘️ Settlement & Building Detection · YOLO
    </h2>
    <p style="color:#4e7aa8; font-size:0.88rem; margin:0;">
        Real-time object detection · Bounding box overlays · Settlement inventory ·
        Geographic coordinates · Impact assessment on flooded areas
    </p>
</div>
""", unsafe_allow_html=True)

# ── Mock YOLO-style detector ──────────────────────────────────────────────────
def detect_settlements(image_array):
    """Simulate YOLO settlement detection with bounding boxes."""
    h, w = image_array.shape[:2]
    
    # Mock detections: randomly generate building-like regions
    np.random.seed(hash(image_array.tobytes()) % 2**32)
    num_detections = np.random.randint(15, 40)
    
    detections = []
    for _ in range(num_detections):
        x1 = np.random.randint(0, max(1, w - 60))
        y1 = np.random.randint(0, max(1, h - 60))
        box_w = np.random.randint(20, 80)
        box_h = np.random.randint(20, 80)
        x2 = min(w, x1 + box_w)
        y2 = min(h, y1 + box_h)
        confidence = np.random.uniform(0.72, 0.98)
        
        detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": confidence,
            "class": "building" if confidence > 0.8 else "settlement"
        })
    
    return detections

# ── UI Layout ─────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📤 Detect & Analyze", "📖 Model Info"])

with tab1:
    st.markdown("### Upload satellite or UAV imagery")
    uploaded_file = st.file_uploader("Choose an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        st.markdown("---")
        
        # Run detection
        with st.spinner("🔄 Running YOLO settlement detection..."):
            detections = detect_settlements(img_array)
            detections_sorted = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        # Metrics
        total_detections = len(detections_sorted)
        buildings = sum(1 for d in detections_sorted if d["class"] == "building")
        settlements = total_detections - buildings
        avg_confidence = np.mean([d["confidence"] for d in detections_sorted])
        
        st.markdown("### 📊 Detection Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Detections", total_detections)
        m2.metric("Buildings", buildings)
        m3.metric("Settlements", settlements)
        m4.metric("Avg Confidence", f"{avg_confidence:.2%}")
        
        # Visualization
        st.markdown("### 🖼️ Detection Results")
        
        viz_c1, viz_c2 = st.columns(2)
        
        with viz_c1:
            st.markdown("**Bounding Box Overlay**")
            img_with_boxes = image.copy()
            draw = ImageDraw.Draw(img_with_boxes)
            
            colors = {
                "building": (185, 28, 28, 200),      # Red
                "settlement": (168, 85, 247, 200)     # Purple
            }
            
            for det in detections_sorted[:30]:  # Show top 30
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                color = colors[det["class"]]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                # Label
                label = f"{det['class']}: {det['confidence']:.0%}"
                draw.text((x1, y1 - 10), label, fill=color)
            
            st.image(img_with_boxes, use_column_width=True)
        
        with viz_c2:
            st.markdown("**Confidence Distribution**")
            confidences = [d["confidence"] for d in detections_sorted]
            
            fig_conf, ax_conf = plt.subplots(figsize=(8, 5))
            fig_conf.patch.set_facecolor("#07111f"); ax_conf.set_facecolor("#0a1628")
            
            ax_conf.hist(confidences, bins=20, color="#a78bfa", edgecolor="#7c3aed", linewidth=1.5)
            ax_conf.axvline(avg_confidence, color="#f59e0b", linestyle="--", linewidth=2, label=f"Mean: {avg_confidence:.2%}")
            
            ax_conf.set_xlabel("Confidence Score", color="#4e7aa8")
            ax_conf.set_ylabel("Count", color="#4e7aa8")
            ax_conf.tick_params(colors="#4e7aa8")
            ax_conf.spines[["top", "right"]].set_visible(False)
            ax_conf.legend(loc="upper left", facecolor="#07111f", edgecolor="#112240", labelcolor="#7fa8d4")
            ax_conf.grid(axis="y", alpha=0.2, color="#1e3a5f")
            
            fig_conf.tight_layout()
            st.pyplot(fig_conf); plt.close(fig_conf)
        
        # Settlement inventory
        st.markdown("---")
        st.markdown("### 📋 Settlement Inventory")
        
        inv_data = []
        for i, det in enumerate(detections_sorted[:50], 1):
            width = det["x2"] - det["x1"]
            height = det["y2"] - det["y1"]
            area = width * height
            inv_data.append({
                "ID": i,
                "Type": det["class"].title(),
                "Confidence": f"{det['confidence']:.2%}",
                "X1 (px)": det["x1"],
                "Y1 (px)": det["y1"],
                "Width (px)": width,
                "Height (px)": height,
                "Area (px²)": area
            })
        
        inv_df = st.dataframe(
            inv_data, use_container_width=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1)
            }
        )
        
        # Risk analysis with settlements
        st.markdown("---")
        st.markdown("### ⚠️ Flood Risk to Settlements")
        st.markdown("""
        <p style="color:#4e7aa8; font-size:0.85rem;">
            To assess flood risk to detected settlements, overlay this detection output with the flood segmentation mask
            from the <strong style="color:#c8e0f8;">Flood Segmentation</strong> module. This enables:
        </p>
        <ul style="color:#4e7aa8; font-size:0.85rem; line-height:1.8;">
            <li>Count of settlements in flooded areas</li>
            <li>Population exposure estimation</li>
            <li>Priority zones for evacuation (highest settlement density in flood zones)</li>
            <li>Infrastructure impact assessment</li>
        </ul>
        """, unsafe_allow_html=True)
        
        # Spatial statistics
        st.markdown("### 📍 Spatial Statistics")
        
        spatial_c1, spatial_c2, spatial_c3 = st.columns(3)
        
        with spatial_c1:
            st.markdown("**Detection Density**")
            total_area = img_array.shape[0] * img_array.shape[1]
            density = total_detections / (total_area / 10000)  # Per 10k pixels
            st.metric("Detections per 10k px²", f"{density:.2f}")
        
        with spatial_c2:
            st.markdown("**Building Footprint**")
            total_building_area = sum((d["x2"]-d["x1"])*(d["y2"]-d["y1"]) 
                                     for d in detections_sorted if d["class"]=="building")
            coverage = (total_building_area / total_area) * 100
            st.metric("Total Building Area", f"{coverage:.1f}%")
        
        with spatial_c3:
            st.markdown("**Largest Structure**")
            if detections_sorted:
                largest = max(detections_sorted, 
                             key=lambda x: (x["x2"]-x["x1"])*(x["y2"]-x["y1"]))
                area = (largest["x2"]-largest["x1"]) * (largest["y2"]-largest["y1"])
                st.metric("Max Area (px²)", f"{area:,}")
        
        # Export
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        
        exp_c1, exp_c2 = st.columns(2)
        
        with exp_c1:
            if st.button("📥 Download Settlement Inventory (CSV)"):
                csv_str = "ID,Type,Confidence,X1,Y1,Width,Height,Area\n"
                for row in inv_data:
                    csv_str += f"{row['ID']},{row['Type']},{row['Confidence']},{row['X1 (px)']},{row['Y1 (px)']},{row['Width (px)']},{row['Height (px)']},{row['Area (px²)']}\n"
                st.download_button("💾 Save CSV", csv_str, "settlements.csv", "text/csv")
        
        with exp_c2:
            if st.button("📥 Download Annotated Image"):
                img_ann = image.copy()
                draw = ImageDraw.Draw(img_ann)
                for det in detections_sorted[:50]:
                    color = (185, 28, 28) if det["class"]=="building" else (168, 85, 247)
                    draw.rectangle([det["x1"], det["y1"], det["x2"], det["y2"]], 
                                 outline=color, width=2)
                img_ann.save("/tmp/settlements_annotated.png")
                with open("/tmp/settlements_annotated.png", "rb") as f:
                    st.download_button("💾 Save PNG", f, "settlements_annotated.png", "image/png")

with tab2:
    st.markdown("""
    <h3 style="font-family:'Syne',sans-serif; color:#c8e0f8;">YOLO Architecture Overview</h3>
    <p style="color:#7fa8d4; line-height:1.7;">
    <strong>YOLO (You Only Look Once)</strong> is a real-time object detection framework that predicts bounding boxes
    and class labels in a single forward pass. Unlike traditional detection pipelines that generate region proposals,
    YOLO divides the image into a grid and directly predicts object locations and confidence scores for each grid cell.
    This makes it exceptionally fast for deployment on UAV and satellite imagery.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style="font-family:'Syne',sans-serif; color:#c8e0f8; margin-top:2rem;">Detection Pipeline</h3>
    <div style="background:#0a1628; border:1px solid #112240; border-radius:10px; padding:1.2rem; margin-top:1rem;">
        <ol style="color:#7fa8d4; font-size:0.85rem; line-height:2;">
            <li><strong style="color:#a8c8ef;">Input Image</strong> — High-resolution satellite/UAV imagery (arbitrary resolution)</li>
            <li><strong style="color:#a8c8ef;">Grid Division</strong> — Image divided into S×S cells (default 13×13 or 19×19)</li>
            <li><strong style="color:#a8c8ef;">Feature Extraction</strong> — CNN backbone (Darknet/ResNet) extracts spatial features</li>
            <li><strong style="color:#a8c8ef;">Bounding Box Prediction</strong> — Each grid cell predicts:
                <ul style="margin-top:0.3rem;">
                    <li>B bounding boxes (x, y, w, h) + confidence score</li>
                    <li>C class probabilities (building, settlement, road, etc.)</li>
                </ul>
            </li>
            <li><strong style="color:#a8c8ef;">Post-Processing</strong> — Non-Maximum Suppression (NMS) removes overlapping boxes</li>
            <li><strong style="color:#a8c8ef;">Output</strong> — Bounding boxes with class labels and geographic coordinates</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style="font-family:'Syne',sans-serif; color:#c8e0f8; margin-top:2rem;">Advantages for Settlement Detection</h3>
    <div style="background:#0a1628; border:1px solid #112240; border-radius:10px; padding:1.2rem; margin-top:1rem;">
        <ul style="color:#7fa8d4; font-size:0.85rem; line-height:2;">
            <li><strong style="color:#c8e0f8;">Speed</strong> — Real-time inference (30+ FPS); suitable for rapid flood response</li>
            <li><strong style="color:#c8e0f8;">Multi-scale Detection</strong> — Handles buildings of varying sizes (small huts to large structures)</li>
            <li><strong style="color:#c8e0f8;">Confidence Scoring</strong> — Each detection includes confidence; enables filtering by reliability</li>
            <li><strong style="color:#c8e0f8;">Geographic Mapping</strong> — Bounding box coordinates directly map to geospatial coordinates</li>
            <li><strong style="color:#c8e0f8;">Transferability</strong> — Models trained on one region generalize well across different architectures</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style="font-family:'Syne',sans-serif; color:#c8e0f8; margin-top:2rem;">Training Data & Classes</h3>
    <div style="background:#0a1628; border:1px solid #112240; border-radius:10px; padding:1.2rem; margin-top:1rem;">
        <table style="width:100%; color:#7fa8d4; font-size:0.85rem;">
            <tr style="border-bottom:1px solid #1e3a5f;">
                <td style="padding:0.5rem;"><strong>Class</strong></td>
                <td style="padding:0.5rem;"><strong>Description</strong></td>
                <td style="padding:0.5rem;"><strong>Use</strong></td>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f;">
                <td style="padding:0.5rem;"><strong style="color:#c8e0f8;">Building</strong></td>
                <td style="padding:0.5rem;">Individual residential/commercial structures (rooftop visible)</td>
                <td style="padding:0.5rem;">Infrastructure asset mapping</td>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f;">
                <td style="padding:0.5rem;"><strong style="color:#c8e0f8;">Settlement</strong></td>
                <td style="padding:0.5rem;">Clusters of buildings (hamlet, village extent)</td>
                <td style="padding:0.5rem;">Population center identification</td>
            </tr>
            <tr style="border-bottom:1px solid #1e3a5f;">
                <td style="padding:0.5rem;"><strong style="color:#c8e0f8;">Road</strong></td>
                <td style="padding:0.5rem;">Linear transportation features (optional)</td>
                <td style="padding:0.5rem;">Evacuation route assessment</td>
            </tr>
            <tr>
                <td style="padding:0.5rem;"><strong style="color:#c8e0f8;">Bridge</strong></td>
                <td style="padding:0.5rem;">Critical water-crossing infrastructure (optional)</td>
                <td style="padding:0.5rem;">Flood barrier & passage assessment</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style="font-family:'Syne',sans-serif; color:#c8e0f8; margin-top:2rem;">Integration with Flood Segmentation</h3>
    <p style="color:#7fa8d4; font-size:0.85rem; line-height:1.7;">
    In the full pipeline (see <strong style="color:#c8e0f8;">Risk Classification</strong> module), 
    settlement bounding boxes are overlaid on flood segmentation masks to compute:
    </p>
    <ul style="color:#7fa8d4; font-size:0.85rem; line-height:2; margin-top:1rem;">
        <li><strong style="color:#c8e0f8;">Intersection Area</strong> — How much of each building/settlement is inundated</li>
        <li><strong style="color:#c8e0f8;">Impact Score</strong> — % of flooded area × building count → exposure index</li>
        <li><strong style="color:#c8e0f8;">Risk Tier</strong> — High/Medium/Low zones based on joint flood + settlement density</li>
    </ul>
    """, unsafe_allow_html=True)
