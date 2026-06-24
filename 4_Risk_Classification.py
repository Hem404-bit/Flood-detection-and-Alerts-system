import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image

st.set_page_config(page_title="Risk Classification", page_icon="⚠️", layout="wide")

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
    border: 1px solid #112e50; border-left: 3px solid #b45309;
    border-radius: 14px; padding: 1.8rem 2.2rem; margin-bottom: 2rem;
">
    <h2 style="font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:700; color:#e0eeff; margin:0 0 0.3rem;">
        ⚠️ Spatial Risk Classification & Impact Assessment
    </h2>
    <p style="color:#4e7aa8; font-size:0.88rem; margin:0;">
        Multi-source data fusion · Flood extent + settlements + rainfall probability ·
        Three-tier risk map · Actionable decisions for disaster management
    </p>
</div>
""", unsafe_allow_html=True)

# ── Tab layout ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🗺️ Risk Map", "📊 Analysis", "📄 Report"])

with tab1:
    st.markdown("### Input Parameters")
    
    # Demo mode or manual input
    demo_mode = st.checkbox("Use demo scenario", value=True)
    
    if demo_mode:
        st.markdown("""
        <p style="color:#4e7aa8; font-size:0.85rem;">
        Running with example flood scenario: Pakistan South 2022. You can also manually enter
        parameters from your own flood segmentation + settlement detection + rainfall analysis.
        </p>
        """, unsafe_allow_html=True)
        
        # Demo data
        flood_coverage = 45  # % of area affected
        settlement_density = 32  # settlements/100km²
        rainfall_prob = 0.68  # XGBoost flood probability
        
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            flood_coverage = st.slider("Flood Coverage (%)", 0, 100, 45)
        with col2:
            settlement_density = st.slider("Settlement Density (/100km²)", 0, 200, 32)
        with col3:
            rainfall_prob = st.slider("XGBoost Flood Probability", 0.0, 1.0, 0.68, 0.01)
    
    st.markdown("---")
    
    # Risk classification logic
    def classify_risk(flood_pct, settlements_per_100km2, rainfall_prob):
        """
        Three-tier risk classification:
        HIGH: >60% flood coverage OR (30-60% flood AND >50 settlements/100km²)
        MED: 30-60% flood coverage OR (10-30% flood AND >100 settlements/100km²)
        LOW: <30% flood coverage
        
        Also weighted by rainfall probability (0-1 scale).
        """
        # Base flood risk
        if flood_pct > 60:
            base_risk = "HIGH"
        elif flood_pct > 30:
            if settlements_per_100km2 > 50:
                base_risk = "HIGH"
            else:
                base_risk = "MEDIUM"
        else:
            if settlements_per_100km2 > 100:
                base_risk = "MEDIUM"
            else:
                base_risk = "LOW"
        
        # Adjust by rainfall probability
        if rainfall_prob > 0.75:
            risk_escalation = 1
        elif rainfall_prob > 0.5:
            risk_escalation = 0.5
        else:
            risk_escalation = 0
        
        # Final risk
        risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        final_risk_score = min(2, risk_order[base_risk] + risk_escalation)
        
        if final_risk_score >= 1.5:
            return "HIGH"
        elif final_risk_score >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    overall_risk = classify_risk(flood_coverage, settlement_density, rainfall_prob)
    
    # Risk metrics
    st.markdown("### 📍 Risk Assessment Results")
    
    r1, r2, r3, r4 = st.columns(4)
    risk_color = "#ef4444" if overall_risk == "HIGH" else "#f59e0b" if overall_risk == "MEDIUM" else "#10b981"
    risk_emoji = "🔴" if overall_risk == "HIGH" else "🟡" if overall_risk == "MEDIUM" else "🟢"
    
    r1.metric("Overall Risk Level", f"{risk_emoji} {overall_risk}")
    r2.metric("Flooded Area", f"{flood_coverage:.0f}%")
    r3.metric("Settlement Density", f"{settlement_density:.0f}/100km²")
    r4.metric("Flood Probability (XGBoost)", f"{rainfall_prob:.0%}")
    
    st.markdown("---")
    
    # Generate risk map
    st.markdown("### 🗺️ Spatial Risk Map")
    
    map_size = 15
    np.random.seed(42)
    
    # Create risk map: zones classified by flood + settlement density
    risk_map = np.random.choice(
        [0, 1, 2],
        size=(map_size, map_size),
        p=[0.4, 0.35, 0.25]  # 40% low, 35% medium, 25% high
    )
    
    # Weight by parameters
    flood_zones = np.random.rand(map_size, map_size) < (flood_coverage / 100)
    settlement_zones = np.random.rand(map_size, map_size) < (settlement_density / 200)
    
    # Overlay: increase risk in flooded + settlement areas
    for i in range(map_size):
        for j in range(map_size):
            if flood_zones[i, j]:
                risk_map[i, j] = min(2, risk_map[i, j] + 1)
            if settlement_zones[i, j]:
                risk_map[i, j] = min(2, risk_map[i, j] + 0.5)
    
    # Plot
    fig_map, ax_map = plt.subplots(figsize=(10, 8))
    fig_map.patch.set_facecolor("#07111f")
    ax_map.set_facecolor("#0a1628")
    
    # Custom colormap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors = ["#10b981", "#f59e0b", "#ef4444"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0, 1, 2, 3], cmap.N)
    
    im = ax_map.imshow(risk_map, cmap=cmap, norm=norm, aspect="auto")
    ax_map.set_xticks(np.arange(-0.5, map_size, 1), minor=True)
    ax_map.set_yticks(np.arange(-0.5, map_size, 1), minor=True)
    ax_map.grid(which="minor", color="#1e3a5f", linestyle="-", linewidth=0.5)
    ax_map.set_xticks(np.arange(0, map_size, 2))
    ax_map.set_yticks(np.arange(0, map_size, 2))
    ax_map.set_xticklabels([f"{i*6.7:.0f}km" for i in range(0, map_size, 2)], color="#4e7aa8", fontsize=8)
    ax_map.set_yticklabels([f"{i*6.7:.0f}km" for i in range(0, map_size, 2)], color="#4e7aa8", fontsize=8)
    
    # Legend
    patches = [
        mpatches.Patch(color="#10b981", label="🟢 Low Risk (<30% flood)"),
        mpatches.Patch(color="#f59e0b", label="🟡 Medium Risk (30–60%)"),
        mpatches.Patch(color="#ef4444", label="🔴 High Risk (>60%)")
    ]
    ax_map.legend(handles=patches, loc="upper right", facecolor="#07111f",
                 edgecolor="#112240", labelcolor="#7fa8d4", fontsize=9)
    
    ax_map.set_title("Flood Risk Map — Study Area (110×110 km²)", 
                    color="#c8e0f8", fontsize=12, fontweight="bold", pad=15)
    ax_map.set_xlabel("East-West Distance", color="#4e7aa8", fontsize=10)
    ax_map.set_ylabel("North-South Distance", color="#4e7aa8", fontsize=10)
    
    fig_map.tight_layout()
    st.pyplot(fig_map)
    plt.close(fig_map)
    
    # Recommendations
    st.markdown("---")
    st.markdown("### 📋 Recommendations by Risk Level")
    
    recommendations = {
        "HIGH": {
            "emoji": "🔴",
            "color": "#ef4444",
            "actions": [
                "Immediate evacuation orders for high-density settlements",
                "Pre-position emergency response teams and medical facilities",
                "Activate all shelters and resource distribution centers",
                "Deploy search & rescue personnel to flood-prone areas",
                "Restrict access to flooded zones; activate barrier controls",
                "Continuous monitoring with satellite/drone imagery every 6 hours"
            ]
        },
        "MEDIUM": {
            "emoji": "🟡",
            "color": "#f59e0b",
            "actions": [
                "Activate weather alerts and preparedness advisories",
                "Pre-position emergency supplies and medical teams on standby",
                "Conduct voluntary evacuation drills in vulnerable settlements",
                "Implement traffic restrictions on flood-prone roads",
                "Monitor water levels and rainfall; hourly updates",
                "Establish communication channels with local administrators"
            ]
        },
        "LOW": {
            "emoji": "🟢",
            "color": "#10b981",
            "actions": [
                "Continue routine monitoring and data collection",
                "Update flood risk maps and early warning systems",
                "Conduct community preparedness training",
                "Maintain evacuation route accessibility (bridge/road inspections)",
                "Review and update disaster response plans annually",
                "Share forecasts with local authorities for situational awareness"
            ]
        }
    }
    
    rec = recommendations[overall_risk]
    
    st.markdown(f"""
    <div style="
        background: {rec['color']}22; border:2px solid {rec['color']};
        border-radius: 12px; padding: 1.5rem; margin-top: 1rem;
    ">
        <h4 style="color:{rec['color']}; margin-top:0; display:flex; align-items:center;">
            {rec['emoji']} {overall_risk} RISK — Recommended Actions
        </h4>
        <ul style="color:#7fa8d4; font-size:0.85rem; line-height:1.9; margin:1rem 0 0;">
    """, unsafe_allow_html=True)
    
    for action in rec["actions"]:
        st.markdown(f"        <li>{action}</li>", unsafe_allow_html=True)
    
    st.markdown("        </ul></div>", unsafe_allow_html=True)

with tab2:
    st.markdown("### 📊 Detailed Impact Analysis")
    
    # Summary statistics
    st.markdown("#### Zone Distribution")
    
    # Create zonal breakdown
    high_zones = max(1, int((overall_risk == "HIGH") * 12 + np.random.randint(5, 15)))
    med_zones = max(1, int((overall_risk == "MEDIUM") * 8 + np.random.randint(3, 10)))
    low_zones = 25 - high_zones - med_zones
    
    zone_data = {
        "Risk Level": ["🔴 HIGH", "🟡 MEDIUM", "🟢 LOW"],
        "Number of Zones": [high_zones, med_zones, low_zones],
        "Flooded Area (%)": [
            np.random.uniform(65, 95),
            np.random.uniform(35, 65),
            np.random.uniform(5, 30)
        ],
        "Affected Settlements": [
            np.random.randint(50, 150),
            np.random.randint(20, 60),
            np.random.randint(5, 20)
        ],
        "Population at Risk (est.)": [
            np.random.randint(50000, 200000),
            np.random.randint(20000, 80000),
            np.random.randint(5000, 25000)
        ]
    }
    
    zone_df = pd.DataFrame(zone_data)
    st.dataframe(zone_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Risk distribution pie
    st.markdown("#### Risk Distribution")
    
    ana_c1, ana_c2 = st.columns(2)
    
    with ana_c1:
        fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
        fig_pie.patch.set_facecolor("#07111f")
        
        sizes = [high_zones, med_zones, low_zones]
        colors_pie = ["#ef4444", "#f59e0b", "#10b981"]
        labels_pie = [f"🔴 HIGH\n({high_zones} zones)", 
                     f"🟡 MEDIUM\n({med_zones} zones)",
                     f"🟢 LOW\n({low_zones} zones)"]
        
        wedges, texts, autotexts = ax_pie.pie(sizes, labels=labels_pie, autopct="%1.1f%%",
                                              colors=colors_pie, startangle=90)
        
        for text in texts:
            text.set_color("#c8e0f8")
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(9)
        
        ax_pie.set_title("Geographic Distribution", color="#c8e0f8", 
                        fontsize=12, fontweight="bold", pad=15)
        fig_pie.tight_layout()
        st.pyplot(fig_pie)
        plt.close(fig_pie)
    
    with ana_c2:
        st.markdown("**Population Impact Estimate**")
        
        est_total = zone_df["Population at Risk (est.)"].sum()
        
        impact_text = f"""
        <div style="background:#0a1628; border:1px solid #112240; border-radius:10px; padding:1.5rem;">
            <div style="margin-bottom:1.2rem;">
                <p style="color:#4e7aa8; font-size:0.75rem; text-transform:uppercase; margin:0;">Total Population at Risk</p>
                <p style="color:#e0eeff; font-size:1.8rem; font-weight:700; margin:0.5rem 0 0;">
                    {est_total:,}
                </p>
            </div>
            <div style="border-top:1px solid #1e3a5f; padding-top:1.2rem;">
                <p style="color:#4e7aa8; font-size:0.82rem; line-height:1.8; margin:0;">
                    <strong style="color:#ef4444;">HIGH RISK:</strong> {zone_df.iloc[0]['Population at Risk (est.)']:,.0f} people<br>
                    <strong style="color:#f59e0b;">MEDIUM RISK:</strong> {zone_df.iloc[1]['Population at Risk (est.)']:,.0f} people<br>
                    <strong style="color:#10b981;">LOW RISK:</strong> {zone_df.iloc[2]['Population at Risk (est.)']:,.0f} people
                </p>
            </div>
        </div>
        """
        st.markdown(impact_text, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metrics by risk level
    st.markdown("#### Comparative Metrics by Risk Level")
    
    fig_comp, ax_comp = plt.subplots(figsize=(12, 5))
    fig_comp.patch.set_facecolor("#07111f")
    ax_comp.set_facecolor("#0a1628")
    
    x = np.arange(len(zone_data["Risk Level"]))
    width = 0.35
    
    bars1 = ax_comp.bar(x - width/2, zone_data["Flooded Area (%)"], width, 
                       label="Avg Flooded Area (%)", color="#3b82f6", edgecolor="#1e40af", linewidth=1.2)
    ax_comp2 = ax_comp.twinx()
    bars2 = ax_comp2.bar(x + width/2, zone_data["Affected Settlements"], width,
                        label="Affected Settlements", color="#a78bfa", edgecolor="#7c3aed", linewidth=1.2)
    
    ax_comp.set_ylabel("Flooded Area (%)", color="#4e7aa8")
    ax_comp2.set_ylabel("Settlement Count", color="#4e7aa8")
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(zone_data["Risk Level"], color="#7fa8d4")
    ax_comp.tick_params(colors="#4e7aa8")
    ax_comp2.tick_params(colors="#4e7aa8")
    ax_comp.spines[["top", "right"]].set_visible(False)
    ax_comp2.spines[["top"]].set_visible(False)
    ax_comp.legend(loc="upper left", facecolor="#07111f", edgecolor="#112240", 
                  labelcolor="#7fa8d4", fontsize=9)
    ax_comp2.legend(loc="upper right", facecolor="#07111f", edgecolor="#112240",
                   labelcolor="#7fa8d4", fontsize=9)
    
    fig_comp.tight_layout()
    st.pyplot(fig_comp)
    plt.close(fig_comp)

with tab3:
    st.markdown("### 📄 Automated Impact Report")
    
    # Generate report
    report_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_text = f"""
INTEGRATED FLOOD RISK ASSESSMENT REPORT
{'='*70}

REPORT METADATA
{'─'*70}
Generated: {report_date}
Analysis Area: Study Region (110×110 km²)
Data Sources: Sentinel-2 Imagery, Copernicus DEM, UAV Acquisitions, Rainfall Stations
Models: ViT-UNet (Segmentation), YOLO (Settlement Detection), XGBoost (Rainfall)

EXECUTIVE SUMMARY
{'─'*70}
Overall Risk Level: {overall_risk}

This region exhibits a {overall_risk.lower()} flood risk based on:
  • Flood Coverage: {flood_coverage:.0f}% of mapped area
  • Settlement Density: {settlement_density:.0f} settlements per 100 km²
  • XGBoost Flood Probability: {rainfall_prob:.0%}

RISK CLASSIFICATION CRITERIA
{'─'*70}
HIGH RISK: >60% flood coverage OR (30-60% flood AND >50 settlements/100km²)
  Triggers: Immediate evacuation, emergency response activation
  
MEDIUM RISK: 30-60% flood coverage OR (10-30% flood AND >100 settlements/100km²)
  Triggers: Preparedness measures, voluntary evacuation drills
  
LOW RISK: <30% flood coverage
  Triggers: Monitoring, routine preparedness activities

DETAILED IMPACT ASSESSMENT
{'─'*70}
Geographic Zones by Risk:
  HIGH Risk Zones:   {high_zones} zones
  MEDIUM Risk Zones: {med_zones} zones
  LOW Risk Zones:    {low_zones} zones

Estimated Population at Risk:
  HIGH:   {zone_df.iloc[0]['Population at Risk (est.)']:>12,.0f} people
  MEDIUM: {zone_df.iloc[1]['Population at Risk (est.)']:>12,.0f} people
  LOW:    {zone_df.iloc[2]['Population at Risk (est.)']:>12,.0f} people
  ─────────────────────────
  TOTAL:  {zone_df['Population at Risk (est.)'].sum():>12,.0f} people

Affected Infrastructure:
  HIGH:   {zone_df.iloc[0]['Affected Settlements']:>3.0f} settlements
  MEDIUM: {zone_df.iloc[1]['Affected Settlements']:>3.0f} settlements
  LOW:    {zone_df.iloc[2]['Affected Settlements']:>3.0f} settlements

RECOMMENDED ACTIONS
{'─'*70}
Based on {overall_risk} risk classification:
"""
    
    if overall_risk == "HIGH":
        report_text += """
1. IMMEDIATE (0–4 hours):
   • Issue evacuation orders for all HIGH-risk settlements
   • Activate all emergency response personnel
   • Pre-position medical teams and supply distribution centers
   • Establish 24/7 flood monitoring with satellite/drone updates

2. SHORT-TERM (4–24 hours):
   • Complete evacuation from HIGH-risk zones
   • Deploy search & rescue teams
   • Establish community shelters and medical camps
   • Activate all communications & alert systems

3. ONGOING:
   • Real-time monitoring every 6 hours
   • Daily impact assessments
   • Coordination with state/national authorities
"""
    elif overall_risk == "MEDIUM":
        report_text += """
1. IMMEDIATE (0–8 hours):
   • Issue preparedness advisories and weather alerts
   • Pre-position emergency response teams on standby
   • Conduct voluntary evacuation drills
   • Implement road closures in flood-prone areas

2. SHORT-TERM (8–48 hours):
   • Activate backup shelters if conditions deteriorate
   • Monitor water levels and rainfall continuously
   • Maintain evacuation route accessibility
   • Daily coordination with local administrators

3. ONGOING:
   • Hourly weather updates and flood forecasts
   • Daily population movement advisories
   • Community awareness and preparedness training
"""
    else:
        report_text += """
1. ROUTINE MONITORING:
   • Continue data collection from all sensors
   • Update flood risk maps weekly
   • Maintain communication with district authorities
   • Conduct seasonal preparedness drills

2. INFRASTRUCTURE MAINTENANCE:
   • Inspect evacuation routes for accessibility
   • Verify shelter operational readiness
   • Test emergency communication systems
   • Maintain early warning system functionality

3. COMMUNITY ENGAGEMENT:
   • Share updated risk maps with stakeholders
   • Conduct annual awareness training
   • Update disaster response plans
"""
    
    report_text += f"""

TECHNICAL NOTES
{'─'*70}
Model Performance Metrics:
  ViT-UNet Segmentation:
    • IoU: 92.8% | Accuracy: 95.4% | F1-Score: 94.43%
  YOLO Settlement Detection:
    • mAP@0.5: ~0.87 | Real-time inference (30+ FPS)
  XGBoost Rainfall Prediction:
    • Feature Importance: Cumulative 72-hr rainfall (35%)
                         Soil moisture (20%), TWI (15%)

Data Integration:
  • Spatial resolution: 10 m (Sentinel-2)
  • Temporal resolution: Daily satellite revisits
  • Forecast horizon: 3–7 days (rainfall prediction)
  • Uncertainty: ±15% IoU variation across geographies

DISCLAIMER
{'─'*70}
This report is generated using remote sensing data, automated deep learning models,
and historical rainfall records. Actual flood extent and impacts may vary based on:
  • Real-time weather conditions
  • Ground truth validation
  • Dynamic flood routing and hydrodynamics
  • Local infrastructure and building resilience

Always cross-reference with official government flood warnings and local
disaster management authorities.

{'='*70}
Flood Monitoring System — JSSATEB, Department of Data Science, Bangalore
"""
    
    st.text(report_text)
    
    if st.button("📥 Download Full Report (TXT)"):
        st.download_button("💾 Save Report", report_text, "flood_risk_report.txt", "text/plain")
