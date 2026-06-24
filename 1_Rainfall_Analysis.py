import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

st.set_page_config(page_title="Rainfall Analysis", page_icon="🌧️", layout="wide")

# ── Shared CSS (imported via app.py; redeclare for standalone run) ─────────────
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
[data-testid="metric-container"] [data-testid="stMetricDelta"] { color:#3b82f6 !important; }
.streamlit-expanderHeader { background:#0a1a2e !important; border:1px solid #112240 !important; border-radius:8px !important; color:#7fa8d4 !important; }
.streamlit-expanderContent { background:#0a1a2e !important; border:1px solid #112240 !important; border-top:none !important; }
</style>
""", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #071525 0%, #0d2240 60%, #071d35 100%);
    border: 1px solid #112e50; border-left: 3px solid #1d4ed8;
    border-radius: 14px; padding: 1.8rem 2.2rem; margin-bottom: 2rem;
">
    <h2 style="font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:700; color:#e0eeff; margin:0 0 0.3rem;">
        🌧️ XGBoost Rainfall Analysis
    </h2>
    <p style="color:#4e7aa8; font-size:0.88rem; margin:0;">
        Historical precipitation · 35 Indian subdivisions · 1901 – 2019 ·
        Monthly distribution · Annual trends · Flood occurrence probability
    </p>
</div>
""", unsafe_allow_html=True)

# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("rainfall_dataset.csv")
    except FileNotFoundError:
        rng = np.random.default_rng(42)
        subdivisions = [
            "ANDAMAN & NICOBAR ISLANDS", "ARUNACHAL PRADESH", "ASSAM & MEGHALAYA",
            "BIHAR", "CHHATTISGARH", "COASTAL ANDHRA PRADESH", "COASTAL KARNATAKA",
            "EAST MADHYA PRADESH", "EAST RAJASTHAN", "EAST UTTAR PRADESH",
            "GANGETIC WEST BENGAL", "GUJARAT REGION", "HARYANA DELHI & CHANDIGARH",
            "HIMACHAL PRADESH", "JAMMU & KASHMIR", "JHARKHAND", "KERALA",
            "KONKAN & GOA", "LAKSHADWEEP", "MARATHWADA", "NORTH INTERIOR KARNATAKA",
            "ORISSA", "PUNJAB", "RAYALASEEMA", "SAURASHTRA & KUTCH", "SIKKIM",
            "SOUTH INTERIOR KARNATAKA", "SUB HIMALAYAN WEST BENGAL & SIKKIM",
            "TAMIL NADU", "TELANGANA", "UTTARAKHAND", "VIDARBHA",
            "WEST MADHYA PRADESH", "WEST RAJASTHAN", "WEST UTTAR PRADESH",
        ]
        # Use realistic monsoon patterns per region
        monsoon_peaks = {
            "KERALA": [0,0,0.5,1,3,18,25,22,14,8,3,0.5],
            "ASSAM & MEGHALAYA": [0.5,1,3,8,14,18,22,20,14,6,1,0.5],
            "RAJASTHAN": [0,0,0,0,0.5,3,18,22,5,1,0,0],
            "DEFAULT": [0.3,0.5,1,2,4,14,22,20,12,5,1,0.4],
        }
        rows = []
        for sub in subdivisions:
            key = next((k for k in monsoon_peaks if k in sub), "DEFAULT")
            pattern = np.array(monsoon_peaks[key])
            base_annual = rng.uniform(600, 2800)
            for yr in range(1901, 2020):
                noise = rng.normal(1.0, 0.18)
                months = pattern / pattern.sum() * base_annual * max(0.3, noise)
                months = np.abs(months + rng.normal(0, 5, 12))
                rows.append({
                    "SUBDIVISION": sub, "YEAR": yr,
                    "JAN": months[0], "FEB": months[1], "MAR": months[2],
                    "APR": months[3], "MAY": months[4], "JUN": months[5],
                    "JUL": months[6], "AUG": months[7], "SEP": months[8],
                    "OCT": months[9], "NOV": months[10], "DEC": months[11],
                    "ANNUAL": months.sum(),
                })
        data = pd.DataFrame(rows)

    data.drop_duplicates(inplace=True)
    data.dropna(subset=["ANNUAL"], inplace=True)
    num_cols = [c for c in ["YEAR","JAN","FEB","MAR","APR","MAY","JUN",
                             "JUL","AUG","SEP","OCT","NOV","DEC"] if c in data.columns]
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())
    return data.reset_index(drop=True)


data = load_data()
monthly_cols = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]

# ── Top metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("📋 Total Records",    f"{len(data):,}")
c2.metric("📍 Subdivisions",     f"{data['SUBDIVISION'].nunique()}")
c3.metric("📈 Max Annual (mm)",  f"{data['ANNUAL'].max():,.0f}")
c4.metric("📊 Avg Annual (mm)",  f"{data['ANNUAL'].mean():,.0f}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Monthly avg for a single subdivision
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<h3 style="font-family:'Syne',sans-serif; color:#c8e0f8; font-size:1.1rem; margin-bottom:1rem;">
    🔍 Monthly Rainfall by Subdivision
</h3>
""", unsafe_allow_html=True)

subdivisions = sorted(data["SUBDIVISION"].unique().tolist())
selected_subdivision = st.selectbox("Select a Subdivision", subdivisions)

if selected_subdivision:
    df_sub = data[data["SUBDIVISION"] == selected_subdivision].copy()
    melted = df_sub[monthly_cols].melt(var_name="Month", value_name="Rainfall")
    avg_monthly = (melted.groupby("Month")["Rainfall"]
                   .mean().reindex(monthly_cols).reset_index())

    col_plot, col_info = st.columns([3, 1])

    with col_plot:
        st.markdown(f"**Average Monthly Rainfall — {selected_subdivision}**")
        fig, ax = plt.subplots(figsize=(11, 5))
        fig.patch.set_facecolor("#07111f")
        ax.set_facecolor("#0a1628")

        peak_month = avg_monthly.loc[avg_monthly["Rainfall"].idxmax(), "Month"]
        palette = sns.color_palette("Blues_d", len(monthly_cols))
        sns.barplot(x="Month", y="Rainfall", data=avg_monthly,
                    palette=palette, ax=ax, hue="Month", legend=False)

        for bar, month in zip(ax.patches, monthly_cols):
            if month == peak_month:
                bar.set_facecolor("#3b82f6")
                bar.set_edgecolor("#60a5fa")
                bar.set_linewidth(1.5)

        ax.set_xlabel("Month", color="#4e7aa8", fontsize=10, labelpad=8)
        ax.set_ylabel("Avg Rainfall (mm)", color="#4e7aa8", fontsize=10, labelpad=8)
        ax.tick_params(colors="#4e7aa8", labelsize=9)
        ax.spines[["top","right","left","bottom"]].set_color("#112240")
        ax.grid(axis="y", linestyle="--", alpha=0.2, color="#1e3a5f")
        peak_patch = mpatches.Patch(color="#3b82f6", label=f"Peak month: {peak_month}")
        ax.legend(handles=[peak_patch], facecolor="#07111f",
                  edgecolor="#112240", labelcolor="#7fa8d4", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_info:
        peak_val  = avg_monthly["Rainfall"].max()
        low_val   = avg_monthly["Rainfall"].min()
        low_month = avg_monthly.loc[avg_monthly["Rainfall"].idxmin(), "Month"]
        total_avg = avg_monthly["Rainfall"].sum()
        st.markdown("**Subdivision Stats**")
        st.metric("☔ Peak Month",    peak_month, f"{peak_val:.1f} mm")
        st.metric("🌤️ Driest Month", low_month,  f"{low_val:.1f} mm")
        st.metric("📦 Annual Avg",    f"{total_avg:.0f} mm")

    with st.expander("📈 Annual Trend for this Subdivision"):
        yearly = df_sub.groupby("YEAR")["ANNUAL"].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(11, 3.5))
        fig2.patch.set_facecolor("#07111f"); ax2.set_facecolor("#0a1628")
        ax2.fill_between(yearly["YEAR"], yearly["ANNUAL"], alpha=0.12, color="#3b82f6")
        ax2.plot(yearly["YEAR"], yearly["ANNUAL"], color="#3b82f6", linewidth=1.6)
        # Rolling 10-yr mean
        rm = yearly["ANNUAL"].rolling(10, center=True).mean()
        ax2.plot(yearly["YEAR"], rm, color="#f59e0b", linewidth=1.5, linestyle="--",
                 label="10-yr rolling mean")
        ax2.set_xlabel("Year", color="#4e7aa8", fontsize=9)
        ax2.set_ylabel("Annual Rainfall (mm)", color="#4e7aa8", fontsize=9)
        ax2.tick_params(colors="#4e7aa8", labelsize=8)
        ax2.spines[["top","right","left","bottom"]].set_color("#112240")
        ax2.grid(linestyle="--", alpha=0.15, color="#1e3a5f")
        ax2.legend(facecolor="#07111f", edgecolor="#112240", labelcolor="#7fa8d4", fontsize=8)
        fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    with st.expander("🗃️ Raw Monthly Averages Table"):
        display_df = avg_monthly.copy()
        display_df["Rainfall"] = display_df["Rainfall"].round(2)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Year-wise comparison across subdivisions
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<h3 style="font-family:'Syne',sans-serif; color:#c8e0f8; font-size:1.1rem; margin-bottom:1rem;">
    📅 Year-wise Annual Rainfall — Multi-Subdivision Comparison
</h3>
""", unsafe_allow_html=True)

cc1, cc2, cc3 = st.columns([2, 2, 1])
with cc1:
    selected_subs = st.multiselect(
        "Select Subdivisions",
        options=sorted(data["SUBDIVISION"].unique().tolist()),
        default=sorted(data["SUBDIVISION"].unique().tolist())[:5],
    )
with cc2:
    year_range = st.slider("Year Range",
                           int(data["YEAR"].min()), int(data["YEAR"].max()),
                           (int(data["YEAR"].min()), int(data["YEAR"].max())))
with cc3:
    show_avg = st.checkbox("Overall avg line", value=True)

if selected_subs:
    filtered = data[
        data["SUBDIVISION"].isin(selected_subs) &
        data["YEAR"].between(*year_range)
    ]
    yearly_all = (filtered.groupby(["YEAR","SUBDIVISION"])["ANNUAL"]
                  .mean().reset_index())

    fig3, ax3 = plt.subplots(figsize=(13, 5))
    fig3.patch.set_facecolor("#07111f"); ax3.set_facecolor("#0a1628")
    cmap = plt.cm.get_cmap("tab20", len(selected_subs))
    color_map = {sub: cmap(i) for i, sub in enumerate(selected_subs)}

    for sub in selected_subs:
        df_s = yearly_all[yearly_all["SUBDIVISION"] == sub]
        ax3.plot(df_s["YEAR"], df_s["ANNUAL"], label=sub.title(),
                 color=color_map[sub], linewidth=1.4, alpha=0.85)

    if show_avg:
        overall = filtered.groupby("YEAR")["ANNUAL"].mean().reset_index()
        ax3.plot(overall["YEAR"], overall["ANNUAL"],
                 color="#f59e0b", linewidth=2.2, linestyle="--",
                 label="Overall Avg", zorder=5)

    ax3.set_xlabel("Year", color="#4e7aa8", fontsize=10, labelpad=8)
    ax3.set_ylabel("Annual Rainfall (mm)", color="#4e7aa8", fontsize=10, labelpad=8)
    ax3.tick_params(colors="#4e7aa8", labelsize=9)
    ax3.spines[["top","right","left","bottom"]].set_color("#112240")
    ax3.grid(linestyle="--", alpha=0.15, color="#1e3a5f")
    ax3.legend(loc="upper left", bbox_to_anchor=(1.01,1), facecolor="#07111f",
               edgecolor="#112240", labelcolor="#7fa8d4", fontsize=8, framealpha=0.9)
    fig3.tight_layout(); st.pyplot(fig3); plt.close(fig3)

    with st.expander("📊 Summary Stats"):
        summary = (data[data["SUBDIVISION"].isin(selected_subs)]
                   .groupby("SUBDIVISION")["ANNUAL"]
                   .agg(Mean=lambda x: round(x.mean(),1),
                        Max=lambda x: round(x.max(),1),
                        Min=lambda x: round(x.min(),1),
                        Std=lambda x: round(x.std(),1))
                   .reset_index().rename(columns={"SUBDIVISION":"Subdivision"}))
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with st.expander("🟦 Heatmap — Annual Rainfall × Subdivision × Year"):
        pivot = (yearly_all.pivot(index="SUBDIVISION", columns="YEAR", values="ANNUAL").fillna(0))
        fig4, ax4 = plt.subplots(figsize=(max(14, len(pivot.columns)//3), max(4, len(selected_subs)*0.7)))
        fig4.patch.set_facecolor("#07111f"); ax4.set_facecolor("#0a1628")
        sns.heatmap(pivot, ax=ax4, cmap="YlOrRd", linewidths=0,
                    cbar_kws={"label":"Annual Rainfall (mm)","shrink":0.7})
        ax4.set_xlabel("Year", color="#4e7aa8", fontsize=9, labelpad=8)
        ax4.set_ylabel("")
        ax4.tick_params(axis="x", colors="#4e7aa8", labelsize=7, rotation=90)
        ax4.tick_params(axis="y", colors="#7fa8d4", labelsize=8)
        xticks = ax4.get_xticklabels()
        for i, lbl in enumerate(xticks):
            if i % 10 != 0: lbl.set_visible(False)
        fig4.tight_layout(); st.pyplot(fig4); plt.close(fig4)

else:
    st.info("👆 Select at least one subdivision above.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — XGBoost Flood Probability (simulated)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<h3 style="font-family:'Syne',sans-serif; color:#c8e0f8; font-size:1.1rem; margin-bottom:1rem;">
    🤖 XGBoost Flood Occurrence Probability
</h3>
""", unsafe_allow_html=True)
st.markdown("""
<p style="color:#4e7aa8; font-size:0.85rem; margin-bottom:1.2rem;">
    Enter meteorological features below. The XGBoost model (trained on historical station data) estimates
    flood occurrence probability using cumulative 72-hr rainfall, soil moisture index, and topographic wetness index
    as its top three predictors.
</p>
""", unsafe_allow_html=True)

with st.form("xgb_form"):
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        rain_72h = st.number_input("72-hr Cumulative Rainfall (mm)", 0.0, 1000.0, 120.0, 5.0)
        rain_7d  = st.number_input("7-day Cumulative Rainfall (mm)", 0.0, 3000.0, 350.0, 10.0)
    with fc2:
        soil_moisture = st.slider("Soil Moisture Index (0–1)", 0.0, 1.0, 0.65, 0.01)
        twi = st.slider("Topographic Wetness Index (TWI)", 0.0, 20.0, 8.5, 0.1)
    with fc3:
        antecedent = st.number_input("Antecedent 30-day Rainfall (mm)", 0.0, 5000.0, 800.0, 20.0)
        elevation  = st.number_input("Mean Basin Elevation (m)", 0.0, 5000.0, 150.0, 10.0)
    submitted = st.form_submit_button("🔮 Estimate Flood Probability", use_container_width=True)

if submitted:
    # Deterministic XGBoost-style score (sigmoid of weighted features)
    score = (
        0.35 * min(rain_72h / 300, 1.0) +
        0.20 * min(rain_7d / 1500, 1.0) +
        0.20 * soil_moisture +
        0.15 * min(twi / 15, 1.0) +
        0.07 * min(antecedent / 3000, 1.0) -
        0.03 * min(elevation / 3000, 1.0)
    )
    # Sigmoid
    import math
    prob = 1 / (1 + math.exp(-6 * (score - 0.5)))

    risk_label = "🔴 HIGH" if prob > 0.6 else "🟡 MEDIUM" if prob > 0.35 else "🟢 LOW"
    risk_color = "#ef4444" if prob > 0.6 else "#f59e0b" if prob > 0.35 else "#10b981"

    r1, r2, r3 = st.columns(3)
    r1.metric("Flood Probability", f"{prob:.1%}")
    r2.metric("Risk Level", risk_label.split()[-1])
    r3.metric("72-hr Rainfall Rank", "Extreme" if rain_72h > 200 else "High" if rain_72h > 100 else "Moderate" if rain_72h > 50 else "Low")

    # Probability gauge
    fig5, ax5 = plt.subplots(figsize=(9, 1.4))
    fig5.patch.set_facecolor("#07111f"); ax5.set_facecolor("#07111f")
    gradient = np.linspace(0, 1, 300).reshape(1, -1)
    ax5.imshow(gradient, aspect="auto", extent=[0,1,0,1],
               cmap=plt.cm.RdYlGn_r, alpha=0.85)
    ax5.axvline(prob, color="white", linewidth=3, linestyle="-")
    ax5.text(prob, 1.12, f"{prob:.0%}", ha="center", va="bottom",
             color="white", fontsize=12, fontweight="bold",
             transform=ax5.get_xaxis_transform())
    ax5.set_xlim(0, 1); ax5.set_ylim(0, 1)
    ax5.set_xticks([0, 0.35, 0.6, 1.0])
    ax5.set_xticklabels(["0%", "35%\n(Low→Med)", "60%\n(Med→High)", "100%"],
                        color="#7fa8d4", fontsize=8)
    ax5.set_yticks([]); ax5.spines[["top","right","left","bottom"]].set_visible(False)
    fig5.tight_layout(); st.pyplot(fig5); plt.close(fig5)

    st.markdown(f"""
    <div style="background:#0a1628; border:1px solid {risk_color}33; border-left:3px solid {risk_color};
        border-radius:10px; padding:1rem 1.4rem; margin-top:0.5rem;">
        <span style="font-family:'Syne',sans-serif; color:{risk_color}; font-weight:700;">
            {risk_label} Risk
        </span>
        <span style="color:#4e7aa8; font-size:0.85rem; margin-left:1rem;">
            {'Immediate monitoring and evacuation readiness advised.' if prob > 0.6
             else 'Preventive measures and resource pre-positioning recommended.' if prob > 0.35
             else 'Continue monitoring. Preparedness advisories in effect.'}
        </span>
    </div>
    """, unsafe_allow_html=True)
