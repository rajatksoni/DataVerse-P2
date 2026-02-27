"""
app.py — Risk Underwriting Terminal  (Dark Quant Mode)
======================================================
Bento-grid Streamlit dashboard with:
  • Col 1  : Real-time risk simulator (sliders + verdict)
  • Col 2  : SHAP waterfall — Explainable AI
  • Col 3  : Risk-density KDE curve with applicant marker
  • Row 2  : Concept-drift line chart + Income-paradox bar chart
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ──────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Underwriting Terminal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────
# DARK QUANT TERMINAL — CSS
# ──────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #F8FAFC;
    }

    /* ── Canvas ── */
    .stApp { background-color: #0B1120; }
    header[data-testid="stHeader"] { background: transparent; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    /* ── Bento Card ── */
    .bento-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 14px;
        transition: border-color .3s ease, box-shadow .3s ease;
    }
    .bento-card:hover {
        border-color: #475569;
        box-shadow: 0 0 24px rgba(45,212,191,0.06);
    }

    /* ── Card typography ── */
    .card-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        color: #94A3B8;
        margin-bottom: 2px;
    }
    .card-title {
        font-size: 17px;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 10px;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 30px 38px;
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #2DD4BF, #FB7185, #2DD4BF);
    }
    .hero-title {
        font-size: 24px;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 4px;
    }
    .hero-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #94A3B8;
    }

    /* ── KPI chips ── */
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 26px;
        font-weight: 700;
    }
    .kpi-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        font-weight: 500;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Risk gauge ── */
    .risk-output {
        text-align: center;
        padding: 18px;
        border-radius: 12px;
        margin-top: 6px;
    }
    .risk-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 46px;
        font-weight: 800;
        line-height: 1.1;
    }
    .risk-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 4px;
    }

    /* ── Slider labels ── */
    div[data-testid="stSlider"] label {
        font-size: 12px !important;
        font-weight: 500 !important;
        color: #94A3B8 !important;
    }

    /* ── Plotly spacing ── */
    .stPlotlyChart { margin-top: -8px; }

    /* ── Pulse dot ── */
    .pulse-dot {
        display: inline-block;
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #2DD4BF;
        margin-right: 8px;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%,100% { opacity: 1; box-shadow: 0 0 0 0 rgba(45,212,191,0.5); }
        50%     { opacity: .5; box-shadow: 0 0 8px 2px rgba(45,212,191,0.25); }
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
# DATA · MODEL · EXPLAINER  (single cached init)
# ──────────────────────────────────────────────────────
FEATURES = ['loan_amnt', 'annual_inc', 'dti', 'int_rate', 'revol_util']
FEATURE_DISPLAY = {
    'loan_amnt':  'Loan Amount',
    'annual_inc': 'Annual Income',
    'dti':        'DTI Ratio',
    'int_rate':   'Interest Rate',
    'revol_util': 'Revolving Util',
}

@st.cache_resource
def init_app():
    """Load data, model, SHAP explainer, and precompute baseline probabilities."""
    df = pd.read_csv("dashboard_data.csv")
    model = joblib.load("rf_model.pkl")
    explainer = shap.TreeExplainer(model)
    feats = ['loan_amnt', 'annual_inc', 'dti', 'int_rate', 'revol_util']
    baseline = model.predict_proba(df[feats].values.astype("float32"))[:, 1]
    return df, model, explainer, baseline

df, model, explainer, baseline_probas = init_app()

# ──────────────────────────────────────────────────────
# HERO BANNER
# ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🛡️ Risk Underwriting Terminal</div>
    <div class="hero-sub">
        <span class="pulse-dot"></span>
        LIVE &nbsp;·&nbsp; SHAP Explainable AI &nbsp;·&nbsp; Concept Drift Detector
        &nbsp;·&nbsp; RF 100T / D7 / Balanced &nbsp;·&nbsp; 1 M+ Loans
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
# KPI ROW
# ──────────────────────────────────────────────────────
total_loans  = len(df)
default_rate = df['target'].mean() * 100
avg_loan     = df['loan_amnt'].mean()
avg_income   = df['annual_inc'].mean()

k1, k2, k3, k4 = st.columns(4)
for col, val, label, accent in [
    (k1, f"{total_loans:,}",     "Sample Size",  "#F8FAFC"),
    (k2, f"{default_rate:.1f}%", "Default Rate",  "#FB7185"),
    (k3, f"${avg_loan:,.0f}",   "Avg Loan",      "#F8FAFC"),
    (k4, f"${avg_income:,.0f}",  "Avg Income",    "#2DD4BF"),
]:
    col.markdown(f"""
    <div class="bento-card" style="text-align:center; padding:14px 10px;">
        <div class="kpi-value" style="color:{accent};">{val}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  ROW 1 — Simulator  |  SHAP Waterfall  |  Density
# ══════════════════════════════════════════════════════
col_sim, col_shap, col_kde = st.columns([1, 1.35, 1.35], gap="medium")

# ──────────────────── COLUMN 1: Simulator ─────────────
with col_sim:
    st.markdown("""
    <div class="bento-card">
        <div class="card-header">Risk Simulator</div>
        <div class="card-title">Applicant Profile</div>
    </div>
    """, unsafe_allow_html=True)

    slider_cfg = {
        'loan_amnt': ('💰 Loan Amount ($)',
                      int(df['loan_amnt'].min()),
                      int(df['loan_amnt'].max()),
                      int(df['loan_amnt'].median()), 500),
        'annual_inc': ('📊 Annual Income ($)',
                       int(max(df['annual_inc'].min(), 0)),
                       int(min(df['annual_inc'].max(), 500_000)),
                       int(df['annual_inc'].median()), 1000),
        'dti': ('📐 Debt-to-Income',
                float(max(df['dti'].min(), 0)),
                float(min(df['dti'].max(), 60)),
                float(df['dti'].median()), 0.5),
        'int_rate': ('📈 Interest Rate (%)',
                     float(df['int_rate'].min()),
                     float(df['int_rate'].max()),
                     float(df['int_rate'].median()), 0.25),
        'revol_util': ('🔄 Revolving Util (%)',
                       0.0,
                       float(min(df['revol_util'].max(), 150)),
                       float(df['revol_util'].median()), 0.5),
    }

    inputs = {}
    for feat, (lbl, mn, mx, dv, stp) in slider_cfg.items():
        inputs[feat] = st.slider(lbl, min_value=mn, max_value=mx,
                                 value=dv, step=stp)

    # — Prediction —
    input_arr = np.array([[inputs[f] for f in FEATURES]], dtype="float32")
    proba     = model.predict_proba(input_arr)[0][1]
    risk_pct  = proba * 100

    if risk_pct > 15:
        clr, bg, bdr = "#FB7185", "rgba(251,113,133,0.08)", "rgba(251,113,133,0.30)"
        verdict = "⚠️  HIGH RISK"
    else:
        clr, bg, bdr = "#2DD4BF", "rgba(45,212,191,0.08)", "rgba(45,212,191,0.30)"
        verdict = "✅  LOW RISK"

    st.markdown(f"""
    <div class="risk-output" style="background:{bg}; border:2px solid {bdr};">
        <div class="risk-value" style="color:{clr};">{risk_pct:.1f}%</div>
        <div class="risk-label" style="color:{clr};">{verdict}</div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────── COLUMN 2: SHAP Waterfall ────────────
with col_shap:
    st.markdown("""
    <div class="bento-card">
        <div class="card-header">Explainable AI</div>
        <div class="card-title">SHAP Feature Attribution</div>
    </div>
    """, unsafe_allow_html=True)

    # Build SHAP values for the single applicant instance
    display_names = [FEATURE_DISPLAY[f] for f in FEATURES]
    shap_vals = explainer.shap_values(input_arr)

    # Binary classifier → list of two arrays; take class-1 (default)
    if isinstance(shap_vals, list):
        sv   = shap_vals[1][0]
        base = explainer.expected_value[1]
    else:
        sv   = shap_vals[0]
        base = float(explainer.expected_value)

    explanation = shap.Explanation(
        values=sv,
        base_values=base,
        data=input_arr[0],
        feature_names=display_names,
    )

    # ── Dark-themed matplotlib waterfall ──
    shap.plots.waterfall(explanation, show=False)
    fig_shap = plt.gcf()
    fig_shap.set_size_inches(7, 4.2)
    fig_shap.patch.set_facecolor("#1E293B")

    for ax in fig_shap.axes:
        ax.set_facecolor("#1E293B")
        ax.tick_params(colors="#CBD5E1", labelsize=9)
        ax.xaxis.label.set_color("#94A3B8")
        ax.yaxis.label.set_color("#94A3B8")
        if ax.get_title():
            ax.title.set_color("#F8FAFC")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        for txt in ax.texts:
            txt.set_color("#F8FAFC")
        for lbl in ax.get_yticklabels():
            lbl.set_color("#CBD5E1")
        for lbl in ax.get_xticklabels():
            lbl.set_color("#CBD5E1")

    fig_shap.tight_layout()
    st.pyplot(fig_shap, use_container_width=True)
    plt.close("all")

# ──────────────── COLUMN 3: Risk Density KDE ──────────
with col_kde:
    st.markdown("""
    <div class="bento-card">
        <div class="card-header">Distribution Analysis</div>
        <div class="card-title">Risk Score Density</div>
    </div>
    """, unsafe_allow_html=True)

    paid_p    = baseline_probas[df['target'].values == 0] * 100
    default_p = baseline_probas[df['target'].values == 1] * 100
    x_range   = np.linspace(0, 100, 500)
    kde_paid  = gaussian_kde(paid_p,    bw_method=0.15)(x_range)
    kde_def   = gaussian_kde(default_p, bw_method=0.15)(x_range)

    fig_kde = go.Figure()

    # Fully-Paid curve (teal)
    fig_kde.add_trace(go.Scatter(
        x=x_range, y=kde_paid, fill="tozeroy",
        name="Fully Paid",
        line=dict(color="#2DD4BF", width=2),
        fillcolor="rgba(45,212,191,0.12)",
    ))
    # Charged-Off curve (crimson)
    fig_kde.add_trace(go.Scatter(
        x=x_range, y=kde_def, fill="tozeroy",
        name="Charged Off",
        line=dict(color="#FB7185", width=2),
        fillcolor="rgba(251,113,133,0.12)",
    ))

    # Applicant marker — glowing white line
    fig_kde.add_vline(
        x=risk_pct, line_width=8,
        line_color="rgba(255,255,255,0.12)",
    )
    fig_kde.add_vline(
        x=risk_pct, line_width=3, line_color="#FFFFFF",
        annotation_text=f"  YOU: {risk_pct:.1f}%",
        annotation_font=dict(color="#FFFFFF", size=11,
                             family="JetBrains Mono"),
        annotation_position="top right",
    )

    fig_kde.update_layout(
        plot_bgcolor="#1E293B", paper_bgcolor="#1E293B",
        font=dict(family="Inter", size=11, color="#94A3B8"),
        margin=dict(l=10, r=10, t=28, b=38),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    font=dict(size=11, color="#94A3B8"),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Risk Score %", showgrid=False,
                   color="#94A3B8", ticksuffix="%"),
        yaxis=dict(title="Density", showgrid=True,
                   gridcolor="rgba(51,65,85,0.5)", color="#94A3B8"),
        hovermode="x unified",
        height=380,
    )
    st.plotly_chart(fig_kde, use_container_width=True)


# ══════════════════════════════════════════════════════
#  ROW 2 — Concept Drift  |  Income Paradox
# ══════════════════════════════════════════════════════
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
col_drift, col_paradox = st.columns(2, gap="medium")

# ──────────────── Concept Drift ───────────────────────
with col_drift:
    st.markdown("""
    <div class="bento-card">
        <div class="card-header">Temporal Analysis</div>
        <div class="card-title">Concept Drift — Default Rate by Grade</div>
    </div>
    """, unsafe_allow_html=True)

    drift_grades = ['A', 'C', 'G']
    drift_df  = df[df['grade'].isin(drift_grades)].copy()
    drift_agg = (drift_df.groupby(['issue_year', 'grade'])['target']
                 .mean().reset_index())
    drift_agg['default_rate'] = drift_agg['target'] * 100
    drift_agg = drift_agg.sort_values('issue_year')

    grade_colors = {'A': '#2DD4BF', 'C': '#FBBF24', 'G': '#FB7185'}

    fig_drift = go.Figure()
    for g in drift_grades:
        gdf = drift_agg[drift_agg['grade'] == g]
        fig_drift.add_trace(go.Scatter(
            x=gdf['issue_year'], y=gdf['default_rate'],
            mode="lines+markers", name=f"Grade {g}",
            line=dict(color=grade_colors[g], width=2.5),
            marker=dict(size=7),
        ))

    fig_drift.update_layout(
        plot_bgcolor="#1E293B", paper_bgcolor="#1E293B",
        font=dict(family="Inter", size=11, color="#94A3B8"),
        margin=dict(l=10, r=10, t=24, b=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    font=dict(size=11, color="#94A3B8"),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, dtick=1, color="#94A3B8"),
        yaxis=dict(showgrid=True, gridcolor="rgba(51,65,85,0.5)",
                   title="Default Rate %", ticksuffix="%",
                   color="#94A3B8"),
        hovermode="x unified",
        height=340,
    )
    st.plotly_chart(fig_drift, use_container_width=True)

# ──────────────── Income Paradox ──────────────────────
with col_paradox:
    st.markdown("""
    <div class="bento-card">
        <div class="card-header">Anomaly Detection</div>
        <div class="card-title">The Income Paradox</div>
    </div>
    """, unsafe_allow_html=True)

    inc_df = df.dropna(subset=['annual_inc']).copy()
    inc_df['income_q'] = pd.qcut(
        inc_df['annual_inc'], q=4,
        labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'],
    )
    inc_agg = (inc_df.groupby(['grade', 'income_q'], observed=False)['target']
               .mean().reset_index())
    inc_agg['default_rate'] = inc_agg['target'] * 100

    q_colors = {
        'Q1 (Low)':  '#FB7185',
        'Q2':        '#FBBF24',
        'Q3':        '#60A5FA',
        'Q4 (High)': '#2DD4BF',
    }

    fig_inc = go.Figure()
    for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
        qdf = inc_agg[inc_agg['income_q'] == q]
        fig_inc.add_trace(go.Bar(
            x=qdf['grade'], y=qdf['default_rate'],
            name=q, marker_color=q_colors[q],
            marker_line=dict(width=0),
        ))

    fig_inc.update_layout(
        barmode="group",
        plot_bgcolor="#1E293B", paper_bgcolor="#1E293B",
        font=dict(family="Inter", size=11, color="#94A3B8"),
        margin=dict(l=10, r=10, t=24, b=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    font=dict(size=11, color="#94A3B8"),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, color="#94A3B8"),
        yaxis=dict(showgrid=True, gridcolor="rgba(51,65,85,0.5)",
                   title="Default Rate %", ticksuffix="%",
                   color="#94A3B8"),
        bargap=0.25, bargroupgap=0.08,
        height=340,
    )
    st.plotly_chart(fig_inc, use_container_width=True)


# ──────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:22px 0 10px;
            font-family:'JetBrains Mono',monospace;
            font-size:10px; color:#475569; letter-spacing:1.5px;">
    BUILT WITH STREAMLIT · PLOTLY · SHAP · SCIKIT-LEARN
    &nbsp;·&nbsp; DATA: LENDING CLUB 2007 – 2018
</div>
""", unsafe_allow_html=True)
