"""
app.py — LUMON INDUSTRIES · Risk Underwriting Terminal
======================================================
Severance-inspired brutalist dashboard:
  • Left  : Applicant Terminal — sliders + Calculated Risk Vector
  • Right : Advanced Analytics — SHAP Waterfall + Risk Density KDE
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
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

# ──────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="LUMON · Risk Terminal",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────
# BRUTALIST CSS — Severance / Lumon Industries
# ──────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Mono', monospace;
        color: #D4D4D4;
    }

    /* ── OLED Black Canvas ── */
    .stApp { background-color: #000000; }
    header[data-testid="stHeader"] { background: transparent; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    /* ── Brutal Card ── */
    .brutal-card {
        background: #080808;
        border: 1px solid #262626;
        border-radius: 0px;
        padding: 24px 28px;
        margin-bottom: 16px;
    }

    /* ── Typography ── */
    .lumon-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 9px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: #525252;
        margin-bottom: 2px;
    }
    .lumon-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 18px;
        font-weight: 700;
        color: #E5E5E5;
        margin-bottom: 14px;
        letter-spacing: 0.5px;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: #080808;
        border: 1px solid #262626;
        border-radius: 0px;
        padding: 32px 36px;
        margin-bottom: 24px;
        border-top: 2px solid #262626;
        border-bottom: 2px solid #262626;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: #FAFAFA;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .hero-sub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        color: #525252;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .hero-divider {
        width: 40px;
        height: 1px;
        background: #262626;
        margin: 12px 0;
    }

    /* ── KPI Row ── */
    .kpi-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 24px;
        font-weight: 700;
    }
    .kpi-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 8px;
        font-weight: 500;
        color: #525252;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 2px;
    }

    /* ── Risk Vector ── */
    .risk-output {
        text-align: center;
        padding: 28px 20px;
        background: #080808;
        border: 1px solid #262626;
        border-radius: 0px;
        margin-top: 12px;
    }
    .risk-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 56px;
        font-weight: 700;
        line-height: 1;
    }
    .risk-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-top: 8px;
    }
    .risk-bar {
        height: 2px;
        margin: 14px auto 0;
        width: 60px;
    }

    /* ── Slider Overrides ── */
    div[data-testid="stSlider"] label {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        color: #A3A3A3 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }

    /* ── Section Divider ── */
    .section-rule {
        border: none;
        border-top: 1px solid #1A1A1A;
        margin: 20px 0;
    }

    /* ── Plotly spacing ── */
    .stPlotlyChart { margin-top: -8px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
# DATA · MODEL · EXPLAINER
# ──────────────────────────────────────────────────────
FEATURES = ['loan_amnt', 'annual_inc', 'dti', 'int_rate', 'revol_util']
FEATURE_DISPLAY = {
    'loan_amnt':  'LOAN AMOUNT',
    'annual_inc': 'ANNUAL INCOME',
    'dti':        'DTI RATIO',
    'int_rate':   'INTEREST RATE',
    'revol_util': 'REVOLVING UTIL',
}

@st.cache_resource
def init_app():
    """Load all resources once: data, model, SHAP explainer, baseline probas."""
    _df = pd.read_csv("dashboard_data.csv")
    _model = joblib.load("rf_model.pkl")
    _explainer = shap.TreeExplainer(_model)
    feats = ['loan_amnt', 'annual_inc', 'dti', 'int_rate', 'revol_util']
    _baseline = _model.predict_proba(
        _df[feats].values.astype("float32")
    )[:, 1]
    return _df, _model, _explainer, _baseline

df, model, explainer, baseline_probas = init_app()

# ──────────────────────────────────────────────────────
# HERO BANNER
# ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🏢 &nbsp;Lumon Industries</div>
    <div class="hero-divider"></div>
    <div class="hero-sub">
        Macrodata Refinement · Risk Underwriting Terminal
        &nbsp;&nbsp;·&nbsp;&nbsp; RF-100T / D7 / Balanced
        &nbsp;&nbsp;·&nbsp;&nbsp; 1,000,000+ Loan Records
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
    (k1, f"{total_loans:,}",      "SAMPLE SIZE",   "#D4D4D4"),
    (k2, f"{default_rate:.1f}%",  "DEFAULT RATE",  "#DC2626"),
    (k3, f"${avg_loan:,.0f}",     "AVG LOAN",      "#D4D4D4"),
    (k4, f"${avg_income:,.0f}",   "AVG INCOME",    "#059669"),
]:
    col.markdown(f"""
    <div class="brutal-card" style="text-align:center; padding:16px 10px;">
        <div class="kpi-value" style="color:{accent};">{val}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  MAIN LAYOUT — 2 Columns
# ══════════════════════════════════════════════════════
left_col, right_col = st.columns([1, 1.6], gap="medium")

# ╔═════════════════════════════════════════════════════╗
# ║  LEFT — The Applicant Terminal                      ║
# ╚═════════════════════════════════════════════════════╝
with left_col:

    st.markdown("""
    <div class="brutal-card">
        <div class="lumon-header">Department of Risk Assessment</div>
        <div class="lumon-title">Applicant Terminal</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sliders ──
    slider_cfg = {
        'loan_amnt': ('💰  LOAN AMOUNT ($)',
                      int(df['loan_amnt'].min()),
                      int(df['loan_amnt'].max()),
                      int(df['loan_amnt'].median()), 500),
        'annual_inc': ('📊  ANNUAL INCOME ($)',
                       int(max(df['annual_inc'].min(), 0)),
                       int(min(df['annual_inc'].max(), 500_000)),
                       int(df['annual_inc'].median()), 1000),
        'dti': ('📐  DEBT-TO-INCOME RATIO',
                float(max(df['dti'].min(), 0)),
                float(min(df['dti'].max(), 60)),
                float(df['dti'].median()), 0.5),
        'int_rate': ('📈  INTEREST RATE (%)',
                     float(df['int_rate'].min()),
                     float(df['int_rate'].max()),
                     float(df['int_rate'].median()), 0.25),
        'revol_util': ('🔄  REVOLVING UTILIZATION (%)',
                       0.0,
                       float(min(df['revol_util'].max(), 150)),
                       float(df['revol_util'].median()), 0.5),
    }

    inputs = {}
    for feat, (lbl, mn, mx, dv, stp) in slider_cfg.items():
        inputs[feat] = st.slider(lbl, min_value=mn, max_value=mx,
                                 value=dv, step=stp)

    # ── Prediction ──
    input_arr = np.array([[inputs[f] for f in FEATURES]], dtype="float32")
    proba     = model.predict_proba(input_arr)[0][1]
    risk_pct  = proba * 100

    if risk_pct > 15:
        risk_color = "#DC2626"
        verdict    = "CLASSIFICATION: REJECTED"
        bar_color  = "#DC2626"
    else:
        risk_color = "#059669"
        verdict    = "CLASSIFICATION: APPROVED"
        bar_color  = "#059669"

    st.markdown(f"""
    <div class="risk-output">
        <div class="lumon-header">Calculated Risk Vector</div>
        <div class="risk-value" style="color:{risk_color};">{risk_pct:.1f}%</div>
        <div class="risk-bar" style="background:{bar_color};"></div>
        <div class="risk-label" style="color:{risk_color};">{verdict}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Textual Warning ──
    if risk_pct > 15:
        st.warning(
            f"**Risk Vector: {risk_pct:.1f}%.** "
            f"Applicant profile exceeds the 15% threshold. "
            f"Escalate to Department Head for manual review."
        )
    else:
        st.success(
            f"**Risk Vector: {risk_pct:.1f}%.** "
            f"Applicant profile falls within acceptable parameters. "
            f"Proceed with standard underwriting protocol."
        )


# ╔═════════════════════════════════════════════════════╗
# ║  RIGHT — Advanced Analytics                         ║
# ╚═════════════════════════════════════════════════════╝
with right_col:

    # ───────────── VISUAL 1: SHAP Waterfall ───────────
    st.markdown("""
    <div class="brutal-card">
        <div class="lumon-header">Explainable AI Module</div>
        <div class="lumon-title">SHAP Feature Attribution</div>
    </div>
    """, unsafe_allow_html=True)

    display_names = [FEATURE_DISPLAY[f] for f in FEATURES]
    shap_vals = explainer.shap_values(input_arr)

    # Handle different SHAP return formats (version-dependent)
    ev = explainer.expected_value
    if isinstance(shap_vals, list):
        sv   = shap_vals[1][0]          # older SHAP: list of arrays
        base = ev[1]
    elif shap_vals.ndim == 3:
        sv   = shap_vals[0, :, 1]       # newer SHAP: (samples, feats, classes)
        base = float(ev[1]) if hasattr(ev, '__len__') else float(ev)
    else:
        sv   = shap_vals[0]             # single-output fallback
        base = float(ev[1]) if hasattr(ev, '__len__') and len(ev) > 1 else float(ev)

    explanation = shap.Explanation(
        values=sv,
        base_values=base,
        data=input_arr[0],
        feature_names=display_names,
    )

    # ── Dark brutalist matplotlib waterfall ──
    shap.plots.waterfall(explanation, show=False)
    fig_shap = plt.gcf()
    fig_shap.set_size_inches(8, 4.5)
    fig_shap.patch.set_facecolor("#080808")

    for ax in fig_shap.axes:
        ax.set_facecolor("#080808")
        ax.tick_params(colors="#A3A3A3", labelsize=9)
        ax.xaxis.label.set_color("#525252")
        ax.yaxis.label.set_color("#525252")
        if ax.get_title():
            ax.title.set_color("#E5E5E5")
        for spine in ax.spines.values():
            spine.set_color("#262626")

        # Recolor SHAP bars: red → muted red, blue → muted green
        for patch in ax.patches:
            fc = patch.get_facecolor()
            r, g, b = fc[0], fc[1], fc[2]
            if r > 0.6 and g < 0.3 and b < 0.3:
                patch.set_facecolor("#DC2626")
                patch.set_edgecolor("#DC2626")
            elif b > 0.6 and r < 0.3:
                patch.set_facecolor("#059669")
                patch.set_edgecolor("#059669")
            else:
                patch.set_facecolor("#262626")
                patch.set_edgecolor("#262626")

        # White text for annotations and tick labels
        for txt in ax.texts:
            txt.set_color("#E5E5E5")
        for lbl in ax.get_yticklabels():
            lbl.set_color("#A3A3A3")
            lbl.set_fontfamily("monospace")
        for lbl in ax.get_xticklabels():
            lbl.set_color("#A3A3A3")

    fig_shap.tight_layout()
    st.pyplot(fig_shap, use_container_width=True)
    plt.close("all")

    # ── Section Rule ──
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ───────────── VISUAL 2: Risk Density KDE ─────────
    st.markdown("""
    <div class="brutal-card">
        <div class="lumon-header">Distribution Analysis</div>
        <div class="lumon-title">Risk Density Curve</div>
    </div>
    """, unsafe_allow_html=True)

    paid_p    = baseline_probas[df['target'].values == 0] * 100
    default_p = baseline_probas[df['target'].values == 1] * 100
    x_range   = np.linspace(0, 100, 500)
    kde_paid  = gaussian_kde(paid_p,    bw_method=0.15)(x_range)
    kde_def   = gaussian_kde(default_p, bw_method=0.15)(x_range)

    fig_kde = go.Figure()

    # Fully Paid density (green)
    fig_kde.add_trace(go.Scatter(
        x=x_range, y=kde_paid,
        fill="tozeroy",
        name="FULLY PAID",
        line=dict(color="#059669", width=1.5),
        fillcolor="rgba(5,150,105,0.08)",
    ))

    # Charged Off density (red)
    fig_kde.add_trace(go.Scatter(
        x=x_range, y=kde_def,
        fill="tozeroy",
        name="CHARGED OFF",
        line=dict(color="#DC2626", width=1.5),
        fillcolor="rgba(220,38,38,0.08)",
    ))

    # ── Glowing dashed applicant marker ──
    # Outer glow
    fig_kde.add_vline(
        x=risk_pct, line_width=10,
        line_color="rgba(237,237,237,0.06)",
    )
    # Inner glow
    fig_kde.add_vline(
        x=risk_pct, line_width=4,
        line_color="rgba(237,237,237,0.15)",
    )
    # Core line (dashed)
    fig_kde.add_vline(
        x=risk_pct, line_width=2,
        line_color="#EDEDED", line_dash="dash",
        annotation_text=f"  APPLICANT: {risk_pct:.1f}%",
        annotation_font=dict(
            color="#EDEDED", size=10,
            family="IBM Plex Mono, monospace",
        ),
        annotation_position="top right",
    )

    fig_kde.update_layout(
        plot_bgcolor="#080808",
        paper_bgcolor="#080808",
        font=dict(
            family="IBM Plex Mono, monospace",
            size=10,
            color="#525252",
        ),
        margin=dict(l=10, r=10, t=28, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=9, color="#525252",
                      family="IBM Plex Mono, monospace"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            title="RISK SCORE %",
            showgrid=False,
            color="#525252",
            ticksuffix="%",
            zeroline=False,
            linecolor="#262626",
            linewidth=1,
        ),
        yaxis=dict(
            title="DENSITY",
            showgrid=True,
            gridcolor="#1A1A1A",
            color="#525252",
            zeroline=False,
            linecolor="#262626",
            linewidth=1,
        ),
        hovermode="x unified",
        height=380,
    )

    st.plotly_chart(fig_kde, use_container_width=True)


# ──────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:28px 0 14px;
            font-family:'IBM Plex Mono',monospace;
            font-size:8px; color:#262626; letter-spacing:3px;
            text-transform:uppercase;">
    © Lumon Industries · Kier Eagan Division of Macrodata Refinement
    &nbsp;&nbsp;·&nbsp;&nbsp; Built with Streamlit · SHAP · Plotly · scikit-learn
    &nbsp;&nbsp;·&nbsp;&nbsp; Data: Lending Club 2007–2018
</div>
""", unsafe_allow_html=True)
