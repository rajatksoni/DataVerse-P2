"""
app.py — Risk Underwriting Simulator & Concept Drift Detector
==============================================================
Bento-box Streamlit dashboard with:
  • Left column: Concept Drift line chart  + Income Paradox bar chart
  • Right column: Real-time risk prediction simulator
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ──────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Underwriting Simulator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────
# INJECTED CSS — Bento-Box Design System
# ──────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    /* Hide default Streamlit header / footer */
    header[data-testid="stHeader"] { background: transparent; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    /* ── Bento Card ── */
    .bento-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 24px 28px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 16px;
        transition: box-shadow 0.2s ease;
    }
    .bento-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
    }

    /* ── Card header ── */
    .card-header {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #6b7280;
        margin-bottom: 4px;
    }
    .card-title {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 16px;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1e293b 0%, #334155 50%, #475569 100%);
        color: white;
        border-radius: 16px;
        padding: 36px 40px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(30,41,59,0.25);
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .hero-sub {
        font-size: 14px;
        font-weight: 400;
        color: #94a3b8;
    }

    /* ── KPI row ── */
    .kpi-value {
        font-size: 32px;
        font-weight: 800;
        color: #111827;
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 500;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    /* ── Risk output ── */
    .risk-output {
        text-align: center;
        padding: 20px;
        border-radius: 12px;
        margin-top: 12px;
    }
    .risk-value {
        font-size: 52px;
        font-weight: 800;
        line-height: 1.1;
    }
    .risk-label {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* ── Slider tweaks ── */
    div[data-testid="stSlider"] label {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #374151 !important;
    }

    /* ── Plotly chart spacing ── */
    .stPlotlyChart { margin-top: -10px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
# DATA & MODEL LOADING
# ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("dashboard_data.csv")
    return df

@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl")

df = load_data()
model = load_model()
FEATURES = ['loan_amnt', 'annual_inc', 'dti', 'int_rate', 'revol_util']

# ──────────────────────────────────────────────────────
# HERO BANNER
# ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🛡️ Risk Underwriting Simulator</div>
    <div class="hero-sub">Concept Drift Detector  ·  Powered by Random Forest (100 trees, depth-7, balanced)  ·  Trained on 1M+ Lending Club loans</div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
# KPI ROW
# ──────────────────────────────────────────────────────
total_loans = len(df)
default_rate = df['target'].mean() * 100
avg_loan = df['loan_amnt'].mean()
avg_income = df['annual_inc'].mean()

k1, k2, k3, k4 = st.columns(4)
for col, val, label in [
    (k1, f"{total_loans:,}", "Sample Size"),
    (k2, f"{default_rate:.1f}%", "Default Rate"),
    (k3, f"${avg_loan:,.0f}", "Avg Loan Amount"),
    (k4, f"${avg_income:,.0f}", "Avg Annual Income"),
]:
    col.markdown(f"""
    <div class="bento-card" style="text-align:center; padding:18px 16px;">
        <div class="kpi-value">{val}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
# MAIN LAYOUT — Two columns
# ──────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="medium")

# ╔═════════════════════════════════════════════════════╗
# ║  LEFT COLUMN — Visual Evidence                     ║
# ╚═════════════════════════════════════════════════════╝
with left_col:

    # ── VISUAL 1: Concept Drift Line Chart ──
    st.markdown("""
    <div class="bento-card">
        <div class="card-header">Concept Drift</div>
        <div class="card-title">Default Rate by Grade Over Time</div>
    </div>
    """, unsafe_allow_html=True)

    drift_grades = ['A', 'C', 'G']
    drift_df = df[df['grade'].isin(drift_grades)].copy()
    drift_agg = (
        drift_df.groupby(['issue_year', 'grade'])['target']
        .mean()
        .reset_index()
    )
    drift_agg['default_rate'] = drift_agg['target'] * 100
    drift_agg = drift_agg.sort_values('issue_year')

    color_map = {'A': '#10b981', 'C': '#f59e0b', 'G': '#ef4444'}

    fig_drift = px.line(
        drift_agg,
        x='issue_year',
        y='default_rate',
        color='grade',
        color_discrete_map=color_map,
        markers=True,
        labels={
            'issue_year': 'Issue Year',
            'default_rate': 'Default Rate %',
            'grade': 'Grade',
        },
    )
    fig_drift.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter', size=12),
        margin=dict(l=16, r=16, t=24, b=16),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            font=dict(size=12),
        ),
        xaxis=dict(showgrid=False, dtick=1),
        yaxis=dict(
            showgrid=True, gridcolor='#f3f4f6',
            title='Default Rate %', ticksuffix='%'
        ),
        hovermode='x unified',
    )
    fig_drift.update_traces(line=dict(width=2.5), marker=dict(size=7))
    st.plotly_chart(fig_drift, use_container_width=True)

    # ── VISUAL 2: Income Paradox Bar Chart ──
    st.markdown("""
    <div class="bento-card">
        <div class="card-header">Income Paradox</div>
        <div class="card-title">Default Rate by Grade & Income Quartile</div>
    </div>
    """, unsafe_allow_html=True)

    inc_df = df.dropna(subset=['annual_inc']).copy()
    inc_df['income_q'] = pd.qcut(
        inc_df['annual_inc'], q=4,
        labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
    )
    inc_agg = (
        inc_df.groupby(['grade', 'income_q'], observed=False)['target']
        .mean()
        .reset_index()
    )
    inc_agg['default_rate'] = inc_agg['target'] * 100

    quartile_colors = {
        'Q1 (Low)': '#ef4444',
        'Q2': '#f59e0b',
        'Q3': '#3b82f6',
        'Q4 (High)': '#10b981',
    }

    fig_income = px.bar(
        inc_agg,
        x='grade',
        y='default_rate',
        color='income_q',
        barmode='group',
        color_discrete_map=quartile_colors,
        labels={
            'grade': 'Grade',
            'default_rate': 'Default Rate %',
            'income_q': 'Income Quartile',
        },
    )
    fig_income.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter', size=12),
        margin=dict(l=16, r=16, t=24, b=16),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            font=dict(size=11),
        ),
        xaxis=dict(showgrid=False),
        yaxis=dict(
            showgrid=True, gridcolor='#f3f4f6',
            title='Default Rate %', ticksuffix='%'
        ),
        bargap=0.25,
        bargroupgap=0.08,
    )
    st.plotly_chart(fig_income, use_container_width=True)


# ╔═════════════════════════════════════════════════════╗
# ║  RIGHT COLUMN — ML Simulator                      ║
# ╚═════════════════════════════════════════════════════╝
with right_col:

    st.markdown("""
    <div class="bento-card">
        <div class="card-header">ML Simulator</div>
        <div class="card-title">Real-Time Risk Profile</div>
    </div>
    """, unsafe_allow_html=True)

    # Build sliders with realistic min/max bounds from data
    slider_config = {
        'loan_amnt': {
            'label': '💰 Loan Amount ($)',
            'min': int(df['loan_amnt'].min()),
            'max': int(df['loan_amnt'].max()),
            'default': int(df['loan_amnt'].median()),
            'step': 500,
        },
        'annual_inc': {
            'label': '📊 Annual Income ($)',
            'min': int(max(df['annual_inc'].min(), 0)),
            'max': int(min(df['annual_inc'].max(), 500_000)),
            'default': int(df['annual_inc'].median()),
            'step': 1000,
        },
        'dti': {
            'label': '📐 Debt-to-Income Ratio',
            'min': float(max(df['dti'].min(), 0)),
            'max': float(min(df['dti'].max(), 60)),
            'default': float(df['dti'].median()),
            'step': 0.5,
        },
        'int_rate': {
            'label': '📈 Interest Rate (%)',
            'min': float(df['int_rate'].min()),
            'max': float(df['int_rate'].max()),
            'default': float(df['int_rate'].median()),
            'step': 0.25,
        },
        'revol_util': {
            'label': '🔄 Revolving Utilization (%)',
            'min': 0.0,
            'max': float(min(df['revol_util'].max(), 150)),
            'default': float(df['revol_util'].median()),
            'step': 0.5,
        },
    }

    inputs = {}
    for feat, cfg in slider_config.items():
        inputs[feat] = st.slider(
            cfg['label'],
            min_value=cfg['min'],
            max_value=cfg['max'],
            value=cfg['default'],
            step=cfg['step'],
        )

    # ── Prediction ──
    input_array = np.array([[inputs[f] for f in FEATURES]], dtype='float32')
    proba = model.predict_proba(input_array)[0][1]  # P(default)
    risk_pct = proba * 100

    if risk_pct > 15:
        color = '#ef4444'
        bg_color = '#fef2f2'
        border_color = '#fecaca'
        verdict = '⚠️ HIGH RISK'
        verdict_color = '#dc2626'
    else:
        color = '#10b981'
        bg_color = '#f0fdf4'
        border_color = '#bbf7d0'
        verdict = '✅ LOW RISK'
        verdict_color = '#059669'

    st.markdown(f"""
    <div class="risk-output" style="background:{bg_color}; border: 2px solid {border_color};">
        <div class="risk-value" style="color:{color};">{risk_pct:.1f}%</div>
        <div class="risk-label" style="color:{verdict_color};">{verdict}</div>
    </div>
    """, unsafe_allow_html=True)

    # Warning / success message
    if risk_pct > 15:
        st.warning(
            f"**Predicted default probability is {risk_pct:.1f}%.** "
            f"This applicant profile exceeds the 15% risk threshold. "
            f"Consider additional underwriting scrutiny."
        )
    else:
        st.success(
            f"**Predicted default probability is {risk_pct:.1f}%.** "
            f"This applicant profile falls within the acceptable risk band."
        )

    # ── Feature importance reference ──
    st.markdown("""
    <div class="bento-card" style="margin-top: 16px;">
        <div class="card-header">Model Internals</div>
        <div class="card-title">Feature Importances</div>
    </div>
    """, unsafe_allow_html=True)

    importances = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': model.feature_importances_,
    }).sort_values('Importance', ascending=True)

    fig_imp = px.bar(
        importances,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale=['#e0e7ff', '#4f46e5'],
    )
    fig_imp.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter', size=12),
        margin=dict(l=8, r=16, t=8, b=8),
        showlegend=False,
        coloraxis_showscale=False,
        yaxis_title='',
        xaxis_title='Importance',
        xaxis=dict(showgrid=True, gridcolor='#f3f4f6'),
        yaxis=dict(showgrid=False),
        height=220,
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# ──────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:20px 0 10px; color:#9ca3af; font-size:12px;">
    Built with Streamlit · Plotly · scikit-learn  ·  Data: Lending Club (2007–2018)
</div>
""", unsafe_allow_html=True)
