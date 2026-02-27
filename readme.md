# 🛡️ The Safe Lending — Risk Underwriting Simulator & Concept Drift Detector

> **Hackathon Project** · Built for DataVerse  
> A machine-learning-powered dashboard that exposes hidden risks in consumer lending by detecting **concept drift** and revealing the **income paradox** across 1 million+ Lending Club loans (2007–2018).

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Our Approach](#-our-approach)
- [Key Insights Discovered](#-key-insights-discovered)
- [Live Dashboard](#-live-dashboard)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Data Pipeline](#-data-pipeline)
- [Machine Learning Model](#-machine-learning-model)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Deployment](#-deployment)
- [Team](#-team)

---

## 🎯 Problem Statement

Traditional credit-risk models are trained once and deployed indefinitely, but **borrower behavior evolves over time**. A model that was accurate in 2010 may silently degrade by 2018 as macroeconomic conditions, platform policies, and borrower demographics shift. Lenders lose millions by trusting stale models that fail to detect:

1. **Concept Drift** — The relationship between risk grades and actual default rates changes over time, meaning a "Grade A" loan in 2018 does not carry the same risk as a "Grade A" loan in 2010.
2. **The Income Paradox** — Counterintuitively, higher-income borrowers within a given risk grade can default at rates comparable to (or even exceeding) lower-income borrowers, shattering the assumption that income alone is a reliable safety net.

**Our project tackles these blind spots head-on**, providing lenders with an interactive tool to monitor how risk evolves and simulate underwriting decisions in real time.

---

## 💡 Our Approach

We combined **large-scale data engineering**, **statistical concept-drift analysis**, and a **real-time ML simulator** into a single, cohesive dashboard:

| Phase | What We Did | Why It Matters |
|-------|-------------|----------------|
| **Data Ingestion** | Merged 7 CSV files (~1.2 GB, 1M+ rows) using memory-optimized chunking with dtype downcasting | Demonstrates production-grade data handling |
| **Drift Analysis** | Computed default rates by grade × year to surface temporal concept drift | Reveals model staleness risk |
| **Paradox Detection** | Segmented borrowers into income quartiles within each grade to expose the income paradox | Challenges conventional underwriting assumptions |
| **ML Modeling** | Trained a balanced Random Forest (100 trees, depth-7) on 5 financial features | Provides actionable risk scores |
| **Interactive Dashboard** | Built a Bento-box UI with real-time sliders for live risk prediction | Makes insight accessible to non-technical stakeholders |

---

## 🔍 Key Insights Discovered

### 1. Concept Drift Is Real — and Dangerous

Default rates for the safest loans (Grade A) **more than tripled** between 2010 and 2015, while high-risk grades (G) showed volatile swings of 15–40%. A static model trained on early data would catastrophically under-price risk for later vintages.

### 2. The Income Paradox

Within the same risk grade, **high-income borrowers (Q4) default at nearly the same rate as low-income borrowers (Q1)** — and in some grades, they default *more*. This means income alone is an unreliable protective factor, and models that over-weight income create a false sense of safety.

### 3. Feature Importance Ranking

Our Random Forest reveals that **interest rate** and **revolving utilization** dominate default prediction, while raw loan amount has minimal independent signal — a finding that contradicts naïve "bigger loan = bigger risk" assumptions.

---

## 🖥️ Live Dashboard

The dashboard is structured as a **Bento-box layout** with three core panels:

| Panel | Description |
|-------|-------------|
| **📈 Concept Drift Chart** | Interactive line chart showing default rate trajectories for Grades A, C, and G across issue years (2007–2018) |
| **📊 Income Paradox Chart** | Grouped bar chart revealing default rates segmented by grade × income quartile |
| **🎛️ ML Risk Simulator** | Five real-time sliders (Loan Amount, Income, DTI, Interest Rate, Revolving Utilization) driving a live Random Forest prediction with HIGH/LOW RISK verdict |
| **🔬 Feature Importances** | Horizontal bar chart showing which features the model relies on most |

> The UI features a clean, modern design with custom CSS, smooth hover animations, and a professional KPI header row.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Raw Data Layer                         │
│  loan_core.csv · borrower_profile.csv · credit_history   │
│  account_activity · account_balances · platform_metadata │
│  extra_unassigned.csv                                    │
│  ─────────── (~1.2 GB, 7 files, 1M+ rows) ───────────── │
└────────────────────┬─────────────────────────────────────┘
                     │  Memory-optimized merge (float64→float32)
                     ▼
┌──────────────────────────────────────────────────────────┐
│               Data Processing Layer                      │
│  analyze_drift.py  →  Concept drift & paradox analysis   │
│  train_model.py    →  Feature engineering + RF training  │
└────────────────────┬─────────────────────────────────────┘
                     │  Exports: rf_model.pkl + dashboard_data.csv
                     ▼
┌──────────────────────────────────────────────────────────┐
│                 Application Layer                        │
│  app.py (Streamlit)                                      │
│  ├─ KPI header row (sample size, default rate, etc.)     │
│  ├─ Concept Drift line chart (Plotly)                    │
│  ├─ Income Paradox bar chart (Plotly)                    │
│  ├─ ML Simulator w/ sliders → live RF prediction         │
│  └─ Feature Importance chart (Plotly)                    │
└──────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Interactive dashboard framework |
| **Visualization** | Plotly Express & Graph Objects | Rich, interactive charts |
| **Machine Learning** | scikit-learn (RandomForestClassifier) | Default probability prediction |
| **Data Processing** | Pandas, NumPy | Data wrangling, feature engineering |
| **Model Serialization** | Joblib | Model persistence (.pkl) |
| **Styling** | Custom CSS (Inter font, Bento-box cards) | Premium, modern UI |
| **Deployment** | Streamlit Community Cloud | Free, one-click deployment |

---

## 🔄 Data Pipeline

### Source Data
The Lending Club dataset (~1.2 GB) is distributed across **7 CSV files**:

| File | Size | Contents |
|------|------|----------|
| `extra_unassigned.csv` | 738 MB | Loan amounts, income, DTI, interest rates, revolving utilization |
| `borrower_profile.csv` | 159 MB | Employment, home ownership, verification status |
| `account_balances.csv` | 108 MB | Outstanding balances, payment history |
| `account_activity.csv` | 103 MB | Recent payment activity, delinquency flags |
| `loan_core.csv` | 84 MB | Grade, sub-grade, loan status (target variable) |
| `credit_history.csv` | 43 MB | Credit inquiries, open accounts, revolving balance |
| `platform_metadata.csv` | 39 MB | Application type, listing dates, policy codes |

### Processing Steps
1. **Memory-Optimized Loading** — Selective column reading with `usecols`; downcasting `float64 → float32` and `int64 → int32` to reduce memory footprint by ~50%.
2. **Row-Aligned Merge** — Files share identical row order, enabling a simple `concat` (axis=1) instead of expensive key-based joins.
3. **Target Construction** — Binary label from `loan_status`: Charged Off / Default → `1`, Fully Paid → `0`.
4. **Feature Engineering** — Parsing percentage strings (`int_rate`, `revol_util`), extracting `issue_year` from date strings (`%b-%Y` format).
5. **Missing Value Imputation** — Median fill for NaN values across all 5 features.
6. **Dashboard Export** — Stratified 50,000-row sample exported as `dashboard_data.csv` for fast dashboard loading.

---

## 🤖 Machine Learning Model

### Model Configuration

```
RandomForestClassifier(
    n_estimators   = 100,
    max_depth      = 7,
    class_weight   = 'balanced',
    random_state   = 42,
    n_jobs         = -1
)
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Random Forest** | Robust to outliers, handles mixed feature scales, provides feature importances out-of-the-box |
| **`class_weight='balanced'`** | Lending data is heavily imbalanced (~80% Fully Paid vs ~20% Default); balancing prevents the model from always predicting "safe" |
| **`max_depth=7`** | Prevents overfitting while capturing non-linear interactions between features like DTI × interest rate |
| **5 Features** | `loan_amnt`, `annual_inc`, `dti`, `int_rate`, `revol_util` — chosen for interpretability and availability at underwriting time |

### Evaluation
- **Train/Test Split**: 80/20 stratified split
- **Metric**: ROC-AUC score (reported in training logs)
- **Classification Report**: Precision, recall, F1 for both classes

---

## 📁 Project Structure

```
DataVerse-P2/
├── app.py                 # Streamlit dashboard (main entry point)
├── train_model.py         # ML training pipeline
├── rf_model.pkl           # Serialized Random Forest model
├── dashboard_data.csv     # Pre-processed 50K sample for the dashboard
├── requirements.txt       # Python dependencies (pinned versions)
└── readme.md              # This file
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+
- pip

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/DataVerse-P2.git
cd DataVerse-P2

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

### Re-Training the Model (Optional)
To re-train from raw data, place `loan_core.csv` and `extra_unassigned.csv` in the project directory, then run:

```bash
python train_model.py
```

This will regenerate `rf_model.pkl` and `dashboard_data.csv`.

---

## ☁️ Deployment

The app is deployed on **Streamlit Community Cloud**:

1. Push the repo to GitHub (ensure `rf_model.pkl` and `dashboard_data.csv` are included or handled via Git LFS).
2. Connect the repo on [share.streamlit.io](https://share.streamlit.io).
3. Set `app.py` as the main file.
4. Deploy — the platform auto-installs from `requirements.txt`.




<p align="center">
  <i>Built with Streamlit · Plotly · scikit-learn · Powered by 1M+ Lending Club loans (2007–2018)</i>
</p>
