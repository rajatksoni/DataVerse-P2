"""
train_model.py — ML Pipeline for Risk Underwriting Simulator
=============================================================
- Loads loan_core.csv and extra_unassigned.csv (only needed columns).
- Filters to 'Fully Paid' (0) and 'Charged Off' / 'Default' (1).
- Trains a RandomForestClassifier with class_weight='balanced'.
- Exports rf_model.pkl and dashboard_data.csv (50k sample).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import gc

# ──────────────────────────────────────────────────────
# 1. LOAD ONLY THE COLUMNS WE NEED (memory-efficient)
# ──────────────────────────────────────────────────────
print("▸ Loading loan_core.csv ...")
core_cols = ['grade', 'loan_status']
df_core = pd.read_csv("loan_core.csv", usecols=core_cols)

print("▸ Loading extra_unassigned.csv ...")
extra_cols = ['loan_amnt', 'annual_inc', 'dti', 'int_rate', 'revol_util', 'issue_d', 'sub_grade']
df_extra = pd.read_csv("extra_unassigned.csv", usecols=extra_cols, low_memory=False)

# Merge by row position (both files have identical row alignment)
df = pd.concat([df_core, df_extra], axis=1)
del df_core, df_extra
gc.collect()
print(f"  → Merged shape: {df.shape}")

# ──────────────────────────────────────────────────────
# 2. FILTER TO RESOLVED LOANS ONLY
# ──────────────────────────────────────────────────────
print("\n▸ Filtering to resolved loans ...")
default_labels = ['Charged Off', 'Default',
                  'Does not meet the credit policy. Status:Charged Off']
paid_labels    = ['Fully Paid',
                  'Does not meet the credit policy. Status:Fully Paid']

df = df[df['loan_status'].isin(default_labels + paid_labels)].copy()
df['target'] = df['loan_status'].isin(default_labels).astype(int)
print(f"  → Filtered shape: {df.shape}")
print(f"  → Target distribution:\n{df['target'].value_counts()}")

# ──────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────
print("\n▸ Engineering features ...")

# Parse int_rate (may have '%' suffix) to float
df['int_rate'] = (
    df['int_rate']
    .astype(str)
    .str.replace('%', '', regex=False)
    .str.strip()
)
df['int_rate'] = pd.to_numeric(df['int_rate'], errors='coerce')

# Parse revol_util similarly
df['revol_util'] = (
    df['revol_util']
    .astype(str)
    .str.replace('%', '', regex=False)
    .str.strip()
)
df['revol_util'] = pd.to_numeric(df['revol_util'], errors='coerce')

# Extract issue_year for dashboard
df['issue_year'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce').dt.year

FEATURES = ['loan_amnt', 'annual_inc', 'dti', 'int_rate', 'revol_util']

# ──────────────────────────────────────────────────────
# 4. FILL NaN WITH MEDIAN (robustness)
# ──────────────────────────────────────────────────────
print("\n▸ Handling missing values ...")
for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    n_missing = df[col].isna().sum()
    if n_missing > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  → {col}: filled {n_missing:,} NaNs with median = {median_val:.2f}")

# Down-cast to float32 to save memory
for col in FEATURES:
    df[col] = df[col].astype('float32')

# ──────────────────────────────────────────────────────
# 5. TRAIN RANDOM FOREST
# ──────────────────────────────────────────────────────
print("\n▸ Training RandomForestClassifier ...")
X = df[FEATURES]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=7,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=['Fully Paid', 'Default']))
print(f"  → ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Feature importances
print("\n── Feature Importances ──")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:>15s}: {imp:.4f}")

# ──────────────────────────────────────────────────────
# 6. SAVE MODEL
# ──────────────────────────────────────────────────────
model_path = "rf_model.pkl"
joblib.dump(model, model_path)
print(f"\n▸ Model saved → {model_path}")

# ──────────────────────────────────────────────────────
# 7. EXPORT DASHBOARD SAMPLE (50k rows)
# ──────────────────────────────────────────────────────
print("\n▸ Exporting dashboard_data.csv (50,000 rows) ...")
dashboard_cols = FEATURES + ['grade', 'sub_grade', 'target', 'issue_year', 'loan_status']
sample = df[dashboard_cols].sample(n=50_000, random_state=42)
sample.to_csv("dashboard_data.csv", index=False)
print(f"  → Saved {len(sample):,} rows  × {len(dashboard_cols)} cols")

print("\n✅  Pipeline complete.")
