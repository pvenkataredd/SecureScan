"""
SecureScan - XGBoost Classifier
ECE 547 Security Engineering | Spring 2026
Group 6: Isabel Wabno, Szymon Wabno, Neha Janardhan, Parinitha Venkata Reddy, AJ El Hout

Usage:
    python classifier.py feature_matrix.npy
"""

import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import joblib
import time

# ── 1. Load data ─────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print("Usage: python classifier.py feature_matrix.npy")
    sys.exit(1)

path = sys.argv[1]
print(f"\nLoading {path} ...")
data = np.load(path, allow_pickle=True)
print(f"  Loaded {len(data):,} rows")

# ── 2. Unpack into X / y ──────────────────────────────────────────────────────

print("Unpacking features and labels ...")
X = np.stack([d["features"] for d in data]).astype(np.float32)
y = np.array([d["label"]   for d in data], dtype=np.int8)

print(f"  X shape : {X.shape}")
print(f"  y shape : {y.shape}")
print(f"  Class 0 (safe)      : {(y == 0).sum():,}")
print(f"  Class 1 (malicious) : {(y == 1).sum():,}")

# ── 3. Train / test split (shuffle is critical — data was collected in blocks) ─

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    shuffle=True,
    stratify=y          # keeps 50/50 balance in both splits
)

print(f"\nSplit → train: {len(X_train):,}  |  test: {len(X_test):,}")

# ── 4. Train XGBoost ──────────────────────────────────────────────────────────

print("\nTraining XGBoost ...")
print("  (this may take a few minutes on 160k rows x 1049 features)\n")

clf = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    verbosity=1
)

t0 = time.time()
clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)
elapsed = time.time() - t0
print(f"\nTraining complete in {elapsed/60:.1f} min")

# ── 5. Evaluate ───────────────────────────────────────────────────────────────

y_pred = clf.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("  EVALUATION RESULTS  (test set -- 20% held out)")
print("=" * 60)
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {prec*100:.2f}%")
print(f"  Recall    : {rec*100:.2f}%")
print(f"  F1 Score  : {f1*100:.2f}%")
print()
print("  Confusion Matrix:")
print(f"              Pred Safe  Pred Malicious")
print(f"  True Safe      {cm[0,0]:>6,}          {cm[0,1]:>6,}")
print(f"  True Malicious {cm[1,0]:>6,}          {cm[1,1]:>6,}")
print()
print("  Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Safe", "Malicious"]))
print("=" * 60)

# ── 6. Top features ───────────────────────────────────────────────────────────

LEXICAL_FEATURE_NAMES = [
    "url_length", "domain_length", "path_length", "query_length",
    "uses_https", "uses_http", "non_standard_scheme", "num_dots",
    "num_hyphens", "num_underscores", "num_slashes", "num_question_marks",
    "num_equals", "num_ampersands", "num_at_signs", "num_percent_signs",
    "num_digits", "has_redirect_param", "has_ip_address", "num_subdomains",
    "has_port", "domain_entropy", "suspicious_keyword_count",
    "pct_encoded_chars", "has_double_encoding"
]
feature_names = LEXICAL_FEATURE_NAMES + [f"vgg19_{i}" for i in range(1024)]

importances = clf.feature_importances_
top_idx     = np.argsort(importances)[::-1][:15]

print("\n  Top 15 Most Important Features:")
print(f"  {'Rank':<5} {'Feature':<30} {'Importance':>10}")
print("  " + "-" * 48)
for rank, idx in enumerate(top_idx, 1):
    print(f"  {rank:<5} {feature_names[idx]:<30} {importances[idx]:>10.4f}")
print()

# ── 7. Save model ─────────────────────────────────────────────────────────────

model_path = "securescan_xgboost.joblib"
joblib.dump(clf, model_path)
print(f"  Model saved -> {model_path}")
print("\nNext step: integrate model into app.py using joblib.load()\n")
