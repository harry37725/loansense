import numpy as np
import pandas as pd
import sys
import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.decision_tree import DecisionTreeClassifier

# ------------------------------------------------------------------ #
#  Load & Prepare                                                      #
# ------------------------------------------------------------------ #
df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'cs-training-clean.csv'))
X  = df.drop('defaulted', axis=1).values
y  = df['defaulted'].values

X_mean = X.mean(axis=0)
X_std  = X.std(axis=0) + 1e-8
X_norm = (X - X_mean) / X_std

np.random.seed(42)
idx   = np.random.permutation(len(X_norm))
split = int(0.8 * len(idx))
X_train, X_test = X_norm[idx[:split]], X_norm[idx[split:]]
y_train, y_test = y[idx[:split]],      y[idx[split:]]

# Subsample for training
sub_idx     = np.random.choice(len(X_train), size=10000, replace=False)
X_train_sub = X_train[sub_idx]
y_train_sub = y_train[sub_idx]

# ------------------------------------------------------------------ #
#  Best Hyperparameters from Hypertune                                 #
# ------------------------------------------------------------------ #
BEST_MAX_DEPTH         = 5
BEST_MIN_SAMPLES_SPLIT = 500
BEST_THRESHOLD         = 0.2

# ------------------------------------------------------------------ #
#  Retrain                                                             #
# ------------------------------------------------------------------ #
print("🔄 Retraining Decision Tree with best hyperparameters...")

model = DecisionTreeClassifier(hyperparameters={
    'max_depth'        : BEST_MAX_DEPTH,
    'min_samples_split': BEST_MIN_SAMPLES_SPLIT,
})
model.fit(X_train_sub, y_train_sub)
print("✅ Training done!")

# ------------------------------------------------------------------ #
#  Evaluate with best threshold                                        #
# ------------------------------------------------------------------ #
y_prob = model.predict_proba(X_test)
y_pred = (y_prob >= BEST_THRESHOLD).astype(int)

tp = np.sum((y_pred == 1) & (y_test == 1))
fp = np.sum((y_pred == 1) & (y_test == 0))
fn = np.sum((y_pred == 0) & (y_test == 1))
tn = np.sum((y_pred == 0) & (y_test == 0))

accuracy  = np.mean(y_pred == y_test)
precision = tp / (tp + fp + 1e-8)
recall    = tp / (tp + fn + 1e-8)
f1        = 2 * precision * recall / (precision + recall + 1e-8)

# AUC
thresholds = np.linspace(0, 1, 100)
tprs, fprs = [], []
for t in thresholds:
    p    = (y_prob >= t).astype(int)
    tp_t = np.sum((p == 1) & (y_test == 1))
    fp_t = np.sum((p == 1) & (y_test == 0))
    fn_t = np.sum((p == 0) & (y_test == 1))
    tn_t = np.sum((p == 0) & (y_test == 0))
    tprs.append(tp_t / (tp_t + fn_t + 1e-8))
    fprs.append(fp_t / (fp_t + tn_t + 1e-8))
auc = abs(np.trapezoid(tprs, fprs))

print("\n" + "=" * 50)
print("  Final Model Performance")
print("=" * 50)
print(f"  Accuracy  : {accuracy*100:.2f}%")
print(f"  Precision : {precision*100:.2f}%")
print(f"  Recall    : {recall*100:.2f}%")
print(f"  F1 Score  : {f1*100:.2f}%")
print(f"  AUC-ROC   : {auc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"  TP={tp:,}  FP={fp:,}")
print(f"  FN={fn:,}  TN={tn:,}")
print("=" * 50)

# ------------------------------------------------------------------ #
#  Save Final Model + Config                                           #
# ------------------------------------------------------------------ #
save_dir = os.path.join(BASE_DIR, 'models', 'saved')

# Save model
model.save(os.path.join(save_dir, 'best_model.pkl'))

# Save config — Flask API reads this
config = {
    'model_name'  : 'Decision Tree',
    'threshold'   : BEST_THRESHOLD,
    'max_depth'   : BEST_MAX_DEPTH,
    'min_samples' : BEST_MIN_SAMPLES_SPLIT,
    'auc_roc'     : round(auc, 4),
    'f1_score'    : round(f1 * 100, 2),
    'features'    : list(df.drop('defaulted', axis=1).columns),
    'norm_mean'   : X_mean.tolist(),
    'norm_std'    : X_std.tolist(),
}

with open(os.path.join(save_dir, 'model_config.pkl'), 'wb') as f:
    pickle.dump(config, f)

print(f"\n✅ best_model.pkl      saved → models/saved/")
print(f"✅ model_config.pkl    saved → models/saved/")
print(f"\n→ Ready to build Flask API!")
