import numpy as np
import pandas as pd
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.decision_tree import DecisionTreeClassifier

# ------------------------------------------------------------------ #
#  Load Data                                                           #
# ------------------------------------------------------------------ #
df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'cs-training-clean.csv'))
X  = df.drop('defaulted', axis=1).values
y  = df['defaulted'].values

# Normalize
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0) + 1e-8
X_norm = (X - X_mean) / X_std

# Split
np.random.seed(42)
idx   = np.random.permutation(len(X_norm))
split = int(0.8 * len(idx))
X_train, X_test = X_norm[idx[:split]], X_norm[idx[split:]]
y_train, y_test = y[idx[:split]],      y[idx[split:]]

# Subsample train for speed
sub_idx     = np.random.choice(len(X_train), size=10000, replace=False)
X_train_sub = X_train[sub_idx]
y_train_sub = y_train[sub_idx]

# ------------------------------------------------------------------ #
#  Metrics                                                             #
# ------------------------------------------------------------------ #
def compute_metrics(y_true, y_pred, y_prob, threshold=0.3):
    # Apply custom threshold
    y_pred = (y_prob >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    thresholds = np.linspace(0, 1, 100)
    tprs, fprs = [], []
    for t in thresholds:
        p    = (y_prob >= t).astype(int)
        tp_t = np.sum((p == 1) & (y_true == 1))
        fp_t = np.sum((p == 1) & (y_true == 0))
        fn_t = np.sum((p == 0) & (y_true == 1))
        tn_t = np.sum((p == 0) & (y_true == 0))
        tprs.append(tp_t / (tp_t + fn_t + 1e-8))
        fprs.append(fp_t / (fp_t + tn_t + 1e-8))
    auc = abs(np.trapezoid(tprs, fprs))

    return {
        'Precision' : round(precision * 100, 2),
        'Recall'    : round(recall    * 100, 2),
        'F1'        : round(f1        * 100, 2),
        'AUC-ROC'   : round(auc, 4),
    }

# ------------------------------------------------------------------ #
#  Grid Search                                                         #
# ------------------------------------------------------------------ #
param_grid = {
    'max_depth'        : [3, 5, 7, 10, 15],
    'min_samples_split': [50, 100, 200, 500],
    'threshold'        : [0.2, 0.3, 0.4],
}

results = []
total   = len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['threshold'])
done    = 0

print(f"Running {total} combinations...\n")

for max_depth in param_grid['max_depth']:
    for min_split in param_grid['min_samples_split']:
        for threshold in param_grid['threshold']:

            model = DecisionTreeClassifier(hyperparameters={
                'max_depth'        : max_depth,
                'min_samples_split': min_split,
            })
            model.fit(X_train_sub, y_train_sub)

            y_prob = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred, y_prob, threshold=threshold)

            results.append({
                'max_depth'        : max_depth,
                'min_samples_split': min_split,
                'threshold'        : threshold,
                **metrics
            })

            done += 1
            print(f"[{done:2d}/{total}] depth={max_depth} min_split={min_split} "
                  f"thresh={threshold} | F1={metrics['F1']}% AUC={metrics['AUC-ROC']}")

# ------------------------------------------------------------------ #
#  Best Result                                                         #
# ------------------------------------------------------------------ #
results_df = pd.DataFrame(results).sort_values('F1', ascending=False)

print("\n" + "=" * 65)
print("  Top 5 Combinations by F1")
print("=" * 65)
print(results_df.head(5).to_string(index=False))

best = results_df.iloc[0]
print("\n" + "=" * 65)
print(f"  🏆 Best Hyperparameters")
print(f"     max_depth         : {int(best['max_depth'])}")
print(f"     min_samples_split : {int(best['min_samples_split'])}")
print(f"     threshold         : {best['threshold']}")
print(f"     F1                : {best['F1']}%")
print(f"     AUC-ROC           : {best['AUC-ROC']}")
print("=" * 65)

# Save results
results_df.to_csv(os.path.join(BASE_DIR, 'models', 'saved', 'hypertune_results.csv'), index=False)
print(f"\n✅ Hypertune results saved → models/saved/hypertune_results.csv")