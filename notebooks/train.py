import numpy as np
import pandas as pd
import sys
import os
import time

# ------------------------------------------------------------------ #
#  Paths                                                               #
# ------------------------------------------------------------------ #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.logistic_regression import LogisticRegression
from models.lda                 import LDA
from models.naive_bayes         import NaiveBayes
from models.bayesian_classifier import BayesianClassifier
from models.decision_tree       import DecisionTreeClassifier
from models.random_forest       import RandomForestClassifier
from models.neural_network      import NeuralNetworkClassifier

# ------------------------------------------------------------------ #
#  1. Load & Prepare Data                                              #
# ------------------------------------------------------------------ #
print("=" * 60)
print("  LoanSense — Training All Models")
print("=" * 60)

df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'cs-training-clean.csv'))

X = df.drop('defaulted', axis=1).values
y = df['defaulted'].values

print(f"\n✅ Data loaded: {X.shape[0]:,} rows, {X.shape[1]} features")
print(f"   Default rate: {y.mean()*100:.1f}%")

# ------------------------------------------------------------------ #
#  2. Normalize                                                        #
# ------------------------------------------------------------------ #
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0) + 1e-8
X_norm = (X - X_mean) / X_std

# Save normalization params — needed by Flask API later
save_dir = os.path.join(BASE_DIR, 'models', 'saved')
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, 'norm_params.npy'), {'mean': X_mean, 'std': X_std})
print(f"\n✅ Normalization params saved")

# ------------------------------------------------------------------ #
#  3. Train / Test Split (80/20)                                       #
# ------------------------------------------------------------------ #
np.random.seed(42)
idx   = np.random.permutation(len(X_norm))
split = int(0.8 * len(idx))

X_train, X_test = X_norm[idx[:split]], X_norm[idx[split:]]
y_train, y_test = y[idx[:split]],      y[idx[split:]]

print(f"\n✅ Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

n_features = X_train.shape[1]   # = 10, used by NeuralNetwork

# ------------------------------------------------------------------ #
#  4. Metrics Helper                                                   #
# ------------------------------------------------------------------ #
def compute_metrics(y_true, y_pred, y_prob):
    """Accuracy, Precision, Recall, F1, AUC-ROC"""
    accuracy  = np.mean(y_pred == y_true)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    # AUC-ROC (trapezoidal)
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
        'Accuracy'  : round(accuracy  * 100, 2),
        'Precision' : round(precision * 100, 2),
        'Recall'    : round(recall    * 100, 2),
        'F1'        : round(f1        * 100, 2),
        'AUC-ROC'   : round(auc,             4),
    }


def extract_proba(y_prob_raw, y_pred):
    """
    Normalise whatever predict_proba returns into a 1-D array
    of P(class=1) probabilities.

    Handles:
      - np.ndarray shape (N,)             LogisticRegression
      - np.ndarray shape (N, 2)           NeuralNetwork softmax
      - np.ndarray shape (N, 1)           LDA discriminant score
      - np.ndarray shape (N, 2) row-wise  BayesianClassifier
    """
    # ---- list output ------------------------------------------------
    if isinstance(y_prob_raw, list):
        probs = []
        for item in y_prob_raw:
            if isinstance(item, dict):
                probs.append(float(item.get(1, item.get(1.0, 0.0))))
            elif isinstance(item, np.ndarray):
                probs.append(float(item[1]) if len(item) > 1 else float(item[0]))
            else:
                probs.append(float(item))
        return np.array(probs)

    # ---- numpy array output -----------------------------------------
    arr = np.array(y_prob_raw)

    if arr.ndim == 1:
        return arr                        # already P(class=1)

    if arr.ndim == 2:
        if arr.shape[1] == 2:
            return arr[:, 1]             # softmax / two-column
        if arr.shape[1] == 1:
            # LDA discriminant scores — min-max scale to [0,1]
            col = arr[:, 0].astype(float)
            rng = col.max() - col.min()
            return (col - col.min()) / (rng + 1e-8)

    # Fallback
    return y_pred.astype(float)


# ------------------------------------------------------------------ #
#  5. Define All Models                                                #
# ------------------------------------------------------------------ #
models = {
    'Logistic Regression' : LogisticRegression(
                                learning_rate=0.01,
                                n_iterations=1000),

    'LDA'                 : LDA(),

    'Naive Bayes'         : NaiveBayes(),

    'Bayesian Classifier' : BayesianClassifier(
                                hyperparameters={'alpha': 1.0}),

    'Decision Tree'       : DecisionTreeClassifier(
                                hyperparameters={'max_depth': 5,
                                                 'min_samples_split': 500}),

    'Random Forest'       : RandomForestClassifier(
                                hyperparameters={'n_estimators': 10,
                                                 'max_depth': 5,
                                                 'min_samples_split': 500}),

    'Neural Network'      : NeuralNetworkClassifier(
                                hyperparameters={'input_dim'     : n_features,
                                                 'hidden_dim'    : 32,
                                                 'output_dim'    : 2,
                                                 'learning_rate' : 0.01,
                                                 'n_epochs'      : 500}),
}

# ------------------------------------------------------------------ #
#  6. Train, Evaluate & Save All Models                                #
# ------------------------------------------------------------------ #
results = {}

# LogisticRegression.fit() accepts verbose — silence it in batch mode
fit_kwargs = {
    'Logistic Regression': {'verbose': False}
}

print("\n" + "=" * 60)
print("  Training Models...")
print("=" * 60)

for name, model in models.items():
    print(f"\n🔄  Training {name} ...")
    start = time.time()

    # Train
    kwargs = fit_kwargs.get(name, {})
    model.fit(X_train, y_train, **kwargs)
    train_time = time.time() - start

    # Predict
    y_pred     = model.predict(X_test)
    y_prob_raw = model.predict_proba(X_test)
    y_prob     = extract_proba(y_prob_raw, y_pred)

    # Metrics
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics['Train Time (s)'] = round(train_time, 2)
    results[name] = metrics

    # Save — all models use pickle except LogisticRegression (uses .npy)
    ext      = '.npy' if name == 'Logistic Regression' else '.pkl'
    filename = name.lower().replace(' ', '_') + ext
    model.save(os.path.join(save_dir, filename))

    print(f"   ✅ Done in {train_time:.2f}s  |  "
          f"Acc: {metrics['Accuracy']}%  |  "
          f"F1: {metrics['F1']}%  |  "
          f"AUC: {metrics['AUC-ROC']}")

# ------------------------------------------------------------------ #
#  7. Results Table                                                    #
# ------------------------------------------------------------------ #
print("\n" + "=" * 60)
print("  Final Results — All Models")
print("=" * 60)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('AUC-ROC', ascending=False)
print(results_df.to_string())

results_path = os.path.join(save_dir, 'results.csv')
results_df.to_csv(results_path)
print(f"\n✅ Results saved → models/saved/results.csv")

# ------------------------------------------------------------------ #
#  8. Best Model                                                       #
# ------------------------------------------------------------------ #
best_name = results_df['AUC-ROC'].idxmax()
best      = results_df.loc[best_name]

print("\n" + "=" * 60)
print(f"  🏆  Best Model : {best_name}")
print(f"      AUC-ROC   : {best['AUC-ROC']}")
print(f"      F1 Score  : {best['F1']}%")
print(f"      Accuracy  : {best['Accuracy']}%")
print(f"\n  → Use '{best_name}' in your Flask API (api/app.py)")
print("=" * 60)