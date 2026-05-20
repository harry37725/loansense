import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.logistic_regression import LogisticRegression
from models.lda                 import LDA
from models.naive_bayes         import NaiveBayes
from models.bayesian_classifier import BayesianClassifier
from models.decision_tree       import DecisionTreeClassifier
from models.random_forest       import RandomForestClassifier
from models.neural_network      import NeuralNetworkClassifier

plt.style.use('seaborn-v0_8')

# ------------------------------------------------------------------ #
#  1. Load Data                                                        #
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

sub_idx     = np.random.choice(len(X_train), size=10000, replace=False)
X_train_sub = X_train[sub_idx]
y_train_sub = y_train[sub_idx]

feature_names = list(df.drop('defaulted', axis=1).columns)
save_dir      = os.path.join(BASE_DIR, 'models', 'saved')
assets_dir    = os.path.join(BASE_DIR, 'assets')
os.makedirs(assets_dir, exist_ok=True)

print("✅ Data loaded")

# ------------------------------------------------------------------ #
#  2. Load All Saved Models                                            #
# ------------------------------------------------------------------ #
def load_model(cls, path, **kwargs):
    # NeuralNetwork initializes weights in __init__ — must load differently
    if cls == NeuralNetworkClassifier:
        with open(path, 'rb') as f:
            import pickle
            m = pickle.load(f)
        return m
    m = cls(**kwargs)
    m.load(path)
    return m

models = {
    'Logistic Regression' : load_model(LogisticRegression,
                                os.path.join(save_dir, 'logistic_regression.npy')),
    'LDA'                 : load_model(LDA,
                                os.path.join(save_dir, 'lda.pkl')),
    'Naive Bayes'         : load_model(NaiveBayes,
                                os.path.join(save_dir, 'naive_bayes.pkl')),
    'Bayesian Classifier' : load_model(BayesianClassifier,
                                os.path.join(save_dir, 'bayesian_classifier.pkl')),
    'Decision Tree (Best)': load_model(DecisionTreeClassifier,
                                os.path.join(save_dir, 'best_model.pkl')),
    'Random Forest'       : load_model(RandomForestClassifier,
                                os.path.join(save_dir, 'random_forest.pkl')),
    'Neural Network'      : load_model(NeuralNetworkClassifier,
                                os.path.join(save_dir, 'neural_network.pkl')),
}
print("✅ All models loaded")

# ------------------------------------------------------------------ #
#  3. Helpers                                                          #
# ------------------------------------------------------------------ #
def extract_proba(y_prob_raw, y_pred):
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
    arr = np.array(y_prob_raw)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[1] == 2:
            return arr[:, 1]
        if arr.shape[1] == 1:
            col = arr[:, 0].astype(float)
            rng = col.max() - col.min()
            return (col - col.min()) / (rng + 1e-8)
    return y_pred.astype(float)


def compute_roc(y_true, y_prob):
    thresholds = np.linspace(0, 1, 200)
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
    return fprs, tprs, auc


def compute_pr_curve(y_true, y_prob):
    thresholds = np.linspace(0, 1, 200)
    precisions, recalls = [], []
    for t in thresholds:
        p   = (y_prob >= t).astype(int)
        tp  = np.sum((p == 1) & (y_true == 1))
        fp  = np.sum((p == 1) & (y_true == 0))
        fn  = np.sum((p == 0) & (y_true == 1))
        precisions.append(tp / (tp + fp + 1e-8))
        recalls.append(tp / (tp + fn + 1e-8))
    return recalls, precisions


# ------------------------------------------------------------------ #
#  4. Collect Predictions                                              #
# ------------------------------------------------------------------ #
all_probs  = {}
all_preds  = {}
THRESHOLD  = 0.2

print("\nRunning predictions...")
for name, model in models.items():
    y_pred     = model.predict(X_test)
    y_prob_raw = model.predict_proba(X_test)
    y_prob     = extract_proba(y_prob_raw, y_pred)
    all_probs[name] = y_prob
    all_preds[name] = (y_prob >= THRESHOLD).astype(int)
    print(f"  ✅ {name}")

# ------------------------------------------------------------------ #
#  5. Plot 1 — ROC Curves                                              #
# ------------------------------------------------------------------ #
print("\n📊 Generating ROC curves...")

colors = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22']
fig, ax = plt.subplots(figsize=(10, 7))

for (name, y_prob), color in zip(all_probs.items(), colors):
    fprs, tprs, auc = compute_roc(y_test, y_prob)
    ax.plot(fprs, tprs, color=color, linewidth=2,
            label=f'{name} (AUC = {auc:.3f})')

ax.plot([0,1],[0,1], 'k--', linewidth=1, label='Random classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ roc_curves.png saved")

# ------------------------------------------------------------------ #
#  6. Plot 2 — Confusion Matrix (Best Model Only)                      #
# ------------------------------------------------------------------ #
print("\n📊 Generating confusion matrix...")

y_pred_best = all_preds['Decision Tree (Best)']
cm = np.array([
    [np.sum((y_pred_best==0)&(y_test==0)), np.sum((y_pred_best==1)&(y_test==0))],
    [np.sum((y_pred_best==0)&(y_test==1)), np.sum((y_pred_best==1)&(y_test==1))]
])

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax)

labels = ['Non-Default (0)', 'Default (1)']
ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=11)
ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Confusion Matrix — Decision Tree (Best Model)', fontsize=13, fontweight='bold')

for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i,j]:,}',
                ha='center', va='center',
                fontsize=14, fontweight='bold',
                color='white' if cm[i,j] > cm.max()/2 else 'black')

plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ confusion_matrix.png saved")

# ------------------------------------------------------------------ #
#  7. Plot 3 — Metrics Comparison Bar Chart                            #
# ------------------------------------------------------------------ #
print("\n📊 Generating metrics comparison...")

metrics_data = {}
for name, y_prob in all_probs.items():
    y_pred = all_preds[name]
    tp = np.sum((y_pred==1)&(y_test==1))
    fp = np.sum((y_pred==1)&(y_test==0))
    fn = np.sum((y_pred==0)&(y_test==1))
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2*precision*recall / (precision+recall+1e-8)
    _, _, auc = compute_roc(y_test, y_prob)
    metrics_data[name] = {
        'F1'     : f1 * 100,
        'Recall' : recall * 100,
        'AUC'    : auc * 100,
    }

x       = np.arange(len(metrics_data))
names   = list(metrics_data.keys())
f1s     = [metrics_data[n]['F1']     for n in names]
recalls = [metrics_data[n]['Recall'] for n in names]
aucs    = [metrics_data[n]['AUC']    for n in names]
width   = 0.25

fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(x - width, f1s,     width, label='F1 Score',  color='#3498db', edgecolor='black')
ax.bar(x,         recalls, width, label='Recall',     color='#e74c3c', edgecolor='black')
ax.bar(x + width, aucs,    width, label='AUC-ROC %',  color='#2ecc71', edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Model Comparison — F1, Recall, AUC-ROC', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ model_comparison.png saved")

# ------------------------------------------------------------------ #
#  8. Plot 4 — Feature Importance (Decision Tree Weights via LR)       #
# ------------------------------------------------------------------ #
print("\n📊 Generating feature importance...")

lr_model = models['Logistic Regression']
importance = np.abs(lr_model.weights)
importance = importance / importance.sum() * 100
sorted_idx = np.argsort(importance)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh([feature_names[i] for i in sorted_idx],
               importance[sorted_idx],
               color='#3498db', edgecolor='black')

for bar, val in zip(bars, importance[sorted_idx]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=10)

ax.set_xlabel('Relative Importance (%)', fontsize=12)
ax.set_title('Feature Importance (via Logistic Regression Weights)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ feature_importance.png saved")

# ------------------------------------------------------------------ #
#  9. Plot 5 — Precision-Recall Curve (Best Model)                    #
# ------------------------------------------------------------------ #
print("\n📊 Generating precision-recall curve...")

fig, ax = plt.subplots(figsize=(8, 6))
for (name, y_prob), color in zip(all_probs.items(), colors):
    recalls_pr, precisions_pr = compute_pr_curve(y_test, y_prob)
    ax.plot(recalls_pr, precisions_pr, color=color, linewidth=2, label=name)

ax.axhline(y=y_test.mean(), color='black', linestyle='--',
           linewidth=1, label=f'Baseline ({y_test.mean()*100:.1f}%)')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves — All Models', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'precision_recall.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  ✅ precision_recall.png saved")

# ------------------------------------------------------------------ #
#  10. Summary                                                         #
# ------------------------------------------------------------------ #
print("\n" + "=" * 55)
print("  ✅ All charts saved to assets/")
print("=" * 55)
print("  roc_curves.png")
print("  confusion_matrix.png")
print("  model_comparison.png")
print("  feature_importance.png")
print("  precision_recall.png")
print("\n  → Ready for GitHub!")
print("=" * 55)