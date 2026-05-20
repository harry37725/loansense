import sys
import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
import json
from datetime import datetime
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


# Directories
REPORTS_DIR = Path(BASE_DIR) / 'reports'
LOGS_DIR    = Path(BASE_DIR) / 'logs'
REPORTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


from models.decision_tree import DecisionTreeClassifier

app = Flask(__name__)

# ------------------------------------------------------------------ #
#  Load Model & Config Once at Startup                                 #
# ------------------------------------------------------------------ #
save_dir = os.path.join(BASE_DIR, 'models', 'saved')

with open(os.path.join(save_dir, 'model_config.pkl'), 'rb') as f:
    config = pickle.load(f)

model = DecisionTreeClassifier()
model.load(os.path.join(save_dir, 'best_model.pkl'))

THRESHOLD  = config['threshold']    # 0.2
FEATURES   = config['features']     # feature order
NORM_MEAN  = np.array(config['norm_mean'])
NORM_STD   = np.array(config['norm_std'])

print(f"✅ Model loaded | Threshold: {THRESHOLD} | Features: {FEATURES}")

# ------------------------------------------------------------------ #
#  Helper                                                              #
# ------------------------------------------------------------------ #
def get_risk_factors(features_raw, feature_names):
    """Return top 3 human-readable risk flags"""
    factors = []

    revolving_util  = features_raw[0]
    late_30_59      = features_raw[2]
    debt_ratio      = features_raw[3]
    monthly_income  = features_raw[4]
    late_90         = features_raw[6]
    late_60_89      = features_raw[8]

    if revolving_util > 0.7:
        factors.append(f"High credit utilization ({revolving_util*100:.0f}%)")
    if late_90 > 0:
        factors.append(f"{int(late_90)} serious late payment(s) over 90 days")
    if late_30_59 > 1:
        factors.append(f"{int(late_30_59)} late payment(s) in 30-59 day range")
    if late_60_89 > 0:
        factors.append(f"{int(late_60_89)} late payment(s) in 60-89 day range")
    if debt_ratio > 0.5:
        factors.append(f"Debt ratio at {debt_ratio:.2f} (above safe threshold)")
    if monthly_income < 3000:
        factors.append(f"Low monthly income (${monthly_income:,.0f})")

    return factors[:3] if factors else ["No major risk factors identified"]


def get_recommendation(prob, threshold):
    if prob >= 0.6:
        return "REJECT — High default risk"
    elif prob >= threshold:
        return "REVIEW — Moderate risk, require collateral or co-signer"
    else:
        return "APPROVE — Low default risk"


# ------------------------------------------------------------------ #
#  Routes                                                              #
# ------------------------------------------------------------------ #
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': config['model_name']}), 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No JSON body received'}), 400

        # Extract features in correct order
        raw_features = []
        missing = []
        for feat in FEATURES:
            if feat not in data:
                missing.append(feat)
            else:
                raw_features.append(float(data[feat]))

        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        raw_array = np.array(raw_features)

        # Normalize
        X_norm = (raw_array - NORM_MEAN) / NORM_STD
        X_norm = X_norm.reshape(1, -1)

        # Predict
        prob   = float(model.predict_proba(X_norm)[0])
        pred   = int(prob >= THRESHOLD)

        # Risk level
        if prob >= 0.6:
            risk_level = "HIGH"
        elif prob >= THRESHOLD:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return jsonify({
            'default_probability' : round(prob, 3),
            'risk_level'          : risk_level,
            'prediction'          : pred,
            'recommendation'      : get_recommendation(prob, THRESHOLD),
            'top_risk_factors'    : get_risk_factors(raw_array, FEATURES),
            'model'               : config['model_name'],
            'threshold_used'      : THRESHOLD,
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model'     : config['model_name'],
        'auc_roc'   : config['auc_roc'],
        'f1_score'  : config['f1_score'],
        'threshold' : config['threshold'],
        'features'  : config['features'],
    }), 200


@app.route('/save-report', methods=['POST'])
def save_report():
    try:
        data     = request.get_json()
        filename = data.get('filename', f"report_{datetime.now().timestamp()}.html")
        html     = data.get('html_report', '')

        filepath = REPORTS_DIR / filename
        filepath.write_text(html, encoding='utf-8')

        return jsonify({'saved': True, 'path': str(filepath)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/log-prediction', methods=['POST'])
def log_prediction():
    try:
        data     = request.get_json()
        log_file = LOGS_DIR / 'all_predictions.json'

        logs = []
        if log_file.exists():
            try:    logs = json.loads(log_file.read_text())
            except: logs = []

        # Don't store html in log file
        record = {k: v for k, v in data.items() if k != 'html_report'}
        logs.append(record)
        log_file.write_text(json.dumps(logs, indent=2))

        return jsonify({'logged': True, 'total': len(logs)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/log-high-risk', methods=['POST'])
def log_high_risk():
    try:
        data      = request.get_json()
        alert_file = LOGS_DIR / 'high_risk_alerts.json'

        alerts = []
        if alert_file.exists():
            try:    alerts = json.loads(alert_file.read_text())
            except: alerts = []

        record = {k: v for k, v in data.items() if k != 'html_report'}
        record['flagged_at'] = datetime.now().isoformat()
        alerts.append(record)
        alert_file.write_text(json.dumps(alerts, indent=2))

        return jsonify({'logged': True, 'total_alerts': len(alerts)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ------------------------------------------------------------------ #
#  Run                                                                 #
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  LoanSense API — Starting")
    print(f"  AUC-ROC : {config['auc_roc']}")
    print(f"  F1 Score: {config['f1_score']}%")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)