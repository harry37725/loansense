import streamlit as st
import requests
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------ #
#  Config                                                              #
# ------------------------------------------------------------------ #
BASE_DIR   = Path(__file__).parent
ASSETS_DIR = BASE_DIR / 'assets'
LOGS_DIR   = BASE_DIR / 'logs'
API_URL    = "http://127.0.0.1:5000"
N8N_URL    = "http://localhost:5678/webhook-test/loan-risk"

st.set_page_config(
    page_title  = "LoanSense Dashboard",
    page_icon   = "🏦",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ------------------------------------------------------------------ #
#  Global CSS — LoanSense Navy + Green Theme                           #
# ------------------------------------------------------------------ #
st.markdown("""
<style>
    :root {
        --navy      : #1a2744;
        --navy-light: #243460;
        --green     : #00c896;
        --green-dark: #00a37a;
        --bg        : #f4f6fb;
    }
    .stApp { background-color: var(--bg); }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2744 0%, #243460 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }

    [data-testid="stMetric"] {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(26,39,68,0.08);
        border-left: 4px solid #00c896;
    }
    [data-testid="stMetricLabel"] { color: #6b7a99 !important; font-size: 12px !important; }
    [data-testid="stMetricValue"] { color: #1a2744 !important; font-size: 24px !important; }

    .stButton > button {
        background: linear-gradient(135deg, #00c896, #00a37a);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 28px;
        font-weight: 700;
        font-size: 15px;
        width: 100%;
        transition: 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00a37a, #008060);
        transform: translateY(-1px);
    }

    .badge-high   { background:#e74c3c; color:white !important; padding:6px 18px; border-radius:20px; font-weight:700; font-size:16px; display:inline-block; }
    .badge-medium { background:#f39c12; color:white !important; padding:6px 18px; border-radius:20px; font-weight:700; font-size:16px; display:inline-block; }
    .badge-low    { background:#00c896; color:white !important; padding:6px 18px; border-radius:20px; font-weight:700; font-size:16px; display:inline-block; }

    .ls-card {
        background: white;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(26,39,68,0.08);
        margin-bottom: 16px;
    }
    .ls-header {
        background: linear-gradient(135deg, #1a2744, #243460);
        border-radius: 14px;
        padding: 32px;
        margin-bottom: 24px;
        border: 1px solid rgba(0,200,150,0.3);
    }
    .ls-header h1 { color: #ffffff !important; margin:0; font-size:32px; font-weight:800; text-shadow: 0 1px 4px rgba(0,0,0,0.4); }
    .ls-header p  { color: #00c896 !important; margin:6px 0 0; font-size:15px; font-weight:500; }

    .stForm { background: white; border-radius: 14px; padding: 20px; }
    .stTextInput input   { color: #1a2744 !important; background: #f4f6fb !important; }
    .stNumberInput input { color: #1a2744 !important; background: #f4f6fb !important; }
    label                { color: #1a2744 !important; }
    p, li                { color: #1a2744 !important; }
    .stMarkdown p        { color: #1a2744 !important; }
    h1, h2, h3           { color: #1a2744 !important; }

    .stTabs [data-baseweb="tab"]      { color: #1a2744 !important; }
    .stTabs [aria-selected="true"]    { color: #00c896 !important; border-bottom: 2px solid #00c896 !important; }
    .stDataFrame                      { color: #1a2744 !important; }
    .stProgress > div > div           { background: #00c896 !important; }
    .stAlert p                        { color: white !important; }

    .factor-item {
        background: #fff8f0;
        border-left: 4px solid #f39c12;
        border-radius: 0 8px 8px 0;
        padding: 10px 16px;
        margin: 6px 0;
        font-size: 14px;
        color: #1a2744 !important;
    }
    .section-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #6b7a99;
        margin-bottom: 12px;
    }
    .rec-approve { background:#00c896; color:white !important; border-radius:10px; padding:16px; text-align:center; font-weight:700; font-size:16px; }
    .rec-review  { background:#f39c12; color:white !important; border-radius:10px; padding:16px; text-align:center; font-weight:700; font-size:16px; }
    .rec-reject  { background:#e74c3c; color:white !important; border-radius:10px; padding:16px; text-align:center; font-weight:700; font-size:16px; }

    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except:
        return False


def check_n8n():
    try:
        r = requests.get("http://localhost:5678", timeout=3)
        return True
    except:
        return False


def call_predict(payload):
    """Always call Flask directly — fast and reliable"""
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        return r.json(), None
    except Exception as e:
        return None, str(e)


def send_to_n8n(payload):
    """Send to n8n pipeline — saves report + logs"""
    try:
        r = requests.post(N8N_URL, json=payload, timeout=15)
        return r.json(), r.status_code, None
    except Exception as e:
        return None, None, str(e)


def load_logs():
    log_file = LOGS_DIR / 'all_predictions.json'
    if log_file.exists():
        try:
            return pd.DataFrame(json.loads(log_file.read_text()))
        except:
            return pd.DataFrame()
    return pd.DataFrame()


def load_high_risk_logs():
    alert_file = LOGS_DIR / 'high_risk_alerts.json'
    if alert_file.exists():
        try:
            return pd.DataFrame(json.loads(alert_file.read_text()))
        except:
            return pd.DataFrame()
    return pd.DataFrame()


def risk_badge(level):
    cls   = {'HIGH': 'badge-high', 'MEDIUM': 'badge-medium', 'LOW': 'badge-low'}.get(level, 'badge-low')
    emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(level, '⚪')
    return f"<span class='{cls}'>{emoji} {level} RISK</span>"


def rec_banner(rec):
    if 'REJECT' in rec.upper():   cls = 'rec-reject'
    elif 'REVIEW' in rec.upper(): cls = 'rec-review'
    else:                          cls = 'rec-approve'
    return f"<div class='{cls}'>{rec}</div>"


# ------------------------------------------------------------------ #
#  Sidebar                                                             #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("## 🏦 LoanSense")
    st.markdown("*Intelligent Loan Risk Scoring*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Predict", "📊 Analytics", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    if check_api():
        st.success("✅ Flask API Online")
    else:
        st.error("❌ Flask API Offline")
        st.caption("Run: `python api/app.py`")

    if check_n8n():
        st.success("✅ n8n Online")
    else:
        st.warning("⚠️ n8n Offline")
        st.caption("Run: `npx n8n`")

    st.markdown("---")
    st.caption("LoanSense v1.0")
    st.caption("Built with NumPy + Flask + n8n")


# ================================================================== #
#  PAGE 1 — HOME                                                       #
# ================================================================== #
if page == "🏠 Home":

    st.markdown("""
    <div class='ls-header'>
        <h1>🏦 LoanSense Dashboard</h1>
        <p>ML-powered loan default prediction | 7 models from scratch | Automated reporting pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    df_logs  = load_logs()
    df_high  = load_high_risk_logs()
    total    = len(df_logs)
    high_risk   = len(df_high)
    low_risk    = len(df_logs[df_logs['risk_level'] == 'LOW'])    if total > 0 else 0
    medium_risk = len(df_logs[df_logs['risk_level'] == 'MEDIUM']) if total > 0 else 0
    approve_rate = round(low_risk / total * 100, 1)               if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Applications", f"{total:,}")
    col2.metric("High Risk Flagged",  f"{high_risk:,}")
    col3.metric("Medium Risk",        f"{medium_risk:,}")
    col4.metric("Approval Rate",      f"{approve_rate}%")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='ls-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Best Model Performance</div>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("AUC-ROC",  "0.8293")
        m2.metric("F1 Score", "42.16%")
        m3, m4 = st.columns(2)
        m3.metric("Recall",    "44.86%")
        m4.metric("Precision", "39.77%")
        st.markdown("**Model:** Decision Tree (hypertuned)")
        st.markdown("**Threshold:** 0.20 (optimized for recall)")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='ls-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Dataset Overview</div>", unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        d1.metric("Total Samples", "149,999")
        d2.metric("Default Rate",  "6.7%")
        d3, d4 = st.columns(2)
        d3.metric("Features",       "10")
        d4.metric("Models Trained", "7")
        st.markdown("**Dataset:** Give Me Some Credit (Kaggle)")
        st.markdown("**Task:** Binary classification (default prediction)")
        st.markdown("</div>", unsafe_allow_html=True)

    if total > 0:
        st.markdown("### 📋 Recent Predictions")
        display_cols = ['timestamp', 'applicant_id', 'risk_level',
                        'default_probability_pct', 'recommendation']
        available = [c for c in display_cols if c in df_logs.columns]
        st.dataframe(df_logs[available].tail(10).iloc[::-1],
                     use_container_width=True, hide_index=True)
    else:
        st.info("No predictions yet. Go to **Predict** to score your first applicant!")


# ================================================================== #
#  PAGE 2 — PREDICT                                                    #
# ================================================================== #
elif page == "🔍 Predict":

    st.markdown("""
    <div class='ls-header'>
        <h1>🔍 Score an Applicant</h1>
        <p>Fill in the applicant details below and get an instant risk assessment</p>
    </div>
    """, unsafe_allow_html=True)

    if not check_api():
        st.error("⚠️ Flask API is offline. Start it with: `python api/app.py`")
        st.stop()

    with st.form("predict_form"):
        st.markdown("### 👤 Applicant Identity")
        applicant_id = st.text_input("Applicant ID", placeholder="e.g. APP-2024-001")

        st.markdown("### 💰 Financial Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            monthly_income = st.number_input("Monthly Income ($)",      min_value=0.0, value=5000.0, step=100.0)
            debt_ratio     = st.number_input("Debt Ratio (0-1)",        min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        with col2:
            revolving_util    = st.number_input("Credit Utilization (0-1)", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
            open_credit_lines = st.number_input("Open Credit Lines",        min_value=0, value=5, step=1)
        with col3:
            real_estate_loans = st.number_input("Real Estate Loans", min_value=0, value=1, step=1)
            dependents        = st.number_input("Dependents",         min_value=0, value=0, step=1)

        st.markdown("### 📅 Personal Information")
        age = st.slider("Age", min_value=18, max_value=100, value=35)

        st.markdown("### ⚠️ Late Payment History")
        col6, col7, col8 = st.columns(3)
        with col6:
            late_30_59 = st.number_input("Late 30-59 Days", min_value=0, value=0, step=1)
        with col7:
            late_60_89 = st.number_input("Late 60-89 Days", min_value=0, value=0, step=1)
        with col8:
            late_90 = st.number_input("Late 90+ Days", min_value=0, value=0, step=1)

        submitted = st.form_submit_button("🔍 Analyze Risk")

    # ── Store payload in session state so n8n button can access it ──
    if submitted:
        if not applicant_id:
            applicant_id = f"APP-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"

        payload = {
            "applicant_id"     : applicant_id,
            "revolving_util"   : revolving_util,
            "age"              : age,
            "late_30_59"       : late_30_59,
            "debt_ratio"       : debt_ratio,
            "monthly_income"   : monthly_income,
            "open_credit_lines": open_credit_lines,
            "late_90"          : late_90,
            "real_estate_loans": real_estate_loans,
            "late_60_89"       : late_60_89,
            "dependents"       : dependents,
        }

        # Save to session state
        st.session_state['last_payload'] = payload

        with st.spinner("Analyzing applicant risk..."):
            result, error = call_predict(payload)

        if error:
            st.error(f"API Error: {error}")
        else:
            # Store result in session state
            st.session_state['last_result'] = result

    # ── Show result if available ──
    if 'last_result' in st.session_state and st.session_state['last_result']:
        result  = st.session_state['last_result']
        payload = st.session_state.get('last_payload', {})

        st.markdown("---")
        st.markdown("## 📋 Risk Assessment Report")

        # ── Parse — handles both Flask direct and n8n wrapped response ──
        risk           = result.get('risk_assessment', result)
        prob           = float(risk.get('default_probability', result.get('default_probability', 0)))
        risk_level     = risk.get('risk_level',     result.get('risk_level', 'LOW'))
        prediction     = risk.get('prediction',     result.get('prediction', 'N/A'))
        recommendation = risk.get('recommendation', result.get('recommendation', ''))
        factors        = result.get('risk_factors', result.get('top_risk_factors', ['N/A', 'N/A', 'N/A']))

        # ── Metrics ──
        c1, c2, c3 = st.columns(3)
        c1.metric("Default Probability", f"{prob*100:.1f}%")
        c2.metric("Risk Level",          risk_level)
        c3.metric("Prediction",          prediction)

        st.markdown(f"<br>{risk_badge(risk_level)}<br>", unsafe_allow_html=True)
        st.markdown("**Default Probability**")
        st.progress(min(prob, 1.0))
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(rec_banner(recommendation), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("<div class='section-title'>Top Risk Factors</div>", unsafe_allow_html=True)
            for f in factors:
                st.markdown(f"<div class='factor-item'>⚠️ {f}</div>", unsafe_allow_html=True)

        with col_r:
            st.markdown("<div class='section-title'>Applicant Summary</div>", unsafe_allow_html=True)
            st.markdown(f"**ID:** {payload.get('applicant_id','N/A')}")
            st.markdown(f"**Age:** {payload.get('age','N/A')}")
            st.markdown(f"**Monthly Income:** ${payload.get('monthly_income',0):,.0f}")
            st.markdown(f"**Debt Ratio:** {payload.get('debt_ratio','N/A')}")
            st.markdown(f"**Credit Utilization:** {payload.get('revolving_util',0)*100:.0f}%")
            st.markdown(f"**Late 90+ Days:** {payload.get('late_90','N/A')}")

        # ── n8n button — outside form, uses session state payload ──
        st.markdown("---")
        st.markdown("**📤 Save report & log via n8n pipeline:**")
        if st.button("Send through n8n Pipeline"):
            n8n_result, status_code, n8n_error = send_to_n8n(payload)
            if n8n_error:
                st.error(f"❌ n8n error: {n8n_error} — Is n8n running and workflow in listen mode?")
            elif status_code == 200:
                st.success("✅ Report saved and logged via n8n pipeline!")
                st.json(n8n_result)
            else:
                st.warning(f"n8n returned status {status_code}")


# ================================================================== #
#  PAGE 3 — ANALYTICS                                                  #
# ================================================================== #
elif page == "📊 Analytics":

    st.markdown("""
    <div class='ls-header'>
        <h1>📊 Analytics</h1>
        <p>Model evaluation charts + live prediction log analysis</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📈 Model Evaluation", "🗂️ Prediction Logs"])

    with tab1:
        st.markdown("### Model Performance Charts")
        st.caption("Generated from test set evaluation across all 7 models")

        charts = [
            ("ROC Curves — All Models",              "roc_curves.png"),
            ("Precision-Recall Curves — All Models", "precision_recall.png"),
            ("Model Comparison — F1, Recall, AUC",   "model_comparison.png"),
            ("Confusion Matrix — Best Model",         "confusion_matrix.png"),
            ("Feature Importance",                    "feature_importance.png"),
        ]

        for i in range(0, len(charts), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(charts):
                    title, filename = charts[i + j]
                    path = ASSETS_DIR / filename
                    with col:
                        st.markdown(f"**{title}**")
                        if path.exists():
                            st.image(str(path), use_container_width=True)
                        else:
                            st.warning(f"`assets/{filename}` not found")

    with tab2:
        st.markdown("### Live Prediction Log Analysis")
        df      = load_logs()
        df_high = load_high_risk_logs()

        if df.empty:
            st.info("No predictions logged yet. Run some predictions first!")
        else:
            total  = len(df)
            n_high = len(df[df['risk_level'] == 'HIGH'])   if 'risk_level' in df.columns else 0
            n_med  = len(df[df['risk_level'] == 'MEDIUM']) if 'risk_level' in df.columns else 0
            n_low  = len(df[df['risk_level'] == 'LOW'])    if 'risk_level' in df.columns else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Logged", total)
            c2.metric("🔴 High Risk", n_high)
            c3.metric("🟡 Medium",    n_med)
            c4.metric("🟢 Low Risk",  n_low)

            st.markdown("<br>", unsafe_allow_html=True)

            if 'risk_level' in df.columns:
                st.markdown("**Risk Level Distribution**")
                st.bar_chart(df['risk_level'].value_counts())

            if 'default_probability' in df.columns:
                st.markdown("**Default Probability Distribution**")
                probs = df['default_probability'].astype(float)
                hist, edges = np.histogram(probs, bins=10)
                labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(hist))]
                st.bar_chart(pd.Series(hist, index=labels))

            st.markdown("**Full Prediction Log**")
            display_cols = ['timestamp', 'applicant_id', 'age', 'monthly_income',
                            'default_probability_pct', 'risk_level', 'recommendation']
            available = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available].iloc[::-1], use_container_width=True, hide_index=True)

            if not df_high.empty:
                st.markdown("### 🔴 High Risk Alerts")
                alert_cols = ['flagged_at', 'applicant_id', 'default_probability_pct',
                              'risk_factor_1', 'risk_factor_2']
                avail = [c for c in alert_cols if c in df_high.columns]
                st.dataframe(df_high[avail].iloc[::-1], use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False)
            st.download_button(
                label     = "⬇️ Download Full Log as CSV",
                data      = csv,
                file_name = "loansense_predictions.csv",
                mime      = "text/csv"
            )


# ================================================================== #
#  PAGE 4 — ABOUT                                                      #
# ================================================================== #
elif page == "ℹ️ About":

    st.markdown("""
    <div class='ls-header'>
        <h1>ℹ️ About LoanSense</h1>
        <p>Technical details, architecture, and project background</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='ls-card'>", unsafe_allow_html=True)
        st.markdown("### 🧠 Models Built From Scratch")
        models = [
            ("Logistic Regression", "MLE + Gradient Descent"),
            ("LDA",                 "Class-conditional Gaussians"),
            ("Naive Bayes",         "Bayes theorem + Gaussian PDF"),
            ("Bayesian Classifier", "MAP estimation"),
            ("Decision Tree",       "Gini impurity + recursive splitting"),
            ("Random Forest",       "Bootstrap aggregation"),
            ("Neural Network",      "Backprop + mini-batch SGD"),
        ]
        for name, concept in models:
            st.markdown(f"**{name}** — *{concept}*")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='ls-card'>", unsafe_allow_html=True)
        st.markdown("### 🏗️ System Architecture")
        st.code("""
Streamlit Dashboard
      ↓
Flask REST API (:5000)
      ↓
Decision Tree Model
(NumPy from scratch)
      ↓
n8n Automation Pipeline
      ↓
HTML Reports + JSON Logs
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='ls-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Dataset")
    st.markdown("""
- **Source:** [Give Me Some Credit — Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)
- **Size:** 149,999 loan applicants
- **Features:** 10 financial & behavioral features
- **Target:** Binary (defaulted in 2 years: yes/no)
- **Class Imbalance:** 93.3% non-default vs 6.7% default
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='ls-card'>", unsafe_allow_html=True)
    st.markdown("### 🔑 Key Technical Achievements")
    for a in [
        "All 7 ML models implemented from mathematical first principles using NumPy only",
        "Handled severe class imbalance (6.7%) through threshold tuning — F1 improved 22% → 42%",
        "Built production REST API with 6 endpoints, input validation, and explainable outputs",
        "Designed full agentic n8n pipeline — raw input to saved HTML report in under 2 seconds",
        "Hypertuned best model across 60 combinations of depth, split size, and threshold",
    ]:
        st.markdown(f"✅ {a}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='ls-card'>", unsafe_allow_html=True)
    st.markdown("### 🛠️ Tech Stack")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**ML**\n\nNumPy\n\nPandas\n\nMatplotlib")
    c2.markdown("**API**\n\nFlask\n\nREST\n\nJSON")
    c3.markdown("**Automation**\n\nn8n\n\nWebhooks\n\nHTML Reports")
    c4.markdown("**Dashboard**\n\nStreamlit\n\nCSV Export")
    st.markdown("</div>", unsafe_allow_html=True)