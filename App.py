# ============================================================
# Heart Disease Prediction - ProHealth Style Dashboard
# Author: M V Kiran | Version: Final
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import time

st.set_page_config(
    page_title="CardioAI - Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

.stApp { background: #f0f4f8; }
#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stSidebarNav"] { display: none !important; }

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8edf2 !important;
    box-shadow: 4px 0 20px rgba(0,0,0,0.05) !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }

/* SIDEBAR ITEMS */
.sb-logo { padding: 24px 20px 18px; border-bottom: 1px solid #f3f4f6; margin-bottom: 8px; text-align:center; }
.sb-logo-icon { font-size: 2.5rem; }
.sb-brand { font-size: 1.2rem; font-weight: 900; color: #1a1a2e; margin-top: 6px; }
.sb-brand span { color: #2563eb; }
.sb-sub { font-size: 0.72rem; color: #9ca3af; margin-top: 2px; text-transform: uppercase; letter-spacing: 1px; }
.sb-sec { padding: 14px 16px 6px; font-size: 0.65rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; }
.sb-item { display: flex; align-items: center; gap: 10px; padding: 10px 18px; margin: 2px 8px; border-radius: 12px; cursor: pointer; font-size: 0.88rem; font-weight: 500; color: #6b7280; border: 1px solid transparent; transition: all 0.2s; }
.sb-item:hover { background: #f3f4f6; color: #1a1a2e; }
.sb-item.active { background: #eff6ff; color: #2563eb; border-color: #bfdbfe; font-weight: 600; }
.sb-ico { font-size: 1rem; width: 20px; text-align: center; }
.sb-bdg { margin-left: auto; font-size: 0.65rem; font-weight: 700; padding: 2px 7px; border-radius: 10px; }
.sb-bdg-blue { background: #dbeafe; color: #1d4ed8; }
.sb-bdg-red { background: #fee2e2; color: #991b1b; }
.sb-div { height: 1px; background: #f3f4f6; margin: 8px 16px; }
.sb-stats { margin: 10px 12px; background: #f9fafb; border-radius: 14px; padding: 14px; border: 1px solid #f3f4f6; }
.sb-stat-row { display: flex; justify-content: space-between; padding: 5px 0; }
.sb-stat-k { font-size: 0.78rem; color: #6b7280; }
.sb-stat-v { font-size: 0.82rem; font-weight: 700; color: #2563eb; }
.sb-user { margin: 10px 12px; background: #eff6ff; border-radius: 14px; padding: 14px; border: 1px solid #bfdbfe; display: flex; align-items: center; gap: 10px; }
.sb-uav { width: 38px; height: 38px; background: linear-gradient(135deg,#2563eb,#1d4ed8); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0; }
.sb-uname { font-size: 0.88rem; font-weight: 700; color: #1a1a2e; }
.sb-urole { font-size: 0.72rem; color: #2563eb; }
.sb-foot { text-align: center; padding: 16px; font-size: 0.7rem; color: #9ca3af; }

/* NAVBAR */
.navbar { display: flex; align-items: center; justify-content: space-between; padding: 14px 36px; background: #ffffff; border-bottom: 1px solid #eef1f5; box-shadow: 0 2px 12px rgba(0,0,0,0.04); position: sticky; top: 0; z-index: 999; }
.brand { display: flex; align-items: center; gap: 10px; }
.brand-icon { width: 36px; height: 36px; background: linear-gradient(135deg, #2563eb, #1d4ed8); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px; }
.brand-name { font-size: 1.15rem; font-weight: 800; color: #1a1a2e; }
.brand-name span { color: #2563eb; }
.nav-tabs { display: flex; gap: 4px; }
.nav-tab { padding: 7px 18px; border-radius: 20px; font-size: 0.88rem; font-weight: 500; color: #6b7280; cursor: pointer; border: 1px solid transparent; background: transparent; }
.nav-tab.active { background: #2563eb; color: white; border-color: #2563eb; }
.nav-right { display: flex; align-items: center; gap: 12px; }
.nav-icon-btn { width: 36px; height: 36px; border-radius: 50%; background: #f3f4f6; border: 1px solid #e5e7eb; display: flex; align-items: center; justify-content: center; font-size: 16px; cursor: pointer; }
.nav-avatar { width: 38px; height: 38px; border-radius: 50%; background: linear-gradient(135deg, #2563eb, #1d4ed8); display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.9rem; color: white; border: 2px solid #dbeafe; }
.nav-consult { background: #2563eb; color: white !important; padding: 9px 20px; border-radius: 25px; font-size: 0.88rem; font-weight: 600; border: none; cursor: pointer; box-shadow: 0 4px 14px rgba(37,99,235,0.35); }

/* STAT CARDS */
.stat-card { background: white; border-radius: 18px; padding: 18px 20px; border: 1px solid #eef1f5; box-shadow: 0 2px 10px rgba(0,0,0,0.04); transition: all 0.25s; }
.stat-card:hover { box-shadow: 0 6px 24px rgba(0,0,0,0.08); transform: translateY(-2px); }
.stat-icon { width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; margin-bottom: 12px; }
.stat-num { font-size: 1.7rem; font-weight: 900; color: #1a1a2e; line-height: 1; }
.stat-lbl { font-size: 0.75rem; color: #9ca3af; margin-top: 4px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-change { font-size: 0.75rem; font-weight: 600; margin-top: 6px; color: #10b981; }

/* HEART VISUAL */
.heart-visual { background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 24px; padding: 30px; text-align: center; border: 1px solid #bfdbfe; position: relative; overflow: hidden; }
.heart-bg { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 180px; opacity: 0.07; animation: softpulse 2s ease-in-out infinite; }
@keyframes softpulse { 0%,100% { transform: translate(-50%,-50%) scale(1); } 50% { transform: translate(-50%,-50%) scale(1.05); } }
.heart-main { font-size: 90px; position: relative; z-index: 1; animation: heartbeat 1.5s ease-in-out infinite; filter: drop-shadow(0 8px 24px rgba(37,99,235,0.25)); }
@keyframes heartbeat { 0%,100% { transform: scale(1); } 14% { transform: scale(1.1); } 28% { transform: scale(1); } 42% { transform: scale(1.06); } }
.heart-stat-pill { display: inline-flex; align-items: center; gap: 8px; background: white; border-radius: 20px; padding: 8px 16px; margin: 6px 4px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); font-size: 0.82rem; font-weight: 600; color: #1a1a2e; border: 1px solid #eef1f5; }

/* VITAL CARDS */
.vital-card { background: white; border-radius: 16px; padding: 16px 18px; border: 1px solid #eef1f5; box-shadow: 0 2px 8px rgba(0,0,0,0.04); display: flex; align-items: center; gap: 14px; transition: all 0.2s; }
.vital-card:hover { border-color: #bfdbfe; box-shadow: 0 4px 16px rgba(37,99,235,0.08); }
.vital-dot { width: 44px; height: 44px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; flex-shrink: 0; }
.vital-name { font-size: 0.78rem; color: #9ca3af; font-weight: 500; }
.vital-val { font-size: 1.05rem; font-weight: 700; color: #1a1a2e; }
.vital-badge { margin-left: auto; font-size: 0.7rem; font-weight: 600; padding: 3px 10px; border-radius: 20px; }
.vb-normal { background: #d1fae5; color: #065f46; }
.vb-high { background: #fee2e2; color: #991b1b; }
.vb-warn { background: #fef3c7; color: #92400e; }

/* FORM */
.form-sec { background: white; border-radius: 18px; padding: 20px; border: 1px solid #eef1f5; box-shadow: 0 2px 10px rgba(0,0,0,0.04); margin-bottom: 16px; }
.form-sec-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; color: #2563eb; font-weight: 700; margin-bottom: 16px; }

/* ECG BAR */
.ecg-bar { background: #1a1a2e; border-radius: 16px; padding: 16px 20px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; }
.ecg-lbl { font-size: 0.72rem; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.5px; }
.ecg-val { font-size: 1rem; font-weight: 700; color: white; }
.ecg-wave { font-size: 1.4rem; letter-spacing: -3px; color: #2563eb; animation: blink 1.5s infinite; }
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.4;} }

/* RESULT CARDS */
.result-safe { background: linear-gradient(135deg, #ecfdf5, #d1fae5); border: 2px solid #6ee7b7; border-radius: 20px; padding: 24px; text-align: center; margin-bottom: 16px; box-shadow: 0 4px 20px rgba(16,185,129,0.12); }
.result-danger { background: linear-gradient(135deg, #fff1f2, #fee2e2); border: 2px solid #fca5a5; border-radius: 20px; padding: 24px; text-align: center; margin-bottom: 16px; box-shadow: 0 4px 20px rgba(239,68,68,0.12); }
.result-pending { background: linear-gradient(135deg, #eff6ff, #dbeafe); border: 2px dashed #93c5fd; border-radius: 20px; padding: 30px; text-align: center; margin-bottom: 16px; }
.result-emoji { font-size: 3rem; margin-bottom: 10px; }
.result-title-safe { font-size: 1.4rem; font-weight: 900; color: #065f46; margin-bottom: 6px; }
.result-title-danger { font-size: 1.4rem; font-weight: 900; color: #991b1b; margin-bottom: 6px; }
.result-title-pending { font-size: 1.2rem; font-weight: 700; color: #1d4ed8; margin-bottom: 6px; }
.result-sub { font-size: 0.85rem; color: #6b7280; }

/* GAUGE */
.gauge-card { background: white; border-radius: 18px; padding: 20px; border: 1px solid #eef1f5; box-shadow: 0 2px 10px rgba(0,0,0,0.04); margin-bottom: 14px; }
.gauge-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; color: #9ca3af; font-weight: 600; margin-bottom: 14px; }
.gauge-track { height: 10px; background: #f3f4f6; border-radius: 5px; overflow: hidden; margin: 14px 0 6px 0; }
.gauge-ticks { display: flex; justify-content: space-between; font-size: 0.68rem; color: #9ca3af; }

/* PROB */
.prob-card { background: white; border-radius: 18px; padding: 18px 20px; border: 1px solid #eef1f5; box-shadow: 0 2px 10px rgba(0,0,0,0.04); margin-bottom: 14px; }
.prob-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; color: #9ca3af; font-weight: 600; margin-bottom: 14px; }
.prob-track { height: 8px; background: #f3f4f6; border-radius: 4px; overflow: hidden; margin-bottom: 14px; }

/* REPORT */
.report-card { background: white; border-radius: 18px; padding: 18px 20px; border: 1px solid #eef1f5; box-shadow: 0 2px 10px rgba(0,0,0,0.04); margin-bottom: 14px; }
.report-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; color: #9ca3af; font-weight: 600; margin-bottom: 14px; }
.report-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #f9fafb; }
.report-row:last-child { border-bottom: none; }
.report-key { font-size: 0.82rem; color: #6b7280; }
.report-val { font-size: 0.82rem; font-weight: 600; color: #1a1a2e; }
.rb { font-size: 0.7rem; padding: 2px 8px; border-radius: 10px; font-weight: 600; margin-left: 6px; }
.rb-g { background: #d1fae5; color: #065f46; }
.rb-r { background: #fee2e2; color: #991b1b; }
.rb-y { background: #fef3c7; color: #92400e; }

/* DOCTOR */
.doctor-card { background: linear-gradient(135deg, #eff6ff, #dbeafe); border: 1px solid #bfdbfe; border-radius: 18px; padding: 18px 20px; display: flex; align-items: flex-start; gap: 14px; margin-bottom: 14px; }
.doctor-avatar { width: 50px; height: 50px; border-radius: 16px; background: linear-gradient(135deg, #2563eb, #1d4ed8); display: flex; align-items: center; justify-content: center; font-size: 22px; flex-shrink: 0; box-shadow: 0 4px 14px rgba(37,99,235,0.3); }
.doctor-name { font-size: 0.95rem; font-weight: 700; color: #1a1a2e; }
.doctor-role { font-size: 0.75rem; color: #2563eb; margin-bottom: 6px; font-weight: 600; }
.doctor-text { font-size: 0.82rem; color: #6b7280; line-height: 1.5; }

/* TIPS */
.tip-row { display: flex; align-items: center; gap: 10px; padding: 10px 14px; background: #f9fafb; border-radius: 10px; margin-bottom: 7px; border: 1px solid #f3f4f6; transition: all 0.2s; }
.tip-row:hover { background: #eff6ff; border-color: #bfdbfe; }
.tip-ico { font-size: 1rem; width: 22px; text-align: center; }
.tip-txt { font-size: 0.82rem; color: #4b5563; }

/* STREAMLIT OVERRIDES */
.stSlider label { color: #374151 !important; font-size: 0.82rem !important; font-weight: 500 !important; }
.stSelectbox label { color: #374151 !important; font-size: 0.82rem !important; font-weight: 500 !important; }
.stSelectbox > div > div { background: #f9fafb !important; border: 1px solid #e5e7eb !important; border-radius: 10px !important; color: #1a1a2e !important; }
.stRadio label { color: #374151 !important; font-size: 0.85rem !important; font-weight: 500 !important; }
.stRadio > div { background: #f9fafb; border-radius: 12px; padding: 10px; border: 1px solid #e5e7eb; }
.stRadio [data-testid="stMarkdownContainer"] p { color: #374151 !important; font-size: 0.85rem !important; }
.stButton > button { background: linear-gradient(135deg, #2563eb, #1d4ed8) !important; color: white !important; font-size: 1rem !important; font-weight: 700 !important; border-radius: 14px !important; border: none !important; padding: 14px 30px !important; width: 100% !important; box-shadow: 0 5px 20px rgba(37,99,235,0.35) !important; transition: all 0.3s !important; }
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 10px 30px rgba(37,99,235,0.45) !important; }
[data-testid="metric-container"] { background: white !important; border: 1px solid #eef1f5 !important; border-radius: 14px !important; padding: 14px !important; box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important; }
[data-testid="metric-container"] label { color: #9ca3af !important; font-size: 0.75rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #1a1a2e !important; font-size: 1.4rem !important; font-weight: 800 !important; }
hr { border-color: #f3f4f6 !important; }
</style>
""", unsafe_allow_html=True)

# -------------------- Train Models --------------------
@st.cache_resource
def train_models():
    df = pd.read_csv("heart.csv")
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    lr_acc = lr.score(X_test, y_test)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_acc = knn.score(X_test, y_test)
    return lr, knn, scaler, lr_acc, knn_acc

lr_model, knn_model, scaler, lr_acc, knn_acc = train_models()
# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown(f"""
    <div class='sb-logo'>
        <div class='sb-logo-icon'>🫀</div>
        <div class='sb-brand'>Cardio<span>AI</span></div>
        <div class='sb-sub'>Medical AI Platform</div>
    </div>

    <div class='sb-sec'>Main Menu</div>
    <div class='sb-item active'><span class='sb-ico'>📊</span> Overview <span class='sb-bdg sb-bdg-blue'>Live</span></div>
    <div class='sb-item'><span class='sb-ico'>🫀</span> Heart Analysis</div>
    <div class='sb-item'><span class='sb-ico'>📋</span> Patient Reports <span class='sb-bdg sb-bdg-red'>3</span></div>
    <div class='sb-item'><span class='sb-ico'>📅</span> History</div>
    <div class='sb-item'><span class='sb-ico'>📈</span> Analytics</div>

    <div class='sb-div'></div>
    <div class='sb-sec'>Tools</div>
    <div class='sb-item'><span class='sb-ico'>🤖</span> AI Models</div>
    <div class='sb-item'><span class='sb-ico'>🔬</span> Lab Results <span class='sb-bdg sb-bdg-red'>New</span></div>
    <div class='sb-item'><span class='sb-ico'>💊</span> Medications</div>
    <div class='sb-item'><span class='sb-ico'>⚙️</span> Settings</div>

    <div class='sb-div'></div>
    <div class='sb-sec'>Model Performance</div>
    <div class='sb-stats'>
        <div class='sb-stat-row'><span class='sb-stat-k'>🔵 Logistic Reg</span><span class='sb-stat-v'>{lr_acc*100:.2f}%</span></div>
        <div class='sb-stat-row'><span class='sb-stat-k'>🟢 KNN (K=5)</span><span class='sb-stat-v'>{knn_acc*100:.2f}%</span></div>
        <div class='sb-stat-row'><span class='sb-stat-k'>🗃️ Patients</span><span class='sb-stat-v'>1,025</span></div>
        <div class='sb-stat-row'><span class='sb-stat-k'>🔬 Features</span><span class='sb-stat-v'>13</span></div>
    </div>

    <div class='sb-div'></div>
    <div class='sb-user'>
        <div class='sb-uav'>👨‍⚕️</div>
        <div><div class='sb-uname'>M V Kiran</div><div class='sb-urole'>ML Developer · 2025</div></div>
    </div>
    <div class='sb-foot'>CardioAI v1.0 · Educational Use Only</div>
    """, unsafe_allow_html=True)
    
# ==================== NAVBAR ====================
st.markdown("""
<div class='navbar'>
    <div class='brand'>
        <div class='brand-icon'>🫀</div>
        <div class='brand-name'>Cardio<span>AI</span></div>
    </div>
    <div class='nav-tabs'>
        <div class='nav-tab active'>Overview</div>
        <div class='nav-tab'>Document</div>
        <div class='nav-tab'>Messages</div>
        <div class='nav-tab'>Labs</div>
    </div>
    <div class='nav-right'>
        <div class='nav-icon-btn'>💬</div>
        <div class='nav-icon-btn'>🔔</div>
        <button class='nav-consult'>Consultation ↗</button>
        <div class='nav-avatar'>MK</div>
    </div>
</div>
""", unsafe_allow_html=True)
