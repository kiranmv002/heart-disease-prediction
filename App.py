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
