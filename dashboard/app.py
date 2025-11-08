import streamlit as st
import pandas as pd
import numpy as np
import cv2
import requests
import plotly.express as px
import time
import json

# =============================
# 🌐 STREAMLIT PAGE SETUP
# =============================
st.set_page_config(
    page_title="SentinelAI Command Center",
    layout="wide",
    page_icon="🛰️"
)

# =============================
# 💅 CUSTOM STYLES
# =============================
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 42px;
            font-weight: 700;
            color: #00B4D8;
            text-shadow: 0px 0px 12px rgba(0, 180, 216, 0.9);
            letter-spacing: 1px;
        }
        [data-testid="stSidebar"] {
            background-color: #0f1117 !important;
            border-right: 1px solid #00B4D8;
        }
        div[data-testid="stMetricValue"] {
            color: #00B4D8 !important;
            font-size: 28px !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #ccc !important;
            font-size: 15px !important;
        }
        .section-title {
            font-size: 22px;
            font-weight: 600;
            color: white;
            border-left: 5px solid #00B4D8;
            padding-left: 12px;
            margin-top: 25px;
        }
        img {
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0,180,216,0.3);
        }
        .footer {
            text-align: center;
            color: #777;
            margin-top: 40px;
            font-size: 13px;
        }
    </style>
""", unsafe_allow_html=True)

# =============================
# 🛰️ TITLE
# =============================
st.markdown("<h1 class='main-title'>🛰️ SentinelAI Command Center</h1>", unsafe_allow_html=True)
st.markdown("---")

# =============================
# 🌐 BACKEND API BASE URL
# =============================
API_BASE = "http://127.0.0.1:8000/api"

# =============================
# 📊 FETCH BACKEND DATA
# =============================
def fetch_backend_data():
    try:
        zones = requests.get(f"{API_BASE}/zones").json()
    except Exception:
        zones = []

    try:
        metrics = requests.get(f"{API_BASE}/metrics").json()
    except Exception:
        metrics = {
            "current_epoch": 0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "accuracy": 0.0,
            "status": "Idle"
        }

    return zones, metrics

zones, metrics = fetch_backend_data()

# =============================
# 🧠 SIDEBAR ALERT PANEL
# =============================
st.sidebar.header("🚨 Alert Controls")
alert_type = st.sidebar.selectbox("Select Alert Type", ["All", "Weapon", "Intrusion", "Anomaly"])
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Alert Logs")

alert_logs = pd.DataFrame({
    "Time": ["12:00", "12:05", "12:10"],
    "Type": ["Weapon", "Intrusion", "Anomaly"],
    "Confidence": ["98%", "92%", "87%"]
})
st.sidebar.table(alert_logs)

# =============================
# 🔢 DASHBOARD METRICS
# =============================
col1, col2, col3 = st.columns(3)
col1.metric("Active Zones", len(zones))
col2.metric("Training Status", metrics.get("status", "Idle"))
col3.metric("Epoch", metrics.get("current_epoch", 0))

st.markdown("<div class='section-title'>📉 Model Training Metrics</div>", unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)
m1.metric("Train Loss", round(metrics.get("train_loss", 0.0), 4))
m2.metric("Validation Loss", round(metrics.get("val_loss", 0.0), 4))
m3.metric("Accuracy", f"{metrics.get('accuracy', 0.0)}%")

# =============================
# 📈 TRAINING CHART
# =============================
try:
    df = pd.DataFrame(metrics.get("history", []))
    if not df.empty:
        fig = px.line(df, x="epoch", y=["train_loss", "val_loss", "accuracy"], markers=True,
                      title="Training Progress (Weapon Detection Model)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No training data recorded yet.")
except Exception as e:
    st.warning(f"Could not render metrics: {e}")

# =============================
# 🗺️ ZONE OVERVIEW (FIXED)
# =============================
st.markdown("<div class='section-title'>🗺️ Zone Overview</div>", unsafe_allow_html=True)

if zones:
    zone_data = []
    for z in zones:
        try:
            polygon = z.get("polygon") if isinstance(z, dict) else None
            if isinstance(polygon, str):
                polygon = json.loads(polygon)
            if polygon:
                for point in polygon:
                    zone_data.append({"Zone": z.get("name", "Unknown"), "Latitude": point[0], "Longitude": point[1]})
        except Exception as e:
            continue

    if zone_data:
        df_z = pd.DataFrame(zone_data)
        fig = px.scatter_mapbox(df_z, lat="Latitude", lon="Longitude", color="Zone", zoom=3, height=400)
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid zone data available.")
else:
    st.info("No zones configured yet.")

# =============================
# 🎥 LIVE SURVEILLANCE FEED
# =============================
st.markdown("<div class='section-title'>🎥 Live Surveillance Feed</div>", unsafe_allow_html=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frame_placeholder = st.empty()

if not cap.isOpened():
    st.error("❌ Could not access webcam.")
else:
    st.success("✅ Camera active. Press Stop to end.")
    stop_btn = st.button("🛑 Stop Camera")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_btn:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        time.sleep(0.03)
    cap.release()
    st.info("📷 Camera stopped.")

# =============================
# 🧾 FOOTER
# =============================
st.markdown("<div class='footer'>© 2025 SentinelAI | Integrated Dashboard | Designed by Data Dynamos</div>", unsafe_allow_html=True)
