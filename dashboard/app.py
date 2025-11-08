import streamlit as st
import pandas as pd
import numpy as np
import cv2
import base64
import requests
import plotly.express as px
import time

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
# 📋 SIDEBAR - ALERTS
# =============================
st.sidebar.header("📋 Live Alerts")
st.sidebar.write("Monitoring live alerts (connected to backend)...")
alert_placeholder = st.sidebar.empty()

# =============================
# 🌐 BACKEND API URL
# =============================
API_BASE = "http://127.0.0.1:8000/api"

# =============================
# 🧠 FETCH ZONE & METRIC DATA
# =============================
def fetch_backend_data():
    try:
        zones = requests.get(f"{API_BASE}/zones").json()
        alerts = np.random.randint(0, 5)
        return zones, alerts
    except Exception as e:
        st.warning("⚠️ Could not fetch data from backend.")
        return [], 0

zones, alerts = fetch_backend_data()

# =============================
# 📊 TOP METRICS
# =============================
col1, col2, col3 = st.columns(3)
col1.metric("Active Zones", len(zones))
col2.metric("Active Cameras", 1)
col3.metric("Alerts Triggered", alerts)

# =============================
# 🗺️ ZONE MAP DISPLAY
# =============================
st.markdown("<div class='section-title'>🗺️ Zone Overview</div>", unsafe_allow_html=True)

if len(zones) > 0:
    zone_data = []
    for z in zones:
        for point in z["polygon"]:
            zone_data.append({"Zone": z["name"], "Latitude": point[0], "Longitude": point[1]})
    df = pd.DataFrame(zone_data)
    fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color="Zone", zoom=3, height=400)
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No zones configured yet. Add zones from backend or database.")

# =============================
# 🎥 LIVE CAMERA SIMULATION
# =============================
st.markdown("<div class='section-title'>🎥 Live Local Camera Feed</div>", unsafe_allow_html=True)

frame_placeholder = st.empty()
obj_placeholder = st.empty()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    st.error("❌ Could not access webcam.")
else:
    st.success("✅ Live camera started. Press 'Stop Camera' to end.")
    stop_button = st.button("🛑 Stop Camera")

    detected_objects = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_button:
            break

        # ======= Simulated Detection =======
        height, width, _ = frame.shape
        x, y, w, h = 100, 120, 180, 180
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Person: 97%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        detected_objects += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        obj_placeholder.metric("Detected Objects", detected_objects)

        time.sleep(0.03)

    cap.release()
    st.info("📷 Camera stopped.")

# =============================
# 📉 TRAINING PERFORMANCE (SIMULATION)
# =============================
st.markdown("<div class='section-title'>📉 Model Training Performance</div>", unsafe_allow_html=True)
epochs = np.arange(1, 11)
loss = np.random.rand(10)
acc = np.random.uniform(70, 99, 10)
df_perf = pd.DataFrame({"Epoch": epochs, "Loss": loss, "Accuracy": acc})

fig = px.line(df_perf, x="Epoch", y=["Loss", "Accuracy"], markers=True, title="Training Progress Over Epochs")
st.plotly_chart(fig, use_container_width=True)

# =============================
# 🧾 FOOTER
# =============================
st.markdown("<div class='footer'>© 2025 SentinelAI | Designed by Data Dynamos</div>", unsafe_allow_html=True)
