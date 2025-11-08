import streamlit as st
import pandas as pd
import numpy as np
import cv2
import base64

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
st.sidebar.write("Monitoring live alerts (simulation mode)...")
alert_placeholder = st.sidebar.empty()

# =============================
# 🎥 LIVE FEED SECTION
# =============================
st.markdown("<div class='section-title'>🎥 Live Surveillance Feed</div>", unsafe_allow_html=True)
frame_window = st.empty()

# =============================
# 📊 METRICS
# =============================
col1, col2, col3 = st.columns(3)
col1.metric("Active Cameras", 1)
col2.metric("Detected Objects", 0)
col3.metric("Alerts Triggered", 0)

# ============================================================
# 🎥 LOCAL CAMERA DISPLAY LOOP (NO BACKEND)
# ============================================================
import time

def run_local_camera():
    st.markdown("<div class='section-title'>🎥 Live Local Camera Feed</div>", unsafe_allow_html=True)

    frame_placeholder = st.empty()
    obj_placeholder = st.empty()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error("❌ Could not access webcam.")
        return

    st.success("✅ Live camera started. Press 'Stop Camera' to end.")

    stop_button = st.button("🛑 Stop Camera")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Frame not captured.")
            break

        # ======= Simulate object detection (for now) =======
        # (Replace this with your backend or AI model later)
        height, width, _ = frame.shape
        x, y, w, h = 100, 120, 180, 180
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Person: 98%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # ======= Display live frame =======
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # ======= Update live metrics =======
        obj_placeholder.metric("Detected Objects", 1)

        # ======= Stop condition =======
        if stop_button:
            break

        # small delay to avoid CPU overload
        time.sleep(0.03)

    cap.release()
    st.info("📷 Camera stopped.")



# ============================================================
# 🚀 RUN STREAMING DASHBOARD
# ============================================================
run_local_camera()

# =============================
# 🧾 FOOTER
# =============================
st.markdown("<div class='footer'>© 2025 SentinelAI | Designed by Data Dynamos</div>", unsafe_allow_html=True)
