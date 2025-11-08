import streamlit as st
import pandas as pd
import cv2
import requests
import plotly.express as px
import time
import json
import os
from datetime import datetime

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(
    page_title="SentinelAI Command Center",
    layout="wide",
    page_icon="🛰️"
)

# ------------------------- CUSTOM CSS -------------------------
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 42px;
            font-weight: 700;
            color: #00B4D8;
            text-shadow: 0px 0px 12px rgba(0,180,216,0.9);
            letter-spacing: 1px;
        }
        [data-testid="stSidebar"] {
            background-color: #0f1117 !important;
            border-right: 1px solid #00B4D8;
        }
        .section-title {
            font-size: 22px;
            font-weight: 600;
            color: white;
            border-left: 5px solid #00B4D8;
            padding-left: 12px;
            margin-top: 25px;
        }
        .alert-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .alert-table th, .alert-table td {
            border: 1px solid white;
            color: white;
            font-size: 12px;
            padding: 4px 6px;
            text-align: left;
        }
        .alert-table th {
            background-color: #00B4D8;
            color: black;
            position: sticky;
            top: 0;
            z-index: 1;
        }
        .alert-container {
            max-height: 350px;
            overflow-y: auto;
            scrollbar-color: #00B4D8 #0f1117;
            scrollbar-width: thin;
        }
        .footer {
            text-align: center;
            color: #777;
            margin-top: 40px;
            font-size: 13px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------- HEADER -------------------------
st.markdown("<h1 class='main-title'>🛰️ SentinelAI Command Center</h1>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------- BACKEND CONNECTION -------------------------
API_BASE = "http://127.0.0.1:8000/api/zones"
SCREENSHOT_DIR = "dashboard/alert_screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

def fetch_backend_data():
    try:
        zones = requests.get(f"{API_BASE}").json()
    except Exception:
        zones = []
    try:
        metrics = requests.get("http://127.0.0.1:8000/api/metrics").json()
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

# ------------------------- SIDEBAR (Only Alert Logs) -------------------------
st.sidebar.subheader("📋 Alert Logs")

# Initialize alert log with dummy data
if "alert_log" not in st.session_state:
    st.session_state["alert_log"] = [
        {"Time (IST)": "02:31:10", "Alert Type": "Weapon", "Description": "Pistol detected near Gate A"},
        {"Time (IST)": "02:32:44", "Alert Type": "Intrusion", "Description": "Unauthorized entry in Zone 3"},
        {"Time (IST)": "02:34:05", "Alert Type": "Anomaly", "Description": "Sudden crowd movement detected"},
    ]

def save_screenshot(frame, alert_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{alert_type}_{timestamp}.jpg"
    filepath = os.path.join(SCREENSHOT_DIR, filename)
    cv2.imwrite(filepath, frame)

def log_alert(alert_type, description, frame=None):
    new_entry = {
        "Time (IST)": datetime.now().strftime("%H:%M:%S"),
        "Alert Type": alert_type,
        "Description": description
    }
    st.session_state["alert_log"].insert(0, new_entry)
    st.session_state["alert_log"] = st.session_state["alert_log"][:10]

    if frame is not None and alert_type.lower() in ["weapon", "intrusion", "anomaly"]:
        save_screenshot(frame, alert_type)

# Sidebar alert table (fixed header)
alert_table_placeholder = st.sidebar.empty()

def render_alert_table():
    df = pd.DataFrame(st.session_state["alert_log"])
    table_html = "<div class='alert-container'><table class='alert-table'>"
    table_html += "<thead><tr><th>Time (IST)</th><th>Alert Type</th><th>Description</th></tr></thead><tbody>"
    for _, row in df.iterrows():
        table_html += f"<tr><td>{row['Time (IST)']}</td><td>{row['Alert Type']}</td><td>{row['Description']}</td></tr>"
    table_html += "</tbody></table></div>"
    alert_table_placeholder.markdown(table_html, unsafe_allow_html=True)

render_alert_table()

# ------------------------- METRICS -------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Active Zones", len(zones))
col2.metric("Training Status", metrics.get("status", "Idle"))
col3.metric("Epoch", metrics.get("current_epoch", 0))

st.markdown("<div class='section-title'>📉 Model Training Metrics</div>", unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)
m1.metric("Train Loss", round(metrics.get("train_loss", 0.0), 4))
m2.metric("Validation Loss", round(metrics.get("val_loss", 0.0), 4))
m3.metric("Accuracy", f"{metrics.get('accuracy', 0.0)}%")

try:
    df = pd.DataFrame(metrics.get("history", []))
    if not df.empty:
        fig = px.line(df, x="epoch", y=["train_loss", "val_loss", "accuracy"], markers=True,
                      title="Training Progress (Weapon Detection Model)")
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Could not render metrics: {e}")

# ------------------------- ZONE MAP -------------------------
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
        except Exception:
            continue
    if zone_data:
        df_z = pd.DataFrame(zone_data)
        fig = px.scatter_mapbox(df_z, lat="Latitude", lon="Longitude", color="Zone", zoom=3, height=400)
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

# ------------------------- LIVE CAMERA FEED -------------------------
st.markdown("<div class='section-title'>🎥 Live Surveillance Feed (YOLOv8 + Anomaly Detection)</div>", unsafe_allow_html=True)
frame_placeholder = st.empty()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Could not access webcam.")
else:
    st.success("Live camera active. Continuous monitoring in progress...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Camera disconnected.")
            break

        _, img_encoded = cv2.imencode(".jpg", frame)
        try:
            response = requests.post(f"{API_BASE}/detect", files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")})
            detections = response.json().get("detections", [])
        except Exception:
            detections = []

        for det in detections:
            label = det.get("type", "")
            desc = ""

            if label.lower() == "weapon":
                desc = "Weapon detected near restricted zone."
            elif label.lower() == "intrusion":
                desc = "Unauthorized entry detected in secure area."
            elif label.lower() == "anomaly":
                desc = "Abnormal crowd behavior or motion detected."
            else:
                desc = "Unidentified object detected."

            log_alert(label, desc, frame)
            cv2.putText(frame, f"{label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        render_alert_table()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        time.sleep(0.2)

    cap.release()

# ------------------------- FOOTER -------------------------
st.markdown("<div class='footer'>© 2025 SentinelAI | Designed by Data Dynamos</div>", unsafe_allow_html=True)
