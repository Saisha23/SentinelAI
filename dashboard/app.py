import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import asyncio
import websockets
import json
import cv2
import base64
from PIL import Image
import threading

st.set_page_config(
    page_title="SentinelAI Command Center",
    layout="wide",
    page_icon="🛰️"
)

st.markdown("<h1 style='text-align:center; color:#00B4D8;'>🛰️ SentinelAI Command Center</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("📋 Live Alerts")
st.sidebar.write("Monitoring live alerts from FastAPI backend...")

alert_placeholder = st.sidebar.empty()
log_data = []

# Live feed placeholder
st.subheader("🎥 Live Surveillance Feed")
video_placeholder = st.empty()

# Metrics
col1, col2, col3 = st.columns(3)
cam_metric = col1.metric("Active Cameras", 1)
obj_metric = col2.metric("Detected Objects", 0)
alert_metric = col3.metric("Alerts Triggered", 0)

# Globals for live data
latest_frame = None
latest_alert = None

# --- Function to decode base64 frames ---
def decode_frame(base64_str):
    frame_bytes = base64.b64decode(base64_str)
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# --- WebSocket Listener for /ws/stream ---
async def listen_stream():
    uri = "ws://localhost:8000/ws/stream"
    async with websockets.connect(uri) as websocket:
        while True:
            data = await websocket.recv()
            results = json.loads(data)
            
            # Assuming backend sends {'frame': base64_str, 'detections': [...]} format
            if 'frame' in results:
                frame = decode_frame(results['frame'])
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            if 'detections' in results:
                obj_metric.metric("Detected Objects", len(results['detections']))

# --- WebSocket Listener for /ws/alerts ---
async def listen_alerts():
    uri = "ws://localhost:8000/ws/alerts"
    async with websockets.connect(uri) as websocket:
        while True:
            data = await websocket.recv()
            alert = json.loads(data)
            
            # Example alert: {'type': 'Weapon', 'confidence': 0.91, 'zone': 'Restricted Area'}
            log_data.insert(0, alert)
            if len(log_data) > 10:
                log_data.pop()
            
            alert_df = pd.DataFrame(log_data)
            alert_placeholder.dataframe(alert_df, use_container_width=True)
            alert_metric.metric("Alerts Triggered", len(log_data))

# --- Run both connections in parallel ---
def start_async_loops():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [
        loop.create_task(listen_stream()),
        loop.create_task(listen_alerts())
    ]
    loop.run_until_complete(asyncio.wait(tasks))

# Start the threads
threading.Thread(target=start_async_loops, daemon=True).start()
