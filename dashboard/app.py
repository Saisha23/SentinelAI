import streamlit as st
import pandas as pd
import numpy as np
import time
from PIL import Image
import plotly.express as px

# --- Page Config ---
st.set_page_config(
    page_title="SentinelAI Command Center",
    layout="wide",
    page_icon="🛰️"
)

# --- Header ---
st.markdown("<h1 style='text-align:center; color:#00B4D8;'>🛰️ SentinelAI Command Center</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("📋 Alert Controls")
st.sidebar.write("Filter and monitor active alerts in real time.")

alert_filter = st.sidebar.selectbox(
    "Select Alert Type:",
    ["All", "Weapon", "Intrusion", "Anomaly", "Suspicious Crowd"]
)
st.sidebar.markdown("### Alert Logs")
dummy_alerts = pd.DataFrame({
    "Time": ["12:00", "12:05", "12:10"],
    "Type": ["Weapon", "Intrusion", "Anomaly"],
    "Confidence": [0.91, 0.84, 0.76]
})
st.sidebar.dataframe(dummy_alerts, use_container_width=True)

# --- Layout: Main Columns ---
col1, col2 = st.columns([2, 1])

# --- Live Feed Placeholder ---
with col1:
    st.subheader("🎥 Live Surveillance Feed")
    placeholder_image = Image.open("assets/demo_feed.jpg")
    st.image(placeholder_image, caption="Simulated NSG Camera Feed", use_column_width=True)

# --- Stats Panel ---
with col2:
    st.subheader("📊 System Status")
    st.metric("Active Cameras", 4)
    st.metric("Detected Objects", 12)
    st.metric("Alerts Triggered", 3)
    st.metric("Average FPS", "14.6")
    st.metric("Latency", "1.9s")

# --- Tabs for Analytics & Heatmaps ---
tab1, tab2 = st.tabs(["Heatmap View", "Alert Analytics"])

with tab1:
    st.write("🗺️ Placeholder for Heatmap visualization.")
    heat_data = pd.DataFrame(np.random.randn(50, 2), columns=["x", "y"])
    fig = px.density_heatmap(heat_data, x="x", y="y", nbinsx=20, nbinsy=20, color_continuous_scale="blues")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.write("📈 Alert trends and distribution over time.")
    alert_counts = pd.DataFrame({
        "Alert Type": ["Weapon", "Intrusion", "Anomaly"],
        "Count": [5, 3, 2]
    })
    fig2 = px.bar(alert_counts, x="Alert Type", y="Count", color="Alert Type", title="Alerts Summary")
    st.plotly_chart(fig2, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align:center; color:grey;'>© 2025 SentinelAI – Data Dynamos Hack-O-Octo 3.0</p>", unsafe_allow_html=True)
