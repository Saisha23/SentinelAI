# SentinelAI Dashboard (Streamlit)

**Member 5 – Dashboard & Visualization Developer**

## Overview
This module builds the **SentinelAI Command Center Dashboard** using **Streamlit**, providing a unified interface for visualizing live surveillance feeds, alerts, and analytics.

## Features
- Real-time video feed display (from FastAPI WebSocket)
- Alert logs with timestamps and confidence scores  
- Dynamic system metrics (FPS, latency, active cameras)  
- Heatmaps and alert-type analytics  
- Clean, responsive dark-theme UI  

## Folder Structure
dashboard/
│
├── app.py # Main Streamlit app
├── assets/
│ └── demo_feed.jpg # Placeholder image for live feed
└── README.md # Dashboard documentation


## Run Instructions
```bash
# 1. Go to dashboard folder
cd dashboard

# 2. Install dependencies
pip install streamlit plotly pillow pandas numpy --user

# 3. Run the dashboard
python -m streamlit run app.py
Integration

Member 4 (Backend) → Provides FastAPI WebSocket feed.

Member 1 – 3 → Model outputs (Detections, Tracking, Anomalies) visualized here.