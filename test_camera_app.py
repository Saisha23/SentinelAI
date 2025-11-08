import streamlit as st
import cv2

st.set_page_config(page_title="Camera Test", layout="wide")

st.title("🎥 Webcam Live Feed Test")

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

st.sidebar.write("Press **Stop** in Streamlit toolbar to end the stream.")

while True:
    ret, frame = camera.read()
    if not ret:
        st.error("Cannot access webcam.")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
