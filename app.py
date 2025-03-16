import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# 🎨 Streamlit Page Config
st.set_page_config(
    page_title="AI Object Detection",
    page_icon="📸",
    layout="wide"
)

# UI Elements
st.markdown("<h1 class='centered'>🚀 AI Object Detection</h1>", unsafe_allow_html=True)

# Buttons Section
start_btn = st.button("▶️ Start Detection", key="start")
stop_btn = st.button("⏹️ Stop Detection", key="stop")

# Video Frame Placeholder
stframe = st.empty()

# Initialize variables
cap = None

if start_btn:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Camera not found!")
    else:
        st.success("🎥 Camera started!")

while cap and cap.isOpened():
    success, frame = cap.read()
    if not success:
        st.warning("⚠️ Failed to capture frame!")
        break

    # Run YOLOv8
    results = model(frame)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"

            if conf > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Convert to RGB for Streamlit display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    stframe.image(frame, channels="RGB", use_container_width=True)

    # Stop if button is pressed
    if stop_btn:
        cap.release()
        st.warning("🛑 Camera Stopped!")
        break

# 📌 Footer
st.markdown('<p class="footer">Developed by Muhammad Shayan Janjua</p>', unsafe_allow_html=True)
