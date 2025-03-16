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

# Custom CSS for background and styling
st.markdown(
    """
    <style>
    /* Background Image */
    .stApp {
        background: url("https://images.unsplash.com/photo-1557683316-973673baf926");
        background-size: cover;
        background-position: center;
    }
    
    /* Centered Title */
    .centered {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #00D4FF;
        padding: 20px;
    }
    
    /* Buttons Styling */
    .button-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 30px;
    }
    
    .styled-button {
        width: 220px;
        height: 55px;
        font-size: 18px;
        font-weight: bold;
        background-color: #00D4FF;
        color: #1E1E1E;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: 0.3s ease-in-out;
        text-align: center;
        margin-bottom: 15px;
    }
    
    .styled-button:hover {
        background-color: #008CBA;
        color: #FFFFFF;
        transform: scale(1.05);
    }
    
    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        font-size: 16px;
        background: #222;
        color: #fff;
        border-top: 2px solid #444;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# UI Elements
st.markdown("<h1 class='centered'>🚀 AI Object Detection</h1>", unsafe_allow_html=True)

# Adding a short description
st.markdown(
    "<p style='text-align:center; font-size:20px; color:#A9A9A9;'>An advanced AI-powered system for real-time object detection.</p>",
    unsafe_allow_html=True
)

# Camera Selection
camera_option = st.radio("Select Camera:", ["Front Camera (0)", "Back Camera (1)"], horizontal=True)
camera_index = 0 if camera_option == "Front Camera (0)" else 1

# Buttons Section (Aligned Below Each Other)
st.markdown("<div class='button-container'>", unsafe_allow_html=True)
start_btn = st.button("▶️ Start Detection", key="start", help="Detection Start")
stop_btn = st.button("⏹️ Stop Detection", key="stop", help="Stop Detection")
st.markdown("</div>", unsafe_allow_html=True)

# Video Display Placeholder
stframe = st.empty()  

if start_btn:
    cap = cv2.VideoCapture(camera_index)  # Open selected camera

    if not cap.isOpened():
        st.error("⚠️ Camera not found! Try switching the camera.")
    else:
        st.success("🎥 Camera started!")

        while True:
            success, frame = cap.read()
            if not success:
                st.error("⚠️ Unable to access camera!")
                break

            # Run YOLOv8 Model
            results = model(frame)

            # Draw Bounding Boxes
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"{model.names[cls]} {conf:.2f}"

                    if conf > 0.5:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Convert to RGB for Streamlit Display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB", use_container_width=True)

            if stop_btn:
                cap.release()
                st.success("⏹️ Camera stopped!")
                break

# 📌 Footer
st.markdown('<p class="footer">Developed by Muhammad Shayan Janjua</p>', unsafe_allow_html=True)
