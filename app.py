import asyncio
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Ensure an event loop is created
if not asyncio.get_event_loop().is_running():
    asyncio.set_event_loop(asyncio.new_event_loop())

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
    .stApp {
        background: url("https://images.unsplash.com/photo-1557683316-973673baf926");
        background-size: cover;
        background-position: center;
    }
    .centered {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #00D4FF;
        padding: 20px;
    }
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
st.markdown(
    "<p style='text-align:center; font-size:20px; color:#A9A9A9;'>An advanced AI-powered system for real-time object detection.</p>",
    unsafe_allow_html=True
)

# Video Processing Class
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"

                if conf > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return img

# Streamlit WebRTC Video Stream
webrtc_streamer(key="webcam", video_transformer_factory=VideoTransformer)

# 📌 Footer
st.markdown('<p class="footer">Developed by Muhammad Shayan Janjua</p>', unsafe_allow_html=True)
