import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# 🎨 Streamlit Page Config
st.set_page_config(
    page_title="AI Object Detection",
    page_icon="📸",
    layout="wide"
)

# UI Title
st.markdown("<h1 style='text-align:center;'>🚀 AI Object Detection</h1>", unsafe_allow_html=True)

# 📸 Camera selection dropdown (Front or Back)
camera_option = st.selectbox("Select Camera", ["Front Camera", "Back Camera"])
video_source = 0 if camera_option == "Front Camera" else 1

# 📹 Start/Stop Recording Toggle
recording = st.checkbox("🎥 Record Video")
video_frames = []

# Define Video Processing Class
class YOLOTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLOv8 Object Detection
        results = self.model(img)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{self.model.names[cls]} {conf:.2f}"

                # Draw bounding boxes
                if conf > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save frames for recording
        if recording:
            video_frames.append(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start the webcam stream
webrtc_ctx = webrtc_streamer(
    key="object-detection",
    video_transformer_factory=YOLOTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

# 🎬 Save recorded video
if recording and video_frames:
    if st.button("💾 Save Recording"):
        output_filename = f"recording_{int(time.time())}.mp4"
        output_path = os.path.join("recordings", output_filename)

        # Create recordings directory if it doesn't exist
        os.makedirs("recordings", exist_ok=True)

        # Save video
        height, width, _ = video_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20, (width, height))

        for frame in video_frames:
            out.write(frame)

        out.release()
        st.success(f"🎥 Video saved: {output_path}")

# 📌 Footer
st.markdown('<p style="text-align:center; font-size:16px;">Developed by Muhammad Shayan Janjua</p>', unsafe_allow_html=True)
