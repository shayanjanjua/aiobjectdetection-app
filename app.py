import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Streamlit UI
st.set_page_config(page_title="Real-Time Object Detection", page_icon="📹", layout="wide")

st.title("📹 Real-Time Object Detection with YOLOv8")
st.sidebar.title("⚙️ Settings")

# Recording Variables
recording = st.sidebar.checkbox("Enable Recording", False)
output_video_path = None

# Initialize Webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

if not cap.isOpened():
    st.error("❌ Could not access the webcam.")
else:
    # Video Recording Setup
    if recording:
        st.sidebar.success("🎥 Recording in Progress...")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
        output_video_path = temp_file.name
        fps = 20
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Start Streamlit Video Capture
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to capture frame.")
            break

        # Run YOLOv8 Detection
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

        # Save Video if Recording is Enabled
        if recording:
            out.write(frame)

        # Convert to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_container_width=True)

        # Break loop if 'Stop Recording' is clicked
        if not recording:
            break

    # Release Resources
    cap.release()
    if recording:
        out.release()
        st.sidebar.success(f"✅ Video saved: {output_video_path}")
        st.sidebar.download_button(label="📥 Download Video", data=open(output_video_path, "rb"), file_name="recorded_video.avi", mime="video/x-msvideo")
