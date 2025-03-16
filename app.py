import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# 🎨 Streamlit Page Config
st.set_page_config(
    page_title="AI Object Detection from Video",
    page_icon="📹",
    layout="wide"
)

st.markdown("<h1 style='text-align: center; color: #00D4FF;'>🚀 AI Video Object Detection</h1>", unsafe_allow_html=True)

# File uploader for video
uploaded_video = st.file_uploader("📤 Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_video:
    # Save uploaded video to a temporary file
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())

    # Process video
    cap = cv2.VideoCapture(temp_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video path
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    st.text("🔄 Processing video... Please wait.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"

                if conf > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    # Display processed video
    st.text("✅ Processing complete! Download or play the detected video below.")
    st.video(output_video_path)

    # Provide download link
    with open(output_video_path, "rb") as file:
        st.download_button("⬇️ Download Processed Video", file, file_name="detected_video.mp4", mime="video/mp4")

    # Clean up temporary files
    os.remove(temp_video_path)
    os.remove(output_video_path)
