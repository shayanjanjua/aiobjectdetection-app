import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Streamlit UI
st.title("📹 AI Object Detection from Video")

# File uploader
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video:
    # Save uploaded video temporarily
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())

    # Process video frame-by-frame
    cap = cv2.VideoCapture(temp_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video path
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    st.text("Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)  # Run YOLO object detection
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

    # Show processed video
    st.text("Processing complete! Watch or download the detected video.")
    st.video(output_video_path)

    # Download button
    with open(output_video_path, "rb") as file:
        st.download_button("⬇️ Download Processed Video", file, file_name="detected_video.mp4", mime="video/mp4")

    # Cleanup
    os.remove(temp_video_path)
    os.remove(output_video_path)

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align: center; font-size: 16px;">Developed by <b>Muhammad Shayan Janjua</b></p>
    """,
    unsafe_allow_html=True
)
