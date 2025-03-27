import streamlit as st
import cv2
import os
import tempfile
from datetime import datetime
from ultralytics import YOLO

# Load Pretrained YOLOv8 Model
model = YOLO("yolov8n.pt")

# Mapping class IDs to names
class_list = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck', 0: 'Person'}

st.title("Traffic Detection & Counting (YOLOv8)")
st.write("Upload a video to detect and count the vehicles. You can download the processed output.")


uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(tfile.name)

    # Get video properties for writer setup
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up the output video file path
    out_path = os.path.join(tempfile.gettempdir(), "result_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))


    unique_objects = {name: set() for name in class_list.values()}

    # Streamlit progress bar
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    st.write("Processing video, please wait...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Detect and track objects using YOLOv8
        results = model.track(frame, persist=False)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0].item())
                obj_id = int(box.id[0].item()) if box.id is not None else None
                class_name = class_list.get(cls)
                if class_name and obj_id is not None:
                    unique_objects[class_name].add(obj_id)


        annotated_frame = results[0].plot()
        out.write(annotated_frame)


        progress_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    out.release()
    progress_bar.empty()

    # Count unique detections per class
    counts = {name: len(ids) for name, ids in unique_objects.items()}

    st.subheader("Detected Vehicles")
    for key, value in counts.items():
        if value > 0:
            st.write(f"**{key}:** {value}")  # Displays each vehicle type on a new line

    # Download button for the processed video
    with open(out_path, "rb") as video_file:
        st.download_button(
            label="Download Processed Video",
            data=video_file,
            file_name=f"result_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
            mime="video/mp4"
        )


