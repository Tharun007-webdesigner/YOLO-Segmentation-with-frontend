import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from tempfile import NamedTemporaryFile
import os

# Load the YOLOv8 segmentation model
model = YOLO(r"C:\Tharun4\Models\yolov8n-seg.pt")

# Streamlit app title and description
st.title("YOLOv8 Instance Segmentation on Video")
st.markdown(
    """
    This application uses the YOLOv8 model for instance segmentation on uploaded video files.
    Please upload a video below to see the segmented output.
    """
)

# File uploader for video input
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

# Button to run segmentation
if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open video.")
    else:
        stframe = st.empty()
        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # If the frame is not captured properly, break the loop
            if not ret:
                break

            # Perform instance segmentation using the YOLOv8 model
            results = model(frame)

            # Get the annotated frame with masks and bounding boxes
            annotated_frame = results[0].plot()

            # Convert the annotated frame to RGB for displaying in Streamlit
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Display the segmented output
            stframe.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

        cap.release()
        os.remove(video_path)  # Remove the temporary file after processing
