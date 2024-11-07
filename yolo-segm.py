import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load the YOLOv8 segmentation model
model = YOLO(r"C:\Tharun4\Models\yolov8n-seg.pt")

# Streamlit app title
st.title("YOLOv8 Instance Segmentation - Webcam")

# Start webcam button with a unique key
start_webcam = st.button("Start Webcam", key="start_webcam")

# Placeholder to display the webcam output and segmented image
frame_placeholder = st.empty()
segmentation_placeholder = st.empty()

# Variable to control the webcam state
webcam_active = False

# If the start webcam button is clicked
if start_webcam:
    webcam_active = True  # Set the webcam state to active

# If the webcam is active, run the webcam feed
if webcam_active:
    # Open webcam (use 0 for the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
    else:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Check if frame was successfully captured
            if not ret:
                st.error("Error: Failed to capture frame from webcam.")
                break

            # Perform instance segmentation using the YOLOv8 model
            results = model(frame)

            # Get the annotated frame with masks and bounding boxes
            annotated_frame = results[0].plot()

            # Convert the frame to RGB for displaying in Streamlit (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Display the original frame and segmented output
            frame_placeholder.image(frame_rgb, caption="Webcam Input", use_column_width=True)
            segmentation_placeholder.image(annotated_frame_rgb, caption="Segmented Output", use_column_width=True)

            # Stop the webcam stream if the user presses the stop button
            stop_webcam = st.button("Stop Webcam", key="stop_webcam")
            if stop_webcam:
                webcam_active = False
                break

        # Release the video capture object after the loop ends
        cap.release()

# Close any OpenCV windows after exiting the loop
cv2.destroyAllWindows()
