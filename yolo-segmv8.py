import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# Load the YOLOv8 segmentation model
model = YOLO(r"C:\Tharun4\Models\yolov8n-seg.pt")

# Streamlit app
st.title("YOLOv8 Instance Segmentation")

# File uploader for images
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Process and display results if an image is uploaded
if uploaded_file is not None:
    # Convert the uploaded file to a PIL image
    image = Image.open(uploaded_file)
    
    # Convert the image to a numpy array (YOLO model expects numpy format)
    image_np = np.array(image)

    # Perform instance segmentation using the YOLOv8 model
    results = model(image_np)

    # Get the annotated image with masks and bounding boxes
    annotated_image = results[0].plot()

    # Convert the image from RGB (Matplotlib format) to BGR for OpenCV
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # Display the original and annotated image in Streamlit
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.image(annotated_image, caption="Segmented Image", use_column_width=True)

    # Option to download the segmented image
    result_image_pil = Image.fromarray(cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB))
    st.markdown("### Download Segmented Image")
    st.download_button(label="Download Image", data=result_image_pil.tobytes(), file_name="segmented_image.png", mime="image/png")
