import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv10-N model
model = YOLO("C:\Tharun4\Models\yolov10l.pt")

# Open video file or capture from webcam (0 for default camera)
video_path = r"C:\Users\User\Downloads\video.mp4"  #Change to 0 if you want to use the webcam
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # If the frame is not captured properly, break the loop
    if not ret:
        break

    # Run YOLO object detection on the current frame
    results = model(frame)

    # Display the results on the frame (e.g., bounding boxes)
    annotated_frame = results[0].plot()

    # Show the frame with YOLO detections
    cv2.imshow("YOLOv10 Object Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
