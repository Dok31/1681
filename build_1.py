import cv2
import time
from ultralytics import YOLO
import numpy as np

# Load the YOLO model
model_path = 'C:/Work_life/HSE_and_study/Project/model/best.onnx'
model = YOLO(model_path, task='detect')

# Open a connection to the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open video device")

# Set the frames per second (FPS)
fps = 24
cap.set(cv2.CAP_PROP_FPS, fps)

# Desired width and height for display
display_width, display_height = 640, 640

# Set time interval between frames
frame_interval = 1.0 / fps

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture image")
        break

    # Debug print to ensure frame is captured
    print(f"Captured frame with shape: {frame.shape}")

    # Resize frame for display if necessary
    resized_frame = cv2.resize(frame, (display_width, display_height))

    # Get predictions from the model
    results = model(resized_frame)

    # Draw bounding boxes and labels on the frame
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs

        for box in boxes:
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]  # Convert tensor to numpy array and then to integers
            x1, y1, x2, y2 = xyxy
            cls = int(box.cls.item())  # Class label
            confidence = box.conf.item()  # Confidence score

            # Draw bounding box
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display label and confidence
            label_text = f'{cls}: {confidence:.2f}'
            cv2.putText(resized_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Print class and confidence to console
            print(f'Class: {cls}, Confidence: {confidence:.2f}')

    # Display the frame with bounding boxes
    cv2.imshow('YOLO Object Detection', resized_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Ensure a consistent frame rate
    elapsed_time = time.time() - start_time
    time_to_wait = max(0, frame_interval - elapsed_time)
    time.sleep(time_to_wait)

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
