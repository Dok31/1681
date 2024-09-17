import cv2
from ultralytics import YOLO
import torch

import json
import os


def predict(model, frame):
    result = model(frame)
    return result


def data_loader():
    cur_dir = os.getcwd()
    path = os.path.join(cur_dir, 'params.json')
    params = json.loads(open(path, 'r').read())
    return params


def inference():
    params = data_loader()
    model = YOLO(params['model_path'], task='detect')

    # check if model extenison is .pt and CUDA is available
    if params['model_path'][-1] == 't' and torch.cuda.is_available() is True:
        model.to('cuda')

    # init cameras
    camera_array = []
    for element in range(len(params['cam_array'])):
        cap = cv2.VideoCapture(element)
        if not cap.isOpened():
            raise Exception(f"Could not open video devide {element}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, params['cam_res'][0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, params['cam_res'][1])
        cap.set(cv2.CAP_PROP_FPS, params['max_fps'])
        camera_array.append((cap, element))

    # desired width and height for display
    display_width, display_height = 640, 640

    while True:
        results = []

        for i, (cap, cam_name) in enumerate(camera_array):
            if cap is None:
                results.extend([False, False])
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"Failed to capture image from {cam_name}")
                results.extend([False, False])
                continue

            # Resize frame for display if necessary
            resized_frame = cv2.resize(frame, (display_width, display_height))

            # Get predictions from the model
            predictions = predict(model, resized_frame)

            left_detected = False
            right_detected = False

            # Draw bounding boxes and labels on the frame
            for result in predictions:
                boxes = result.boxes  # Boxes object for bounding box outputs

                for box in boxes:
                    xyxy = box.xyxy.cpu().numpy().astype(int)[0]  # Convert tensor to numpy array and then to integers
                    x1, y1, x2, y2 = xyxy
                    cls = int(box.cls.item())  # Class label index
                    confidence = box.conf.item()  # Confidence score

                    # Only process if the class is 1 and confidence is above the threshold
                    if cls == params['class'] and confidence >= params['confidence_threshold']:
                        # Calculate the midpoint of the bounding box
                        midpoint_x = (x1 + x2) / 2

                        # Determine if the object is on the left or right side
                        frame_midpoint = display_width / 2
                        if midpoint_x < frame_midpoint:
                            left_detected = True
                        else:
                            right_detected = True

                        # Draw bounding box
                        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Display label and confidence
                        label_text = f'Class 1: {confidence:.2f}'
                        cv2.putText(resized_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 2)

            results.extend([left_detected, right_detected])

            # Display the frame with bounding boxes
            cv2.imshow(f'YOLO Object Detection {cam_name}', resized_frame)

        print(results)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the cameras and close all windows
    cv2.destroyAllWindows()
    raise Exception("The program has finished its execution")


if __name__ == '__main__':
    inference()
