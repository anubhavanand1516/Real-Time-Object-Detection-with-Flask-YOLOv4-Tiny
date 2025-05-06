import cv2
import numpy as np
from flask import Flask, render_template, Response
import threading

app = Flask(__name__)

# Load YOLOv4-tiny model (pre-trained weights and config)
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels (coco.names file)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Global variables to hold frames
frame_to_process = None
frame_lock = threading.Lock()

# Helper function to perform YOLOv4-tiny detection
def perform_detection_yolo(frame):
    # Resize the frame to 416x416 for better accuracy
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:  # Lowered confidence threshold to capture more detections
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                x = center_x - w // 2
                y = center_y - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Increased NMS threshold to capture more bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    return boxes, confidences, class_ids, indexes

# Function to read frames from the camera and process them
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Lock frame for thread-safe access
            with frame_lock:
                global frame_to_process
                frame_to_process = frame.copy()

        # Wait to get the processed frame to send to the client
        with frame_lock:
            if frame_to_process is not None:
                boxes, confidences, class_ids, indexes = perform_detection_yolo(frame_to_process)

                # Draw bounding boxes and labels on the frame
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])  # Get class name from class ID
                        cv2.rectangle(frame_to_process, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame_to_process, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                ret, buffer = cv2.imencode('.jpg', frame_to_process)
                frame_to_send = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')

# Route to show index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for video stream
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
