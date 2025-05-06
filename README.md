
# 🧠 Real-Time Object Detection with Flask + YOLOv4-Tiny

This is a real-time object detection web app built using **Flask**, **OpenCV**, and **YOLOv4-tiny**. It captures webcam video, detects objects using a pre-trained YOLO model, and streams the annotated video frames to a browser.

---

## 🚀 Features

- Real-time object detection with bounding boxes and class labels
- Lightweight YOLOv4-tiny for fast inference
- Flask-powered live video streaming (`/video_feed`)
- Thread-safe frame processing using Python's `threading` module

---

## 🗃️ Project Structure

```
.
├── app.py                  # Main Flask application
├── yolov4-tiny.cfg         # YOLOv4-tiny config file
├── yolov4-tiny.weights     # YOLOv4-tiny pre-trained weights
├── coco.names              # COCO class labels file
├── templates/
│   └── index.html          # HTML template to display the video stream
├── static/                 # Optional for CSS/JS assets
└── requirements.txt        # Python dependencies
```

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
flask
opencv-python
numpy
```

---

## 📥 Download Required YOLO Files

You'll need the following files in your project root:

- `yolov4-tiny.weights` → https://pjreddie.com/media/files/yolov4-tiny.weights
- `yolov4-tiny.cfg` → https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny.cfg
- `coco.names` → https://github.com/pjreddie/darknet/blob/master/data/coco.names

Make sure these files are placed next to `app.py`.

---

## ▶️ How to Run

1. Ensure all required files are in place.
2. Start the Flask app:

```bash
python app.py
```

3. Open your browser and go to:

```
http://127.0.0.1:5000/
```

You’ll see your webcam feed with object detections updated in real time.

---

## 💡 Common Fixes

**1. `IndexError` in YOLO output layers?**  
Replace:

```python
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
```

With:

```python
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
```

**2. Webcam not detected?**  
Make sure it's not in use by another application. Try `cv2.VideoCapture(1)` if `0` doesn't work.

---

## 🧪 Example Output

The app detects and draws bounding boxes for COCO-trained object categories (person, car, dog, etc.) in real time from your webcam stream.

---


