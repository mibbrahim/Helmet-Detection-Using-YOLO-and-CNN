from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import imutils
from keras.models import load_model
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLO and CNN model
net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model = load_model('helmet-nonhelmet_cnn.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(url_for('index'))

    file = request.files['video']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = f"{uuid.uuid4().hex}.mp4"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    output_filename = detect_helmet(filepath)
    return render_template('index.html', output_video=output_filename)

def detect_helmet(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_filename = f"output_{os.path.basename(video_path)}"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

    # Ensure frame size is valid
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None

    frame = imutils.resize(frame, width=640)
    height, width = frame.shape[:2]
    out_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=640)
        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        confidences = []
        boxes = []
        classIds = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    classIds.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                if classIds[i] == 0:  # class 0 = bike
                    helmet_roi = frame[max(0, y):y + h // 4, max(0, x):x + w]
                    if helmet_roi.shape[0] > 0 and helmet_roi.shape[1] > 0:
                        helmet_roi = cv2.resize(helmet_roi, (50, 50))
                        helmet_roi = helmet_roi.astype('float32') / 255.0
                        helmet_roi = np.expand_dims(helmet_roi, axis=0)

                        prediction = model.predict(helmet_roi)[0]
                        predicted_class = np.argmax(prediction) if len(prediction) > 1 else int(prediction > 0.5)
                        label = 'Helmet' if predicted_class == 0 else 'No Helmet'
                        color = (0, 255, 0) if label == 'Helmet' else (0, 0, 255)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        out_writer.write(frame)

    cap.release()
    out_writer.release()
    return output_filename

if __name__ == '__main__':
    app.run(debug=True)
