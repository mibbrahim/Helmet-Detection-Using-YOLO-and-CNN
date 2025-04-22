# Helmet and ID Detection Using YOLO and CNN

A Flask-based web application for detecting whether individuals in a video are wearing helmets. It integrates **YOLOv3** for person detection and a custom **CNN model** for helmet classification. Ideal for traffic monitoring and industrial safety use cases.

---

## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Local Setup](#local-setup)
- [Deployment](#deployment)
- [Acknowledgements](#acknowledgements)
- [Author](#author)
- [License](#license)

---

## Features

- Upload video via web interface
- YOLOv3 detects persons in each frame
- CNN classifies whether a helmet is worn
- Annotated output video is generated and downloadable
- Deployed using Render for live demo

---

## Technology Stack

| Component        | Technology                   |
|------------------|-------------------------------|
| Backend          | Python 3.8+, Flask            |
| Object Detection | YOLOv3                        |
| Classification   | TensorFlow/Keras              |
| Video Processing | OpenCV, NumPy                 |
| Frontend         | HTML, CSS (Jinja Templates)   |
| Deployment       | Render (GitHub Integration)   |

---

## Project Structure

```
helmet_detection_flask/
├── static/                         # Processed videos saved here
├── templates/
│   └── index.html                  # Web interface (upload, preview)
├── yolov3-custom.cfg              # YOLOv3 configuration file
├── yolov3-custom_7000.weights     # YOLOv3 trained weights (Git-ignored)
├── helmet-nonhelmet_cnn.h5        # CNN model file (Git-ignored)
├── app.py                         # Flask application logic
├── requirements.txt               # Project dependencies
├── .render.yaml                   # Render deployment configuration
├── .gitignore                     # Ignore venv, models, videos
└── README.md                      # Project documentation
```

---

## How It Works

1. The user uploads a video through the web interface.
2. Flask extracts frames using OpenCV.
3. YOLOv3 detects persons in each frame.
4. Each detected person is cropped and passed to a CNN model.
5. The CNN predicts whether a helmet is worn.
6. Frames are annotated and combined into a processed video.
7. The final video is saved and downloadable from the interface.

---

## Local Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/mibbrahim/Helmet-Detection-Using-YOLO-and-CNN.git
cd Helmet-Detection-Using-YOLO-and-CNN
```

### Step 2: Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate          # On Windows
# OR
source venv/bin/activate       # On macOS/Linux
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the application

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Deployment

The project can be deployed to [Render](https://render.com):

1. Remove `.weights` and `.h5` files from Git and add them to `.gitignore`.
2. Push the rest of the code to a GitHub repository.
3. Create a new Web Service in Render.
4. Connect to the GitHub repo.
5. Set Build Command: `pip install -r requirements.txt`
6. Set Start Command: `python app.py`
7. Done — your app is now live.

---

## Acknowledgements

- **YOLOv3** — Real-time object detection by [Joseph Redmon](https://pjreddie.com/darknet/yolo/)
- **TensorFlow/Keras** — For training the CNN helmet classifier
- **OpenCV** — For handling video frames and annotations
- **Render** — For effortless web deployment

---

## Author

**Muhammad Ibrahim**  
GitHub: [@mibbrahim](https://github.com/mibbrahim)

---

## License

This project is licensed under the MIT License. See `LICENSE` for more info.
