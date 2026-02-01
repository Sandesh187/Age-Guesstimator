# 🎯 Real-Time Age Detection System

A real-time age detection system built using **OpenCV DNN** and pretrained deep learning models.  
The system detects faces from a webcam feed and predicts age ranges in real-time.

---

## 🚀 Features

- Real-time face detection
- Age prediction using CNN models
- Optimized face cropping
- Confidence filtering
- FPS display
- Lightweight and fast

---

## 🧠 How It Works

This project uses **pretrained deep learning models** trained on large-scale face datasets.

- Face detection → SSD-based CNN
- Age prediction → CNN trained on IMDB-WIKI dataset
- OpenCV DNN module for inference

No training was required — pretrained models are used for real-time inference.

---

## 📦 Requirements

Install dependencies:

```bash
pip install opencv-python numpy



## ⚠️ Model Files

Due to GitHub file size limits, model weights are not included in this repo.

Download the required models here:

👉 [Download Models](
https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_deploy.prototxt
https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel
)

After downloading, place them inside:

models/

