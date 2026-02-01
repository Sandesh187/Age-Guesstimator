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
