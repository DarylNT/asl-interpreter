# ASL Interpreter Using MediaPipe and CNN

This project provides a real-time **American Sign Language (ASL) Interpreter** that uses **MediaPipe** to extract hand landmarks from webcam video and a **Convolutional Neural Network (CNN)** to classify each frame into ASL letters or gestures with confidence scores.

---

## Overview

The goal of this project is to bridge the communication gap between sign language users and others by translating ASL gestures into text in real time.

### Pipeline Summary

1. **Video Capture** – A webcam stream is captured using OpenCV.
2. **Landmark Extraction** – MediaPipe Hands processes each frame to obtain 3D hand landmark coordinates.
3. **Data Preprocessing** – Landmarks are normalized and optionally flattened into vectors.
4. **CNN Inference** – A trained convolutional neural network predicts the most likely ASL gesture per frame.
5. **Display Output** – The predicted label and confidence score are shown on-screen.

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
