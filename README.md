# Hand Gesture Recognition System

This project implements a **real-time Hand Gesture Recognition (HGR)** system using:

- **MediaPipe** for hand tracking  
- **OpenCV** for webcam and image processing  
- **Feedforward Neural Network (FNN)** (via TensorFlow/Keras) for gesture classification  

It can be used in applications like **gesture-based gaming**, **virtual interfaces**, or **assistive technology for accessibility**.

---

## Technologies Used

- **MediaPipe** – Hand landmark detection and tracking  
- **OpenCV** – Webcam access, image capture, and overlay visualization  
- **TensorFlow/Keras** – Building and training the gesture classification model  
- **Python** – Programming language (recommended version 3.7 to 3.10)

---

## Installation

### Prerequisites

- Python 3.7 or higher  
  > *(Versions 3.8 to 3.10 are safest for MediaPipe compatibility)*  
- A working **webcam** (make sure no other app is using it)  
- All required libraries listed in `requirements.txt`

### Setup

```bash
pip install -r requirements.txt
```

> If you run into errors related to MediaPipe or TensorFlow, double-check your Python version.

---

## How It Works

1. **Hand Tracking**  
   MediaPipe detects 21 hand landmarks in real-time from your webcam feed.

2. **Feature Extraction**  
   These landmarks are preprocessed into a numerical vector.

3. **Gesture Classification**  
   A trained FNN model predicts the gesture class based on extracted features.

4. **Output Display**  
   The recognized gesture is visualized in a live video overlay.

---

## How to Run

Make sure your webcam is connected and not used by any other app. Then:

```bash
python controller.py
```

You can also:
- Click the **"Run" (▶️)** button in your IDE, or  
- Press **Ctrl + F5** (in editors like VS Code)

---

## Customization

- **Train Your Own Model**  
  Use `train_model.py` to train a custom FNN on your own hand gesture dataset.

- **Add New Gestures**  
  Update your label encoder, expand the dataset, and retrain the model to recognize more gestures.

---
## Author

**Shin Than Thar Aung**
