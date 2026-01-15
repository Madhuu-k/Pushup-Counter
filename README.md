# Real-Time Push-Up Counter using Computer Vision

This project is a real-time push-up counter built using computer vision techniques.  
It uses human pose estimation and joint-angle analysis to reliably count push-ups from a live webcam feed.

The goal of this project was not just to count reps, but to understand how computer vision systems behave when exposed to real-world human movement.

## Demo
(Video demo available in the LinkedIn post)

## Tech Stack
- Python
- OpenCV
- MediaPipe Pose
- NumPy

## How It Works
- Detects full-body pose using MediaPipe Pose
- Computes elbow joint angles in real time
- Uses a finite state machine to track push-up stages (up/down)
- Applies frame-based debouncing to avoid false rep counts
- Handles noisy detections and natural movement variability

## Key Learnings
- Correct logic can still fail without proper biomechanical calibration
- Human movement introduces noise that must be handled explicitly
- Stability in CV systems often comes from state machines, not models alone

## Setup & Run
```bash
pip install -r requirements.txt
python pushup-counter.py
