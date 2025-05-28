# Real-time Object Detection with YOLOv8

## Overview

This project implements a real-time object detection pipeline using a pre-trained YOLOv8 model. The system processes live webcam video, detects and tracks objects, and displays bounding boxes with confidence scores in real time. The pipeline is optimized for speed and supports detection of over 10 object classes.

## Features

- Real-time detection and tracking from webcam
- Uses YOLOv8s (pre-trained on COCO, 80+ classes)
- Draws bounding boxes and confidence scores on detected objects
- Displays total detected objects and FPS on screen
- Saves annotated video output for demonstration

## Repository Structure

.
├── object_detection_yolov8.py # Main detection and tracking script

├── requirements.txt # List of dependencies

├── README.md # This file

├── output.avi # Example output video 

└── Real_time_Object_Detection_Report.pdf # Brief analysis and discussion


## Setup & Usage

### 1. Clone the repository

git clone <https://github.com/iamrealrushi/Real-time-Object-Detection-Pipeline->
cd <Real-time-Object-Detection-Pipeline->


### 2. Install dependencies

pip install -r requirements.txt


### 3. Run the detection script

python real_time_object_detection_pipeline.py


- Press `q` to quit the live window.
- The annotated video will be saved as `output.avi`.

## Requirements

- Python 3.7+
- ultralytics
- opencv-python

## Results

- **Model:** YOLOv8s (pre-trained on COCO)
- **Classes detected:** 80 (person, car, dog, etc.)
- **Achieved FPS:** (16 FPS)
- **Demo video:** See `output.avi` or attached video file

## Notes

- The model is set to `yolov8s.pt` for a balance of speed and accuracy. You may switch to `yolov8n.pt` for higher FPS or `yolov8m.pt` for higher accuracy.
- For best performance, ensure your webcam and system meet minimum requirements.

## Analysis

See `Real_time_Object_Detection_Report.pdf` for a brief discussion of performance, challenges, and improvements.

**References:**
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)


