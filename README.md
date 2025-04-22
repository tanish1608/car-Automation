# Car Automation System

An advanced driver assistance system that uses computer vision and machine learning to detect lanes, analyze road curvature, and identify nearby objects from dashcam footage.

![Demo Screenshot](https://github.com/tanish0000/car-Automation/assets/121498791/38c904a3-0090-4382-a1e9-6da8d56c48e3)

## Overview

This project implements core technologies used in autonomous driving systems:

1. **Lane Detection & Analysis**: Identifies lane markings and calculates the road curvature ahead
2. **Object Detection**: Recognizes and classifies objects in the driving environment
3. **Distance Estimation**: Approximates the distance to detected objects

The system processes real-time dashcam footage to provide these insights, creating a foundation for more advanced autonomous driving capabilities.

## Features

- **Advanced Lane Detection**:
  - Robust lane identification even in varying lighting conditions
  - Curvature calculation to anticipate turns
  - Lane departure warning indication

- **Multi-Class Object Detection**:
  - Identifies vehicles (cars, trucks, buses, motorcycles)
  - Detects pedestrians, cyclists, and other road users
  - Recognizes traffic signs and signals

- **Environment Analysis**:
  - Distance estimation to other road users
  - Visual highlighting of detected objects
  - Real-time processing of dashcam video feed

## Tech Stack

- **Python**: Core programming language
- **OpenCV**: Image processing and computer vision operations
- **TensorFlow**: Machine learning model for object detection
- **EfficientDet-Lite2**: Pre-trained object detection model
- **NumPy**: Numerical computing for data manipulation

## Implementation Details

### Lane Detection Pipeline

The lane detection system follows these steps:
1. Camera calibration to correct for lens distortion
2. Color and gradient thresholding to isolate lane markings
3. Perspective transformation for bird's-eye view analysis
4. Lane line identification with sliding window approach
5. Curvature calculation and vehicle position estimation
6. Visual overlay of detection results on original image

### Object Detection System

For object recognition, the system:
1. Processes each video frame through the TensorFlow EfficientDet model
2. Identifies objects with confidence scores above threshold
3. Calculates approximate distance based on object size and position
4. Annotates the original frame with bounding boxes and information

## Getting Started

### Prerequisites
- Python 3.7+
- OpenCV
- TensorFlow 2.x
- NumPy
- Matplotlib (for visualization)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/car-Automation.git
cd car-Automation
