# AI Assistive Guidance System

An AI-powered assistive vision and navigation prototype designed to help visually impaired users better understand their surroundings.

## Overview

This project combines computer vision, object detection, monocular depth estimation, and audio feedback to explore real-time environmental awareness and obstacle guidance.

## Features

- Object detection using YOLO
- Left, center, and right object positioning
- Relative depth estimation using MiDaS
- Object proximity classification
- Free-space analysis and obstacle awareness
- Directional audio guidance using text-to-speech

## Technologies

- Python
- Ultralytics YOLO
- OpenCV
- PyTorch
- MiDaS
- pyttsx3

## Current Status

This is an experimental prototype under active development. Performance optimization and responsiveness improvements are ongoing.

## Limitations

MiDaS provides relative depth estimates, not directly measured distances. The system has not been validated as a safety-critical navigation aid and should not be relied upon as a substitute for a cane, guide dog, or other established mobility support.

## Setup

1. Install Python and Git.
2. Clone this repository.
3. Install the required packages using `pip install -r requirements.txt`.
4. Download the required model weights according to the project's setup instructions.
5. Connect a compatible webcam and run the main script.

## Roadmap

- Improve real-time responsiveness
- Separate depth estimation from the main detection loop
- Improve speech prioritization and interruption
- Test across varied lighting and environments
- Explore deployment on a mobile device

## Contributions

Suggestions, feedback, and contributions are welcome.
