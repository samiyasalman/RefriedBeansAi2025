# AI-Powered Universal Distress Signal Detection

Overview

This project implements an AI-powered distress signal detection system that combines hand gesture and facial emotion recognition. The system first detects the universal distress hand gesture and then switches to analyzing facial expressions to identify the emotion of "fear." If both signals are detected, the system triggers a 911 emergency call using the Twilio API.

Features

Hand Gesture Recognition: Identifies the universal distress signal, which consists of a consecutive open palm followed by a closed palm.

Facial Emotion Detection: Analyzes facial expressions to recognize fear.

Automated Emergency Call: Initiates a 911 call when both signals are confirmed using the Twilio API.

YOLOv8 Model for Object Detection: Utilizes the YOLOv8 model for accurate and real-time distress signal recognition.

Dataset

The model is trained using a dataset created with Roboflow, containing images of distress hand gestures and facial expressions.

Installation

Clone this repository:

git clone https://github.com/your-username/your-repo.git
cd your-repo

Install dependencies:

pip install -r requirements.txt

Download the trained model (link here if applicable) and place it in the appropriate directory.

Usage

Run the detection script:

python detect_distress.py

The system will activate the camera, detect the distress signal, and trigger a response when necessary.

Contributing

Contributions are welcome! Feel free to submit pull requests or open issues for discussion.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

Roboflow for dataset support.

Open-source AI and computer vision frameworks used in development.

Twilio for enabling automated emergency calls.

YOLOv8 for robust object detection capabilities.
