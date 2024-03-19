# 3D-Facial-Recognition-2D-Cameras
3D Facial Recognition Powered by 2D cameras - The application is designed to develop a cutting-edge 3D facial recognition system that uses Advanced depth estimation algorithms that are implemented to transform 2D images into 3D representations of faces

Real-time Face Detection and Analysis using OpenCV, SimpleFacerec, and Mediapipe
This project is a Python script that uses computer vision libraries such as OpenCV, SimpleFacerec, and Mediapipe to detect faces in real-time video feed from a webcam. It then performs face recognition and face mesh analysis on the detected faces.

# Prerequisites:
- Python 3.x
- OpenCV
- SimpleFacerec
- Mediapipe
- TensorFlow

# Installation:
_Install the required libraries using pip:_
-  ``` pip install opencv-python```
- ``` pip install mediapipe```
-  ``` pip install tensorflow```
- ``` pip install face_recognition```
- **Note : face_recognition library requires Dlib python library for installation**

# Usage
_Run the script using the command:_
- ```python 3DFaceRecognition.py```
- The script will open a window displaying the real-time video feed from the webcam.
- The script will detect faces in the video feed and perform face recognition using the pre-trained model.
- The script will also perform face mesh analysis using Mediapipe and display the face outline, face mesh, and iris for eyes.
- Press 'q' to exit the script.

# Code Review
The script imports the required libraries and initializes the webcam. It then creates an instance of the SimpleFacerec class and loads the pre-trained face recognition model.
The script then enters a while loop that captures frames from the webcam and performs the following operations:
- Detects faces in the frame using the SimpleFacerec class.
- Displays the name of the recognized person on the frame.
- Converts the frame to RGB format and processes it using the Mediapipe face mesh model.
- Draws the face outline, face mesh, and iris for eyes on the frame using the Mediapipe drawing utilities.
- Displays the frame with the annotations.
- **The script continues to capture and process frames until the user presses 'q'.**
