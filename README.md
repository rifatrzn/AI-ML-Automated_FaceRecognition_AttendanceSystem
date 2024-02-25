# Enhanced Face Recognition System for Classroom Attendance

#Project Overview
This project aims to develop an advanced face recognition system designed to streamline the process of recording attendance in educational settings. Leveraging the power of facenet_pytorch for generating face embeddings and MTCNN for accurate face detection, this system introduces multiple embeddings per student to enhance recognition accuracy under various conditions. Furthermore, it utilizes JSON for efficient data storage, making it a robust solution for managing classroom attendance.

$Features
Face Detection: Utilizes MTCNN for detecting faces within the camera's field of view.
Multiple Embeddings per Student: Captures and stores multiple face embeddings for each student to handle appearance variations.
Dynamic Thresholding: Implements dynamic thresholding for improved recognition accuracy.
JSON Storage: Efficiently manages face embeddings and student data using JSON files.

#Getting Started
-Prerequisites
Python 3.6+
PyTorch
facenet_pytorch
OpenCV-Python
PIL (Pillow)

#Installation
-Clone the repository:
git clone https://github.com/<your-username>/face-recognition-attendance.git

-Navigate to the project directory:
cd face-recognition-attendance

-Install the required packages:
pip install -r requirements.txt

#Usage
-Start the face recognition system:
python multiple_recognition.py

#Follow the on-screen instructions to capture embeddings or recognize faces.
Contributing
We welcome contributions to the Enhanced Face Recognition System for Classroom Attendance! Please follow these steps to contribute:

