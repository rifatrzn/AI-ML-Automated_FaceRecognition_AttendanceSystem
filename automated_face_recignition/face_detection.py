
from mtcnn import MTCNN
import cv2

detector = MTCNN()

def detect_faces(image):
    results = detector.detect_faces(image)
    faces = [result['box'] for result in results]  # Extract bounding boxes
    return faces
