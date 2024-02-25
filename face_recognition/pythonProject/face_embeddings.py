# Import the required modules
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import json
from torchvision import transforms


# Initialize the MTCNN face detector
mtcnn = MTCNN()

# Initialize the Inception Resnet V1 face recognizer
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Define a transform to convert the face images to tensors
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert a PIL Image or numpy.ndarray to tensor.
    # Normalize using the mean and standard deviation of the images in your dataset.
    # These values should be calculated if you're training your model. For pretrained models, use the normalization that they were trained with.
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# Create a video capture object to read from the web cam
cap = cv2.VideoCapture(0)


# Create an empty dictionary to store the names and embeddings of the students
students = {}

# Ask the user to enter the name of the student before starting the capture loop
name = input('Enter the name of the student before starting the capture: ')

# Function to capture and add embeddings
def add_embeddings(name, embeddings_list):
    cap = cv2.VideoCapture(0)
    print(f"Capturing embeddings for {name}. Press 'c' to capture, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        cv2.imshow('Capture Embeddings', frame)
        key = cv2.waitKey(1)
        if key == ord('c'):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            box, _ = mtcnn.detect(img)
            if box is not None:
                face = img.crop(box[0])
                face_tensor = transform(face).unsqueeze(0)
                embedding = resnet(face_tensor).detach().numpy()
                embeddings_list.append(embedding.flatten().tolist())
                print("Embedding captured.")
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# Loop until the user presses 'q' to quit
while True:
    # Read a frame from the web cam
    ret, frame = cap.read()
    # If the frame is valid, proceed with face enrollment
    if ret:
        # Convert the frame to RGB color space
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to PIL image format
        img = Image.fromarray(rgb)
        # Detect faces and their bounding boxes using MTCNN
        boxes, _ = mtcnn.detect(img)
        # If at least one face is detected, proceed with face encoding
        if boxes is not None:
            for box in boxes:
                # Convert floating point coordinates to integers
                box = [int(b) for b in box]
                # Extract the face from the image using the bounding box coordinates
                face = img.crop(box)
                # Apply the transform to the face image
                face_tensor = transform(face)
                # Get the face embedding using Inception Resnet V1
                embedding = resnet(face_tensor.unsqueeze(0)).detach().numpy()
                # Add the name and embedding to the dictionary
                students[name] = embedding
                # Draw a rectangle around the face using integer coordinates
                cv2.rectangle(frame,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              (0, 255, 0), 2)
                # Put the name of the student
                cv2.putText(frame,
                            name,
                            (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)
        # Show the frame with the face enrollment results
        cv2.imshow('Face Enrollment', frame)
    # Wait for the user to press 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the video capture object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()


# Save the dictionary as a numpy file

# Convert embeddings to a list and prepare a dictionary for JSON
students_json = {name: embedding.tolist() for name, embedding in students.items()}

# Save the dictionary as a JSON file
with open('students.json', 'w') as f:
    json.dump(students_json, f)
