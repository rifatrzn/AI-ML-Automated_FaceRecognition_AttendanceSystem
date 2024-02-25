# Import necessary libraries
import cv2  # For image capture and processing
import numpy as np  # For numerical operations and embedding manipulation
from facenet_pytorch import MTCNN, InceptionResnetV1  # For face detection and embedding generation
from PIL import Image  # For image handling
import json  # For storing embeddings in a JSON file
from torchvision import transforms  # For preprocessing images for the model

# Initialize the MTCNN face detector
mtcnn = MTCNN()

# Initialize the Inception Resnet V1 face recognizer in evaluation mode
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts images to tensor format
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizes images
])

# Attempt to load existing embeddings from 'students.json'
try:
    with open('multi_students.json', 'r') as file:
        students = json.load(file)
except FileNotFoundError:
    students = {}  # Initialize an empty dictionary if the file does not exist


def capture_embeddings(name):
    """
    Captures and appends multiple embeddings for a given student's name.

    Args:
    - name (str): The name of the student for whom embeddings are being captured.

    This function captures face embeddings by processing frames captured from the webcam.
    Users can capture multiple embeddings by pressing 'c', and finish by pressing 'q'.
    """
    # Retrieve or initialize the list of embeddings for the student
    embeddings_list = students.get(name, [])
    # Start video capture
    cap = cv2.VideoCapture(0)
    print(f"Capturing embeddings for {name}. Press 'c' to capture, 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            continue

        # Display the captured frame
        cv2.imshow('Capture Embeddings', frame)
        # Wait for user input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Process the frame for face detection and embedding generation
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
            img = Image.fromarray(rgb_frame)  # Convert numpy array frame to PIL Image
            boxes, _ = mtcnn.detect(img)  # Detect faces in the image

            if boxes is not None:
                # Crop the detected face and preprocess it
                face = img.crop(boxes[0])
                face_tensor = transform(face).unsqueeze(0)
                # Generate and store the embedding
                embedding = resnet(face_tensor).detach().numpy().flatten()
                embeddings_list.append(embedding.tolist())
                print("Embedding captured.")

        elif key == ord('q'):
            # Quit capturing for the current student
            break

    # Update the student's embeddings in the dictionary
    students[name] = embeddings_list
    # Release the capture and close any open windows
    cap.release()
    cv2.destroyAllWindows()


# Main loop for enrolling students and capturing embeddings
while True:
    # Input the name of the student
    name = input("Enter the name of the student (or 'quit' to exit): ").strip()
    if name.lower() == 'quit':
        # Exit the loop
        break
    # Capture and store embeddings for the student
    capture_embeddings(name)

# Save the collected embeddings to 'students.json'
with open('multi_students.json', 'w') as file:
    json.dump(students, file)

print("All embeddings have been saved to 'multi_students.json'.")
