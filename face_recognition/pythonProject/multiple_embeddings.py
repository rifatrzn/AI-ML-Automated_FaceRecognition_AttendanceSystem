import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import json
from torchvision import transforms

# Initialize face detector and face recognizer
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Initialize the students dictionary to hold lists of embeddings
students = {}


# Function to capture and add embeddings for multiple students
def capture_embeddings():
    while True:
        name = input('Enter the name of the student (or type "quit" to finish): ')
        if name.lower() == 'quit':
            break

        if name not in students:
            students[name] = []

        cap = cv2.VideoCapture(0)
        print(f"Capturing embeddings for {name}. Press 'c' to capture, 'q' to quit capturing for this student.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                continue
            cv2.imshow('Capture Embeddings', frame)
            key = cv2.waitKey(1)
            if key == ord('c'):
                # Process the captured frame to extract and save embeddings
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                boxes, _ = mtcnn.detect(img)
                if boxes is not None:
                    for box in boxes:
                        face = img.crop(box)
                        face_tensor = transform(face).unsqueeze(0)
                        embedding = resnet(face_tensor).detach().numpy().flatten()
                        students[name].append(embedding.tolist())
                        print("Embedding captured for", name)
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Run the capture embeddings function
capture_embeddings()

# Saving the embeddings to a JSON file
with open('students_embeddings.json', 'w') as f:
    json.dump(students, f)

print("All embeddings have been saved.")
