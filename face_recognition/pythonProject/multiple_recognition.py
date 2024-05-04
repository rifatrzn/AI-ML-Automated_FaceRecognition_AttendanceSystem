# Import necessary libraries
import cv2  # Used for image capture and processing
import numpy as np  # Used for numerical operations
from facenet_pytorch import MTCNN, InceptionResnetV1  # Used for face detection and embedding generation
from PIL import Image  # Used for image manipulation
import json  # Used for storing and loading embeddings as JSON
from torchvision import transforms  # Used for preprocessing images for the neural network
from torchvision.transforms import Resize

# Print the OpenCV version being used
print(cv2.__version__)

# Initialize the MTCNN module for face detection
mtcnn = MTCNN()

# Initialize the InceptionResnetV1 module with pretrained weights and set it to evaluation mode
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Define the transformation pipeline for image preprocessing
transform = transforms.Compose([
    Resize((160, 160)),  # Resize the image to 160x160 pixels to meet model input size requirements
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
])

# Load the stored embeddings and names from the 'students.json' file
with open('students_embeddings.json', 'r') as f:
    students_json = json.load(f)

# Create a dictionary where each key is a person's name and the value is their list of embeddings
embeddings_dict = {name: np.array(students_json[name], dtype=np.float32) for name in students_json}

# def dynamic_threshold_for_recognition(embeddings):
#     # Calculate the mean distance and standard deviation among embeddings
#     distances = np.linalg.norm(embeddings - np.mean(embeddings, axis=0), axis=1)
#     return np.mean(distances) + np.std(distances)



def recognize(captured_embedding):
    """
    Recognizes a person based on a captured embedding by comparing it against stored embeddings.

    Args:
        captured_embedding (np.array): The embedding vector of the captured face.

    Returns:
        str: The name of the recognized individual or 'Unknown'.
    """
    global embeddings_dict
    recognized_name = 'Unknown'
    min_distance = float('inf')

    # Iterate through each stored embedding to find the closest match
    for name, embeddings in embeddings_dict.items():
        if embeddings.size == 0:
            continue
        distances = np.linalg.norm(embeddings - captured_embedding, axis=1)
        person_min_distance = np.min(distances)
        if person_min_distance < min_distance:
            min_distance = person_min_distance
            recognized_name = name if person_min_distance < 0.8 else 'Unknown'

    return recognized_name

# Initialize webcam for live face recognition
cap = cv2.VideoCapture(0)

# Set to store the names of recognized individuals
present = set()

# Main loop for continuous face recognition
while True:
    ret, frame = cap.read()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        boxes, _ = mtcnn.detect(img)

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                box = [max(0, int(b)) for b in box]  # Ensure coordinates are within the frame
                face = img.crop((box[0], box[1], box[2], box[3]))
                face_tensor = transform(face)  # Apply preprocessing transformations
                embedding = resnet(face_tensor.unsqueeze(0)).detach().numpy()
                name = recognize(embedding)
                present.add(name)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow('Face Recognition Attendance System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print the names of present individuals
print('The following students are present:')
for name in present:
    print(name)
