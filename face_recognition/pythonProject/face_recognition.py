# Import the required modules
import cv2
print(cv2.__version__)
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
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

# Load the names and embeddings of the registered students from a file
# This file can be created by enrolling the students beforehand
# and saving their names and embeddings in a numpy array
# Load the names and embeddings from the JSON file
with open('students.json', 'r') as f:
    students_json = json.load(f)

# Extract names and embeddings
names = list(students_json.keys())
# Ensure embeddings are in the form of a 2D array
embeddings = np.array([students_json[name] for name in names], dtype=np.float32)

# Assuming each name now maps to a list of embeddings
embeddings_dict = {name: np.array(embeddings) for name, embeddings in students_json.items()}


# Check and reshape embeddings if necessary
if len(embeddings) > 0 and embeddings.ndim == 1:
    embeddings = embeddings.reshape(-1, 512)  # Replace 512 with the actual dimension if different


# Define a function to compare a face embedding with the registered embeddings
# and return the name of the closest match
def recognize(captured_embedding):
    global embeddings_dict
    min_distance = float('inf')
    recognized_name = 'Unknown'
    all_distances = []

    # Loop through all stored embeddings and compare them
    for name, embeddings in embeddings_dict.items():
        for embedding in embeddings:
            distance = np.linalg.norm(embedding - captured_embedding)
            all_distances.append(distance)
            if distance < min_distance:
                min_distance = distance
                recognized_name = name

    # Dynamic thresholding based on the distribution of distances
    if all_distances:
        mean_distance = np.mean(all_distances)
        std_distance = np.std(all_distances)
        threshold = max(0.8, mean_distance - std_distance)  # Example dynamic threshold calculation

        if min_distance > threshold:
            recognized_name = 'Unknown'

    return recognized_name


# Create a video capture object to read from the web cam
cap = cv2.VideoCapture(0)

# Create an empty set to store the names of the present students
present = set()


# Loop until the user presses 'q' to quit
while True:
    # Read a frame from the web cam
    ret, frame = cap.read()
    # If the frame is valid, proceed with face detection and recognition
    if ret:
        # Convert the frame to RGB color space
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to PIL image format
        img = Image.fromarray(rgb)
        # Detect faces and their bounding boxes using MTCNN
        boxes, _ = mtcnn.detect(img)
        # If at least one face is detected, proceed with face recognition
        if boxes is not None:
            for box in boxes:
                # Make sure the box has four coordinates
                if len(box) == 4:
                    # Extract the face from the image using the bounding box coordinates
                    # and ensure the coordinates are integers
                    face = img.crop([int(b) for b in box])
                    # Apply the transform to the face image
                    face_tensor = transform(face)
                    # Get the face embedding using Inception Resnet V1
                    embedding = resnet(face_tensor.unsqueeze(0)).detach().numpy()
                    # Recognize the face and get the name
                    name = recognize(embedding)
                    # Add the name to the set of present students
                    present.add(name)
                    # Draw a rectangle around the face
                    cv2.rectangle(frame,
                                  (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])),
                                  (0, 255, 0), 2)
                    # Put the name of the face
                    cv2.putText(frame, name,
                                (int(box[0]), int(box[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)
        # Show the frame with the face detection and recognition results
        cv2.imshow('Face Recognition Attendance System', frame)
    # Wait for the user to press 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


# Release the video capture object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
# Print the names of the present students
print('The following students are present:')
for name in present:
    print(name)
