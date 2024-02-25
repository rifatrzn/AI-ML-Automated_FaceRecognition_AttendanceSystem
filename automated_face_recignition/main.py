import cv2
import pickle
from face_detection import detect_faces
from face_embeddings import preprocess_face, get_embedding
from face_recognition import recognize_face

# Load the known face embeddings and their corresponding names
with open ('known_face_encodings.pkl', 'rb') as f:
    known_face_encodings = pickle.load (f)
with open ('known_face_names.pkl', 'rb') as f:
    known_face_names = pickle.load (f)

cap = cv2.VideoCapture (0)  # Capture video from the first webcam

while True:
    ret, frame = cap.read ()
    if not ret:
        break

    # Detect faces in the current frame
    bounding_boxes = detect_faces (frame)

    for box in bounding_boxes:
        x, y, width, height = box
        face = frame[y:y + height, x:x + width]
        face = cv2.resize (face, (160, 160))  # Resize to match FaceNet expected input size
        preprocessed_face = preprocess_face (face)
        embedding = get_embedding (preprocessed_face)

        # Recognize the face based on the embedding
        identity = recognize_face (embedding, known_face_encodings, known_face_names)

        # Display the identity on the frame
        cv2.putText (frame, identity, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow ('Webcam Feed', frame)

    if cv2.waitKey (1) & 0xFF == ord ('q'):  # Press 'q' to quit
        break

cap.release ()
cv2.destroyAllWindows ()
