import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from sklearn.cluster import DBSCAN

# Initialize the face detection and recognition system
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load or initialize the embeddings database
embeddings_file = 'students_embeddings.json'
students_embeddings = {}
try:
    with open(embeddings_file, 'r') as file:
        students_embeddings = json.load(file)
except FileNotFoundError:
    print(f"File {embeddings_file} not found, starting with an empty database.")

# GUI Setup
root = tk.Tk()
root.title("Enhanced Face Recognition System")
label = ttk.Label(root)
label.pack(padx=10, pady=10)
listbox = tk.Listbox(root, width=50, height=10)
listbox.pack(pady=20)
cap = cv2.VideoCapture(0)

# Function definitions for the GUI actions
def update_embeddings():
    """ Function to update embeddings in the database """
    try:
        with open(embeddings_file, 'w') as file:
            json.dump(students_embeddings, file, indent=4)
        messagebox.showinfo("Update Successful", "The embeddings database has been updated.")
    except Exception as e:
        messagebox.showerror("Update Failed", f"An error occurred: {e}")

def refine_embeddings():
    """ Refine embeddings using DBSCAN to remove outliers """
    try:
        for name, embeddings in students_embeddings.items():
            if len(embeddings) > 2:
                db = DBSCAN(eps=0.6, min_samples=2).fit(np.array(embeddings))
                mask = db.labels_ != -1
                students_embeddings[name] = np.array(embeddings)[mask].tolist()
        update_embeddings()
    except Exception as e:
        messagebox.showerror("Refinement Failed", f"An error occurred: {e}")

def capture_new_student_embedding():
    """ Capture embedding for a new student """
    student_name = simpledialog.askstring("Student Name", "Enter the name of the new student:")
    if student_name:
        try:
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                boxes, _ = mtcnn.detect(img_pil)
                if boxes is not None:
                    box = boxes[0]  # Take the first detected face
                    face = img_pil.crop(box)
                    face_tensor = transform(face).unsqueeze(0)
                    embedding = resnet(face_tensor).detach().numpy().flatten()
                    if student_name in students_embeddings:
                        students_embeddings[student_name].append(embedding.tolist())
                    else:
                        students_embeddings[student_name] = [embedding.tolist()]
                    update_embeddings()
                    messagebox.showinfo("Capture Successful", f"Captured embedding for {student_name}")
            else:
                messagebox.showerror("Capture Failed", "Failed to capture image from the webcam.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


def recognize_face(embedding):
    """ Recognize face by finding the closest embedding in the database """
    min_distance = float('inf')
    recognized_name = 'Unknown'
    for name, embeddings in students_embeddings.items():
        distances = np.linalg.norm(np.array(embeddings) - embedding, axis=1)
        if distances.size > 0:
            closest_distance = np.min(distances)
            if closest_distance < min_distance and closest_distance < 0.8:  # Threshold for recognition
                min_distance = closest_distance
                recognized_name = name
    return recognized_name


def update_video_feed():
    """ Update the video feed displayed on the GUI """
    try:
        ret, frame = cap.read()
        if not ret:
            root.after(100, update_video_feed)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        boxes, _ = mtcnn.detect(img_pil)
        if boxes is not None:
            for box in boxes:
                face = img_pil.crop(box)
                face_tensor = transform(face).unsqueeze(0)
                embedding = resnet(face_tensor).detach().numpy().flatten()
                name = recognize_face(embedding)
                x1, y1, x2, y2 = [int(b) for b in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                if name not in listbox.get(0, tk.END):
                    listbox.insert(tk.END, name)

        photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        label.config(image=photo)
        label.image = photo
        root.after(100, update_video_feed)  # Adjusted for lower CPU usage
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred in the video feed update: {e}")

# GUI buttons for database management and student capture
update_btn = ttk.Button(root, text="Update Database", command=update_embeddings)
update_btn.pack(side=tk.LEFT, padx=10, pady=10)
refine_btn = ttk.Button(root, text="Refine Embeddings", command=refine_embeddings)
refine_btn.pack(side=tk.LEFT, padx=10, pady=10)
new_student_btn = ttk.Button(root, text="New Student Embedding", command=capture_new_student_embedding)
new_student_btn.pack(side=tk.LEFT, padx=10, pady=10)

# Start the GUI event loop
update_video_feed()
root.mainloop()

# Clean up on closing the GUI
cap.release()
cv2.destroyAllWindows()
