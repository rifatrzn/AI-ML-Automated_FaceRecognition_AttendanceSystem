import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, font as tkFont
from ttkthemes import ThemedTk
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
root.title("Face Recognition Attendance System")
root.geometry("800x600")  # Set a fixed size for the window

# Custom font
customFont = tkFont.Font(family="Helvetica", size=12)

# Styling
style = ttk.Style()
style.theme_use('clam')  # A theme better suited for custom styling
style.configure('TButton', font=customFont, padding=6, relief="flat",
                background="#333", foreground="#fff")
style.map('TButton', foreground=[('active', '#333')],
          background=[('active', '#eee'), ('pressed', '#ccc')])

# Header
header_frame = tk.Frame(root, height=50, bg="#444")
header_frame.pack(fill=tk.X, padx=20, pady=10)
header_label = tk.Label(header_frame, text="Attendance System", font=("Helvetica", 16, "bold"), bg="#444", fg="#fff")
header_label.pack(side=tk.LEFT, padx=10)

# Video Feed Label
video_frame = tk.Frame(root, borderwidth=2, relief="sunken")
video_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
label = tk.Label(video_frame)  # This will display the video
label.pack(fill=tk.BOTH, expand=True)

# Controls Frame
controls_frame = tk.Frame(root, padx=10, pady=10, bg="#333")
controls_frame.pack(fill=tk.X)

# Listbox to display attendance
listbox = tk.Listbox(root, width=50, height=10, font=customFont)
listbox.pack(pady=20)



cap = cv2.VideoCapture(0)

# Function to close the GUI cleanly
def close_app():
    if messagebox.askokcancel("Quit", "Do you want to exit?"):
        root.destroy()

# Create an Exit button and pack it into the GUI
exit_btn = ttk.Button(root, text="Exit", command=close_app, style='TButton')
exit_btn.pack(side=tk.RIGHT, padx=10, pady=10)

global photo_image
photo_image = None
attendance = {}

# Function definitions for the GUI actions
def update_embeddings():
    """ Function to update embeddings in the database """
    try:
        with open(embeddings_file, 'w') as file:
            json.dump(students_embeddings, file, indent=4)
        messagebox.showinfo("Update Successful", "The embeddings database has been updated.")
    except Exception as e:
        messagebox.showerror("Update Failed", f"An error occurred: {e}")

    print("Updating the database...")


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
    print("Refining embeddings...")

def capture_new_student_embedding():
    """ Open a new window to capture embedding for a new student with live preview. """
    student_name = simpledialog.askstring("Student Name", "Enter the name of the new student:")
    if not student_name:
        return  # Exit if the dialog is cancelled or no name is entered.

    # Create a new window for the live video feed and capture controls.
    capture_window = tk.Toplevel(root)
    capture_window.title("Capture New Student Embedding")

    # Frame for video feed
    video_label = tk.Label(capture_window)
    video_label.pack()

    def capture_embedding():
        """ Capture and process the embedding from the live video feed. """
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
                capture_window.destroy()
            else:
                messagebox.showerror("Capture Failed", "No face detected, please try again.")
        else:
            messagebox.showerror("Capture Failed", "Failed to capture image from the webcam.")

    # Button to trigger the capture
    capture_btn = tk.Button(capture_window, text="Capture", command=capture_embedding)
    capture_btn.pack()

print("Capturing new student embedding...")

def recognize_face(embedding):
    """ Recognize face by finding the closest embedding in the database """
    global attendance  # Access the global attendance dictionary
    min_distance = float('inf')
    recognized_name = 'Unknown'
    for name, embeddings in students_embeddings.items():
        distances = np.linalg.norm(np.array(embeddings) - embedding, axis=1)
        if distances.size > 0:
            closest_distance = np.min(distances)
            if closest_distance < min_distance and closest_distance < 0.8:  # Threshold for recognition
                min_distance = closest_distance
                recognized_name = name
                # Update the attendance dictionary when a face is recognized
                attendance[name] = 'Present'
    return recognized_name

def update_attendance_list():
    """ Update the GUI to display attendance status """
    listbox.delete(0, tk.END)  # Clear the listbox
    for name, status in attendance.items():
        listbox.insert(tk.END, f"{name}: {status}")

# Function to delete a student's embeddings from the database
def delete_student_embedding():
    student_name = simpledialog.askstring("Delete Student", "Enter the name of the student to delete:")
    if student_name and student_name in students_embeddings:
        if messagebox.askokcancel("Delete", f"Are you sure you want to delete embeddings for {student_name}?"):
            del students_embeddings[student_name]
            update_embeddings()
            messagebox.showinfo("Deletion Successful", f"Embeddings for {student_name} have been deleted.")
            update_attendance_list()  # Refresh the listbox
    elif student_name:
        messagebox.showerror("Error", f"No embeddings found for {student_name}.")
    else:
        messagebox.showerror("Error", "No name entered.")



def update_video_feed():
    global photo_image
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
                    update_attendance_list()

        photo_image = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        label.config(image=photo_image)
        label.image = photo_image  # This line is likely unnecessary since we're now using a global variable
        root.after(100, update_video_feed)  # Adjusted for lower CPU usage
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred in the video feed update: {e}")

style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12, 'bold'), borderwidth='4')
style.map('TButton', foreground=[('pressed', 'red'), ('active', 'blue')], background=[('pressed', '!disabled', 'black'), ('active', 'white')])

# Adding buttons to the control frame
update_btn = ttk.Button(controls_frame, text="Update Database", command=update_embeddings)
update_btn.pack(side=tk.LEFT, padx=10, pady=10)

refine_btn = ttk.Button(controls_frame, text="Refine Embeddings", command=refine_embeddings)
refine_btn.pack(side=tk.LEFT, padx=10, pady=10)

new_student_btn = ttk.Button(controls_frame, text="New Student Embedding", command=capture_new_student_embedding)
new_student_btn.pack(side=tk.LEFT, padx=10, pady=10)

delete_btn = ttk.Button(controls_frame, text="Delete Student", command=delete_student_embedding)
delete_btn.pack(side=tk.LEFT, padx=10, pady=10)

exit_btn = ttk.Button(controls_frame, text="Exit", command=close_app, style='TButton')
exit_btn.pack(side=tk.RIGHT, padx=10)

# Footer
footer_frame = tk.Frame(root, height=30, bg="#444")
footer_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=10)
footer_label = tk.Label(footer_frame, text="© 2024 Face Recognition Technology", bg="#444", fg="#fff")
footer_label.pack()

# Start the GUI event loop
update_video_feed()
root.mainloop()


# Clean up on closing the GUI
cap.release()
cv2.destroyAllWindows()
