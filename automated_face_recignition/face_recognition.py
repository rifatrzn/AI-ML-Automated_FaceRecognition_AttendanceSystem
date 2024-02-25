from keras.models import load_model
import numpy as np
from numpy import expand_dims

# Correct path according to your directory structure
model_path = './INDEPENDENT/project/models/keras/facenet_keras.h5'
model = load_model(model_path)

def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

# Assuming `known_face_encodings` and `known_face_names` are available
def recognize_face(face_embedding, known_face_encodings, known_face_names):
    min_distance = float('inf')
    name = "Unknown"
    for i, known_embedding in enumerate(known_face_encodings):
        dist = np.linalg.norm(face_embedding - known_embedding)
        if dist < min_distance:
            min_distance = dist
            name = known_face_names[i]
    return name


def preprocess_face():
    return None