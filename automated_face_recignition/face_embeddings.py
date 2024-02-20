from tensorflow.keras.models import load_model
import numpy as np

model_path = 'E:/INDEPENDENT/project/models/keras/facenet_keras.h5'
model = load_model(model_path, compile=False)

def preprocess_face(face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    face = np.expand_dims(face, axis=0)
    return face

def get_embedding(face):
    embedding = model.predict(face)
    return embedding[0]

