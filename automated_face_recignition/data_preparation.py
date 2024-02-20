import os
from imageio import imread
from skimage.transform import resize
from face_detection import detect_faces
from face_embeddings import preprocess_face, get_embedding
import pickle
from tensorflow.keras.models import load_model

# Load the FaceNet model
model_path = 'E:/INDEPENDENT/project/models/keras/facenet_keras.h5'  # Update this path
model = load_model (model_path, compile=False)

image_dir_basepath = 'E:/INDEPENDENT/project/data/images/'  # Update this path
names = ['LarryPage', 'MarkZuckerberg', 'BillGates']


def create_embeddings(names, image_dir_basepath):
    data = {}
    for name in names:
        image_dirpath = os.path.join (image_dir_basepath, name)
        image_filepaths = [os.path.join (image_dirpath, f) for f in os.listdir (image_dirpath)]

        aligned_images = []
        for filepath in image_filepaths:
            img = imread (filepath)
            # Assuming detect_faces returns coordinates of detected faces
            for face_coordinates in detect_faces (img):
                x, y, width, height = face_coordinates
                cropped = img[y:y + height, x:x + width]
                resized = resize (cropped, (160, 160), mode='reflect')
                aligned_images.append (resized)

        embeddings = [get_embedding (preprocess_face (image)) for image in aligned_images]

        for i, emb in enumerate (embeddings):
            data[f'{name}{i}'] = {'image_filepath': image_filepaths[i], 'emb': emb}
    return data


# Generate and save embeddings
data = create_embeddings (names, image_dir_basepath)
with open ('embeddings_dataset.pkl', 'wb') as f:
    pickle.dump (data, f)
