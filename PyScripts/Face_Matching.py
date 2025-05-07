import face_recognition
import numpy as np
import json
import os

EMBEDDINGS_FILE = os.path.join("Media", "Embeddings.json")

def get_face_embedding(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    return encodings[0] if encodings else None

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def find_matching_media(reference_embedding, threshold=0.52):
    with open(EMBEDDINGS_FILE, "r") as f:
        database = json.load(f)

    matches = []

    for entry in database:
        for stored_embedding in entry["embedding"]:
            distance = euclidean_distance(reference_embedding, stored_embedding)
            if distance < threshold:
                matches.append({
                    "type": entry["type"],
                    "path": os.path.basename(entry["path"])  
                })
                break  

    return matches
