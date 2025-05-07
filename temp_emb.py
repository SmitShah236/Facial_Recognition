import os
import face_recognition
import json

INPUT_FOLDER = "Photos and Videos"
OUTPUT_FILE = "Media/Embeddings.json"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

def get_face_embedding(image):
    face_locations = face_recognition.face_locations(image)
    embeddings = face_recognition.face_encodings(image, face_locations)
    return embeddings

def process_image(filepath):
    image = face_recognition.load_image_file(filepath)
    embeddings = get_face_embedding(image)
    if embeddings:
        return {
            "type": "image",
            "path": os.path.relpath(filepath, "Media"),  # store relative path from Media
            "embedding": [list(e) for e in embeddings]
        }
    return None

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    results = []
    for filename in os.listdir(INPUT_FOLDER):
        if any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
            filepath = os.path.join(INPUT_FOLDER, filename)
            print(f"Processing {filename}...")
            result = process_image(filepath)
            if result:
                results.append(result)
            else:
                print(f"⚠️ No face found in: {filename}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Done! Embeddings saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
