import os
import cv2
import face_recognition
import json
from tqdm import tqdm
import numpy as np

INPUT_FOLDER = "Media/Photos and Videos"
OUTPUT_FILE = "Media/Embeddings.json"
FRAME_INTERVAL = 10  
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]


def get_face_embedding(image_input):
    """
    Accepts either a file path (str) or a numpy image (ndarray),
    returns list of 128D face embeddings.
    """
    if isinstance(image_input, str):  # file path
        image = face_recognition.load_image_file(image_input)
    elif isinstance(image_input, np.ndarray):  # already-loaded image
        image = image_input
    else:
        raise ValueError("Unsupported image input type.")

    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        return []

    return face_recognition.face_encodings(image, face_locations)


def process_image(filepath, base_folder):
    try:
        embeddings = get_face_embedding(filepath)
        if embeddings:
            return {
                "type": "image",
                "path": os.path.relpath(filepath, base_folder),
                "embedding": [list(e) for e in embeddings]
            }
    except Exception as e:
        tqdm.write(f"⚠️ Failed to process image {filepath}: {e}")
    return None


def process_video(filepath, base_folder):
    try:
        video = cv2.VideoCapture(filepath)
        frame_count = 0
        video_embeddings = []

        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            if frame_count % FRAME_INTERVAL == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                embeddings = get_face_embedding(rgb_frame)
                video_embeddings.extend([list(e) for e in embeddings])

            frame_count += 1

        video.release()

        if video_embeddings:
            return {
                "type": "video",
                "path": os.path.relpath(filepath, base_folder),
                "embedding": video_embeddings
            }
    except Exception as e:
        tqdm.write(f"⚠️ Failed to process video {filepath}: {e}")
    return None


def main():
    results = []
    media_files = [
        os.path.join(INPUT_FOLDER, f)
        for f in os.listdir(INPUT_FOLDER)
        if os.path.isfile(os.path.join(INPUT_FOLDER, f))
    ]

    for file in tqdm(media_files, desc="Processing media"):
        ext = os.path.splitext(file)[-1].lower()
        result = None

        if ext in IMAGE_EXTENSIONS:
            tqdm.write(f"📷 Image: {os.path.basename(file)}")
            result = process_image(file, INPUT_FOLDER)
        elif ext in VIDEO_EXTENSIONS:
            tqdm.write(f"🎥 Video: {os.path.basename(file)}")
            result = process_video(file, INPUT_FOLDER)
        else:
            tqdm.write(f"❌ Unsupported file type: {file}")

        if result:
            results.append(result)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Embeddings saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
