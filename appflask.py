from flask import Flask, request, render_template, send_file, abort
from werkzeug.utils import secure_filename
import os
import uuid
import sys
import base64
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
from urllib.parse import unquote
import mimetypes

# Add PyScripts to the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PyScripts')))
from PyScripts.Embeddings_Generation import get_face_embedding
from PyScripts.Face_Matching import find_matching_media

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template("index.html", matches=[], message=None, error=False)

@app.route('/upload', methods=['POST'])
def upload_file():
    file_path = None
    try:
        # Handle webcam image (base64)
        image_base64 = request.form.get("image_base64")
        if image_base64:
            if "," not in image_base64:
                return render_template('index.html', message="Invalid base64 image format", error=True, matches=[])
            
            header, encoded = image_base64.split(",", 1)
            image_data = base64.b64decode(encoded)
            image = Image.open(BytesIO(image_data))
            image = ImageOps.exif_transpose(image).convert("RGB")  # Auto-fix orientation
            image_np = np.array(image).astype(np.uint8)

            # Save temp image (optional step for logging/debugging)
            unique_filename = f"{uuid.uuid4()}.png"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(unique_filename))
            Image.fromarray(image_np).save(file_path)
        
        # Handle file upload
        elif 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            if file and allowed_file(file.filename):
                ext = file.filename.rsplit('.', 1)[1].lower()
                unique_filename = f"{uuid.uuid4()}.{ext}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(unique_filename))
                file.save(file_path)
            else:
                return render_template('index.html', message="Invalid file format", error=True, matches=[])
        else:
            return render_template('index.html', message="No image or capture provided", error=True, matches=[])

        # Pass the image array (if base64) or file path (if file) to embedding generator
        reference_embedding = get_face_embedding(image_np if image_base64 else file_path)

        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        if reference_embedding is None or len(reference_embedding) == 0:
            return render_template('index.html', message="No face detected in the image!", error=True, matches=[])

        matches = find_matching_media(reference_embedding)

        if matches:
            return render_template('index.html', matches=matches, message="Matches found!", error=False)
        else:
            return render_template('index.html', matches=[], message="No matching faces found in the database.", error=True)

    except Exception as e:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        return render_template('index.html', matches=[], message=f"Error during processing: {str(e)}", error=True)

@app.route('/media/<path:subpath>')
def serve_media(subpath):
    # Decode and normalize path
    safe_path = unquote(subpath).replace("\\", "/")
    
    # Build absolute path
    file_path = os.path.abspath(os.path.join('Photos and Videos', safe_path))

    # Debug print to confirm path resolution
    print(f"[MEDIA SERVE] Attempting to serve: {file_path}")

    if not os.path.isfile(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return f"File not found: {file_path}", 404

    mimetype, _ = mimetypes.guess_type(file_path)
    return send_file(file_path, mimetype=mimetype or 'application/octet-stream')


if __name__ == '__main__':
    app.run(debug=True)
