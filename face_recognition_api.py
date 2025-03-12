from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_recognition
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load sample known faces (dummy data, replace with database faces)
known_faces = []
known_names = []

@app.route('/detect-face', methods=['POST'])
def detect_face():
    data = request.json
    image_data = data.get("image")
    
    if not image_data:
        return jsonify({"error": "No image provided"}), 400
    
    # Convert base64 to image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    image = np.array(image)
    
    # Convert image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_image)
    if len(face_locations) == 0:
        return jsonify({"message": "No face detected"})
    
    return jsonify({"message": "Face detected", "faces_count": len(face_locations)})

@app.route('/recognize-face', methods=['POST'])
def recognize_face():
    data = request.json
    image_data = data.get("image")
    
    if not image_data:
        return jsonify({"error": "No image provided"}), 400
    
    # Convert base64 to image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    image = np.array(image)
    
    # Convert image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect face encoding
    face_encodings = face_recognition.face_encodings(rgb_image)
    if len(face_encodings) == 0:
        return jsonify({"message": "No face detected"})
    
    face_encoding = face_encodings[0]
    
    # Compare with known faces
    matches = face_recognition.compare_faces(known_faces, face_encoding)
    name = "Unknown"
    if True in matches:
        matched_idx = matches.index(True)
        name = known_names[matched_idx]
    
    return jsonify({"message": "Face recognized", "name": name})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
