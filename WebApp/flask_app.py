"""
Flask app for face verification and deepfake detection.
Uses unified face_client for multi-provider support (OpenRouter, Roboflow).
"""
from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import base64
import tempfile
import os

# Local imports
from config import FACE_API_PROVIDER
from deepfake_detection import predict_deepfake
from face_client import get_face_client

app = Flask(__name__)

# Initialize face client
face_client = get_face_client()


def perform_inference(image):
    """Perform face anti-spoofing inference on image."""
    result = face_client.verify_face(image)
    
    label = result['label']
    confidence = result['confidence']
    
    if label == 'error':
        return None, None
    
    return label.capitalize(), confidence


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file:
            nparr = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert to grayscale for compatibility
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            label, confidence = perform_inference(gray_image)

            # Convert image to base64 for display
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            return render_template(
                'image_prediction.html', 
                image_base64=image_base64, 
                prediction=label, 
                confidence=confidence
            )

    return jsonify({'error': 'No file uploaded or invalid request method'})


@app.route('/upload_video', methods=['POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file:
            # Save uploaded video to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                file.save(tmp.name)
                video_path = tmp.name

            try:
                # Perform deepfake detection
                prediction_result, confidence, image_path = predict_deepfake(video_path)
                return render_template(
                    'results.html', 
                    prediction=prediction_result, 
                    confidence=confidence, 
                    image_path=image_path
                )
            finally:
                # Clean up temp file
                if os.path.exists(video_path):
                    os.unlink(video_path)

    return jsonify({'error': 'No file uploaded or invalid request method'})


@app.route('/get_result_image')
def get_result_image():
    return send_file('result.jpg', mimetype='image/jpg')


if __name__ == "__main__":
    app.run(debug=True)