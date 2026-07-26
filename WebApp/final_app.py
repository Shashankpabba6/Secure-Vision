"""
Combined Streamlit app: Face Verification + Deepfake Detection.
Uses unified face_client for multi-provider support (OpenRouter, Roboflow).
"""
import streamlit as st
import cv2
import numpy as np
import tempfile
import os

from deepfake_detection import predict_deepfake
from config import WEBCAM_INDEX, FACE_API_PROVIDER
from face_client import get_face_client


def perform_face_verification(image):
    """Perform face anti-spoofing inference using configured provider."""
    client = get_face_client()
    result = client.verify_face(image)
    
    label = result['label']
    confidence = result['confidence']
    
    st.markdown(f"### Face Verification Result: {label.capitalize()} ({confidence:.1f}% confidence)")
    
    if result.get('reasoning'):
        st.caption(f"Reasoning: {result['reasoning']}")
    
    if label == 'real':
        st.success("✅ Verified!")
        return True
    elif label == 'error':
        st.error(f"❌ Error: {result['reasoning']}")
        return False
    else:
        st.error("❌ Spoof Detected!")
        return False


def perform_deepfake_detection(video_path):
    """Perform deepfake detection and display results."""
    prediction_result, confidence, image_path = predict_deepfake(video_path)
    
    st.markdown(f"### Deepfake Detection Result: {prediction_result.capitalize()} ({confidence:.2f}% confidence)")
    
    if prediction_result.lower() == 'fake':
        st.error("❌ Deepfake Detected!")
    else:
        st.success("✅ No Deepfake Detected!")
    
    st.video(video_path, format='video/mp4')


def main():
    st.set_page_config(page_title="Secure Vision", page_icon="🔒", layout="wide")
    
    st.title('🔒 Secure Vision')
    
    st.markdown(f"""
    Face verification using **{FACE_API_PROVIDER}** + Deepfake detection.
    First verify the face is live, then check videos for deepfakes.
    """)
    st.markdown("---")

    # Face verification task
    st.sidebar.markdown("## Face Verification")
    
    if st.sidebar.button("Capture Image for Face Verification 📸"):
        cap = cv2.VideoCapture(WEBCAM_INDEX)

        if not cap.isOpened():
            st.error("Error: Could not open the camera.")
            return

        ret, frame = cap.read()

        if not ret:
            st.error("Error: Failed to capture frame from the camera.")
            return

        cap.release()

        # Convert to grayscale for some providers
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if perform_face_verification(gray_frame):
            st.session_state.face_verification_done = True

    # Deepfake detection task (only shown after face verification)
    if st.session_state.get('face_verification_done', False):
        st.sidebar.markdown("## Deepfake Detection")
        
        uploaded_video = st.sidebar.file_uploader(
            "Upload Video for Deepfake Detection", 
            type=["mp4"]
        )
        
        if uploaded_video is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_video.read())
                video_path = tmp.name

            try:
                perform_deepfake_detection(video_path)
            finally:
                if os.path.exists(video_path):
                    os.unlink(video_path)


if __name__ == "__main__":
    main()