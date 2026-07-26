"""
Streamlit app for face verification using OpenRouter vision models.
"""
import streamlit as st
import cv2
import tempfile
import os
import numpy as np

from config import WEBCAM_INDEX, FACE_API_PROVIDER
from face_client import get_face_client


def perform_inference(image):
    """Perform face anti-spoofing inference."""
    client = get_face_client()
    result = client.verify_face(image)
    
    label = result['label']
    confidence = result['confidence']
    
    st.write(f"### Verification Result: {label.capitalize()} ({confidence:.2f} confidence)")
    
    if result.get('reasoning'):
        st.caption(f"Reasoning: {result['reasoning']}")
    
    if label == 'real':
        st.success("✅ Verified!")
    else:
        st.error("❌ Spoof Detected!")


def main():
    st.set_page_config(page_title="Secure Vision", page_icon="🔒", layout="wide")
    
    st.title('🔒 Secure Vision')
    
    st.markdown(f"""
    Face verification using **{FACE_API_PROVIDER}** API.
    Upload an image or capture from webcam to verify face liveness.
    """)
    st.markdown("---")
    st.header("Choose Input Method")
    
    input_method = st.radio("", ("Upload Image 📤", "Capture Image 📸"))

    if input_method == "Upload Image 📤":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            perform_inference(image)
            st.image(image, channels="BGR", caption='Uploaded Image', use_column_width=True)

    else:
        if st.button("Capture Image 📸"):
            cap = cv2.VideoCapture(WEBCAM_INDEX)

            if not cap.isOpened():
                st.error("Error: Could not open the camera.")
                return

            ret, frame = cap.read()

            if not ret:
                st.error("Error: Failed to capture frame from the camera.")
                return

            cap.release()

            perform_inference(frame)
            st.image(frame, channels="BGR", caption='Captured Image', use_column_width=True)


if __name__ == "__main__":
    main()