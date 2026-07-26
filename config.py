"""
Centralized configuration for Secure Vision.
Loads from .env file with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Roboflow
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://classify.roboflow.com")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "face-anti-spoofing-icbck/1")

# Model paths
DEEPFAKE_MODEL_PATH = PROJECT_ROOT / os.getenv("DEEPFAKE_MODEL_PATH", "WebApp/model_97_acc_100_frames_FF_data.pt")
MOBILENET_MODEL_PATH = PROJECT_ROOT / os.getenv("MOBILENET_MODEL_PATH", "lcc-train04b-weight_all/mobilenetv2-epoch_10.hdf5")

# Device
DEVICE_STR = os.getenv("DEVICE", "mps").lower()

def get_device():
    """Get the best available device."""
    import torch
    if DEVICE_STR == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif DEVICE_STR == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# Video processing
SEQUENCE_LENGTH = 20
IM_SIZE = 112
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Streamlit
STREAMLIT_THEME = "dark"

# Flask
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"