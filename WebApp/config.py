"""
Centralized configuration for Secure Vision.
Supports both OpenRouter and Roboflow for face verification.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root
ROOT_DIR = Path(__file__).parent.parent
WEBAPP_DIR = Path(__file__).parent

# Device configuration
def get_device():
    import torch
    device_str = os.getenv("DEVICE", "mps").lower()
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "mps" and hasattr(torch, 'mps') and torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# API Provider: "openrouter" or "roboflow"
FACE_API_PROVIDER = os.getenv("FACE_API_PROVIDER", "openrouter")

# ============================================
# OPENROUTER CONFIGURATION
# ============================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_VISION_MODEL = os.getenv("OPENROUTER_VISION_MODEL", "google/gemma-4-31b-it:free")

# ============================================
# ROBOFLOW CONFIGURATION (legacy)
# ============================================
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://classify.roboflow.com")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "face-anti-spoofing-icbck/1")

# ============================================
# MODEL PATHS
# ============================================
DEEPFAKE_MODEL_PATH = WEBAPP_DIR / os.getenv("DEEPFAKE_MODEL_PATH", "trained_model.pt")
MOBILENET_MODEL_PATH = ROOT_DIR / os.getenv("MOBILENET_MODEL_PATH", "lcc-train04b-weight_all/mobilenetv2-epoch_10.hdf5")
TRAINED_MODEL_PATH = ROOT_DIR / "trained_model.pth"

# ============================================
# VIDEO PROCESSING
# ============================================
SEQUENCE_LENGTH = 20
IM_SIZE = 112
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ============================================
# DATASET PATHS
# ============================================
DATASET_DIR = ROOT_DIR / "Dataset"
DFDC_FAKE_DIR = DATASET_DIR / "DFDC_FAKE_Face_only_data"
DFDC_REAL_DIR = DATASET_DIR / "DFDC_REAL_Face_only_data"
LCC_FASD_DIR = DATASET_DIR / "LCC_FASD"
TESTING_DIR = DATASET_DIR / "Testing"

# ============================================
# WEBCAM
# ============================================
WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX", "0"))

# ============================================
# OUTPUT
# ============================================
OUTPUT_DIR = WEBAPP_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================
# FLASK/STREAMLIT
# ============================================
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))