# 🛡️ Secure Vision

**A real-time face liveness detection and deepfake analysis system built with Streamlit, PyTorch, and computer vision.**

Secure Vision helps you verify whether a face in a photo, live webcam feed, or video is a **real live person** or a **spoof/deepfake**. It combines multiple detection approaches for robust security.

---

## ✨ What It Does

| Feature | Description |
|---------|-------------|
| **Photo Verification** | Upload a selfie or take a webcam photo → checks if it's a live person using cloud vision AI (OpenRouter) + local physiological analysis |
| **Live Video Analysis** | Real-time webcam monitoring → detects pulse (rPPG), micro-movements, and 3D depth consistency over time |
| **Deepfake Detection** | Upload a video → ResNeXt50 + LSTM model analyzes frames for manipulation artifacts with attention heatmaps |

---

## 🎯 Three Detection Layers

### 1. Cloud Vision API (Primary)
- Uses **OpenRouter** (free tier available) with vision models like Gemma-4-31B
- Analyzes facial texture, lighting, reflections for liveness
- Falls back gracefully if API key not configured

### 2. PhysioFusion (Local, Real-time)
A multi-physiological engine that fuses three signals from webcam frames:
- **rPPG (Remote Photoplethysmography)** — detects blood pulse from subtle skin color changes
- **Micro-motion** — tracks involuntary head tremors via optical flow on facial landmarks
- **Depth Consistency** — validates 3D geometric structure of the face

> Runs entirely locally on CPU/MPS/CUDA. No cloud dependency.

### 3. Deepfake Detector (Video Forensics)
- **Architecture**: ResNeXt50 backbone + LSTM temporal modeling
- **Input**: 20-frame clips, adaptive face-cropping (MediaPipe)
- **Output**: REAL/FAKE classification + confidence score + attention heatmap showing where the model looked
- **Trained on**: DFDC (DeepFake Detection Challenge) face-cropped data

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Webcam (for live features)
- [OpenRouter API key](https://openrouter.ai/keys) (optional but recommended for photo verification)

### Installation

```bash
# Clone the repo
git clone https://github.com/Shashankpabba6/Secure-Vision.git
cd Secure-Vision

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
# Required for cloud vision liveness checks
OPENROUTER_API_KEY=your_key_here

# Optional: switch provider (openrouter | roboflow)
FACE_API_PROVIDER=openrouter

# Optional: device selection (cuda | mps | cpu)
DEVICE=mps

# Optional: custom model paths
DEEPFAKE_MODEL_PATH=trained_model.pt
```

### Run the App

```bash
# From project root
streamlit run WebApp/app.py
```

Open http://localhost:8501 in your browser.

---

## 📁 Project Structure

```
Secure-Vision/
├── WebApp/
│   ├── app.py                 # Main Streamlit app (3 tabs)
│   ├── config.py              # Centralized configuration
│   ├── deepfake_detection.py  # ResNeXt50+LSTM video classifier
│   ├── face_client.py         # OpenRouter/Roboflow API clients
│   ├── physiofusion/          # Physiological liveness engine
│   │   ├── __init__.py
│   │   └── train.py           # Training utilities
│   ├── .streamlit/config.toml # Streamlit theme/config
│   └── output/                # Generated attention heatmaps
├── Dataset/                   # Training/test datasets (DFDC, LCC-FASD)
├── requirements.txt           # Python dependencies
├── trained_model.pth          # Pre-trained deepfake model (1.8GB)
├── .env.example               # Environment template
└── README.md                  # This file
```

---

## 🔧 How Each Tab Works

### Tab 1: Photo Verification
1. **Upload** an image or **capture** from webcam
2. Cloud vision model (OpenRouter) analyzes liveness → returns LIVE/SPOOF with reasoning
3. PhysioFusion runs local static analysis → texture quality, depth naturalness scores
4. Results shown as color-coded verdict banners with confidence gauges

### Tab 2: Live Video Analysis
1. Check **"Start live capture"**
2. App captures frames from webcam continuously
3. PhysioFusion accumulates frames → computes rPPG, micro-motion, depth scores
4. Periodic intermediate verdicts → final verdict after N frames (configurable)
5. **Reset** button clears buffer for fresh session

### Tab 3: Deepfake Detection
1. **Upload** a video file (MP4, AVI, MOV, MKV)
2. Model extracts 20 frames, crops faces adaptively
3. ResNeXt50+LSTM inference → REAL/FAKE + confidence %
4. **Attention heatmap** saved to `WebApp/output/result.jpg` showing model focus regions

---

## 🧠 Model Details

### Deepfake Detector
| Component | Specification |
|-----------|---------------|
| Backbone | ResNeXt50-32x4d (ImageNet pretrained) |
| Temporal | 1-layer LSTM, 2048 hidden dim, dropout 0.4 |
| Input | 20 frames × 112×112, ImageNet normalized |
| Classes | 2 (REAL=1, FAKE=0) |
| Face Crop | MediaPipe FaceDetection, 35% margin, full-frame fallback |
| Weights | `trained_model.pth` (or `WebApp/trained_model.pt`) |

### PhysioFusion Pipeline
- **rPPG**: CHROM method on forehead/cheek ROIs from MediaPipe landmarks
- **Micro-motion**: Lucas-Kanade optical flow on 468 facial landmarks
- **Depth**: Plane-fitting on landmark 3D positions (from MediaPipe depth)
- **Fusion**: Weighted scoring → `is_live` boolean + confidence

---

## 🔑 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | `""` | **Required** for cloud vision liveness |
| `FACE_API_PROVIDER` | `openrouter` | `openrouter` or `roboflow` |
| `OPENROUTER_VISION_MODEL` | `google/gemma-4-31b-it:free` | Vision model slug |
| `DEVICE` | `mps` | `cuda`, `mps`, or `cpu` |
| `DEEPFAKE_MODEL_PATH` | `trained_model.pt` | Path to .pth/.pt weights |
| `WEBCAM_INDEX` | `0` | Camera device index |
| `STREAMLIT_PORT` | `8501` | Streamlit server port |
| `FLASK_PORT` | `5000` | Flask server port (if used) |

---

## 📦 Dependencies (Key)

```
torch, torchvision          # Deep learning
streamlit                   # Web UI
opencv-python               # Computer vision
numpy, pillow               # Image processing
inference-sdk               # Roboflow client
mediapipe                   # Face landmarks (auto-installed)
python-dotenv               # .env loading
tqdm                        # Progress bars
```

---

## 🎥 Demo Assets

The repo includes sample media for testing:
- `output.mp4` — sample video
- `video_with_overlay.mp4` — demo with annotations
- `annotated_image.jpg` — example heatmap
- `2.png` — test image

---

## 🛠️ Development Notes

### Adding a New Model
1. Place weights in `WebApp/` or project root
2. Update `DEEPFAKE_MODEL_PATH` in `.env` or `config.py`
3. Restart Streamlit

### Switching API Providers
```bash
# In .env
FACE_API_PROVIDER=roboflow
ROBOFLOW_API_KEY=your_key
ROBOFLOW_MODEL_ID=your_model_id
```

### Training PhysioFusion
```bash
cd WebApp/physiofusion
python train.py  # See train.py for args
```

---

## ⚠️ Known Limitations

- **Deepfake model** trained on face-cropped data; full-frame videos use adaptive cropping (may miss context)
- **PhysioFusion** needs good lighting and stable camera for reliable rPPG
- **Cloud vision** requires internet + valid OpenRouter key (free tier: ~50 req/day)
- **Model size**: `trained_model.pth` is ~1.8GB (git-lfs recommended)

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## Acknowledgments

- **OpenRouter** for free vision model access
- **MediaPipe** for face detection/landmarks
- **DFDC** dataset creators
- **PyTorch/TorchVision** teams
- **Streamlit** for the delightful UI framework

---
