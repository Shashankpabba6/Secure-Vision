"""
PhysioFusion: Multi-physiological liveness detection
Combines rPPG (remote photoplethysmography), micro-motion, and depth consistency
from a single RGB video stream.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import butter, filtfilt
import time
from pathlib import Path


@dataclass
class PhysioSignals:
    """Container for extracted physiological signals."""
    rppg: np.ndarray          # T - Blood volume pulse signal
    micro_motion: np.ndarray  # T - Involuntary head motion energy
    depth_consistency: np.ndarray  # T - Geometric depth coherence
    timestamps: np.ndarray    # T - Frame timestamps


@dataclass
class LivenessResult:
    """Liveness classification result with explanations."""
    is_live: bool
    confidence: float
    rppg_score: float
    motion_score: float
    depth_score: float
    attention_map: np.ndarray  # H x W - Spatial attention for spoof regions
    explanation: str


# =============================================================================
# rPPG Extraction (POS Algorithm - Plane Orthogonal to Skin)
# =============================================================================
class RPPExtractor:
    """
    Extract remote photoplethysmography signal from face video.
    Uses POS (Plane Orthogonal to Skin) algorithm for robust pulse extraction.
    
    References:
    - Wang et al. "Algorithmic principles of remote PPG" (2017)
    - POS algorithm: De Haan & Jeanne "Robust pulse rate from chrominance-based rPPG" (2013)
    """
    
    def __init__(self, fps: int = 30, window_sec: float = 8.0):
        self.fps = fps
        self.window_size = int(fps * window_sec)
        self.min_freq = 0.75  # 45 BPM
        self.max_freq = 3.0   # 180 BPM
        
        # Bandpass filter for pulse
        nyquist = fps / 2
        self.b, self.a = butter(
            4, 
            [self.min_freq / nyquist, self.max_freq / nyquist], 
            btype='bandpass'
        )
    
    def extract_skin_pixels(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract skin pixels from forehead and cheek regions using facial landmarks.
        
        Args:
            frame: BGR image (H, W, 3)
            landmarks: 68 facial landmarks (68, 2)
            
        Returns:
            Mean RGB values for skin regions (3,)
        """
        # Forehead region (landmarks 17-26: eyebrows, use upper portion)
        forehead_pts = landmarks[17:27]  # Eyebrow points
        forehead_y = forehead_pts[:, 1].min() - 20
        forehead_x_min = forehead_pts[:, 0].min()
        forehead_x_max = forehead_pts[:, 0].max()
        
        # Cheek regions (landmarks 1-5: left cheek, 11-15: right cheek)
        left_cheek = landmarks[1:6]
        right_cheek = landmarks[11:16]
        
        h, w = frame.shape[:2]
        skin_pixels = []
        
        # Forehead ROI
        fy1, fy2 = int(max(0, forehead_y - 30)), int(max(0, forehead_y))
        fx1, fx2 = int(max(0, forehead_x_min)), int(min(w, forehead_x_max))
        if fy2 > fy1 and fx2 > fx1:
            skin_pixels.append(frame[fy1:fy2, fx1:fx2].reshape(-1, 3))
        
        # Left cheek ROI
        ly1 = int(max(0, left_cheek[:, 1].min() - 10))
        ly2 = int(min(h, left_cheek[:, 1].max() + 10))
        lx1 = int(max(0, left_cheek[:, 0].min() - 10))
        lx2 = int(min(w, left_cheek[:, 0].max() + 10))
        if ly2 > ly1 and lx2 > lx1:
            skin_pixels.append(frame[ly1:ly2, lx1:lx2].reshape(-1, 3))
        
        # Right cheek ROI
        ry1 = int(max(0, right_cheek[:, 1].min() - 10))
        ry2 = int(min(h, right_cheek[:, 1].max() + 10))
        rx1 = int(max(0, right_cheek[:, 0].min() - 10))
        rx2 = int(min(w, right_cheek[:, 0].max() + 10))
        if ry2 > ry1 and rx2 > rx1:
            skin_pixels.append(frame[ry1:ry2, rx1:rx2].reshape(-1, 3))
        
        if skin_pixels:
            all_skin = np.vstack(skin_pixels)
            # Remove outliers (top/bottom 5%)
            all_skin = all_skin[all_skin[:, 0].argsort()]
            n = len(all_skin)
            all_skin = all_skin[int(0.05*n):int(0.95*n)]
            return all_skin.mean(axis=0)
        return frame.mean(axis=(0, 1))
    
    def pos_algorithm(self, rgb_signal: np.ndarray) -> np.ndarray:
        """
        POS algorithm for rPPG extraction.
        
        Args:
            rgb_signal: T x 3 mean RGB values over time
            
        Returns:
            T - Pulse signal
        """
        T = len(rgb_signal)
        if T < self.window_size:
            return np.zeros(T)
        
        pulse = np.zeros(T)
        
        for i in range(self.window_size, T + 1):
            window = rgb_signal[i - self.window_size:i]  # W x 3
            
            # Normalize
            mean_rgb = window.mean(axis=0)
            normalized = window / (mean_rgb + 1e-6)
            
            # POS projection matrix
            # Project onto plane orthogonal to skin tone
            # C = [0, 1, -1; -2, 1, 1] * normalized.T
            c1 = normalized[:, 1] - normalized[:, 2]  # G - B
            c2 = -2 * normalized[:, 0] + normalized[:, 1] + normalized[:, 2]  # -2R + G + B
            
            # Weight by std ratio
            alpha = c1.std() / (c2.std() + 1e-6)
            s = c1 + alpha * c2
            
            pulse[i - 1] = s[-1]
        
        # Bandpass filter
        pulse_filtered = filtfilt(self.b, self.a, pulse)
        return pulse_filtered
    
    def estimate_hr(self, pulse: np.ndarray) -> float:
        """Estimate heart rate from pulse signal using FFT."""
        if len(pulse) < self.fps * 2:
            return 0.0
        
        # Remove trend
        pulse = pulse - np.mean(pulse)
        
        # FFT
        freqs = np.fft.rfftfreq(len(pulse), 1/self.fps)
        fft_vals = np.abs(np.fft.rfft(pulse))
        
        # Find peak in physiological range
        mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        if not mask.any():
            return 0.0
        
        peak_idx = np.argmax(fft_vals[mask])
        hr = freqs[mask][peak_idx] * 60  # BPM
        return hr


# =============================================================================
# Micro-Motion Extraction
# =============================================================================
class MicroMotionExtractor:
    """
    Extract involuntary micro-motions (head tremor, facial muscle activity)
    using optical flow on facial landmarks.
    
    Live persons exhibit:
    - 0.1-0.5mm involuntary head sway (~0.5-2 Hz)
    - Micro-expressions, blink dynamics
    - Natural facial muscle tone variations
    
    Spoofs show:
    - Static (printed photo)
    - Periodic screen refresh artifacts (video replay)
    - Mechanical rigid motion (mask on stick)
    """
    
    def __init__(self, fps: int = 30):
        self.fps = fps
        self.prev_gray = None
        self.prev_landmarks = None
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
    
    def extract(self, frame: np.ndarray, landmarks: np.ndarray) -> float:
        """
        Compute micro-motion energy for current frame.
        
        Args:
            frame: BGR image
            landmarks: 68 facial landmarks (68, 2)
            
        Returns:
            Micro-motion energy (scalar)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None or self.prev_landmarks is None:
            self.prev_gray = gray
            self.prev_landmarks = landmarks.astype(np.float32)
            return 0.0
        
        # Reset if frame dimensions changed (e.g., different uploaded image)
        if self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            self.prev_landmarks = landmarks.astype(np.float32)
            return 0.0
        
        # Ensure landmarks have expected shape
        if landmarks.shape[0] < 10 or self.prev_landmarks.shape[0] < 10:
            self.prev_gray = gray
            self.prev_landmarks = landmarks.astype(np.float32)
            return 0.0
        
        # Optical flow on landmarks
        next_landmarks, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray,
            self.prev_landmarks[:68].reshape(-1, 1, 2).astype(np.float32),
            landmarks[:68].astype(np.float32).reshape(-1, 1, 2),
            **self.lk_params
        )
        
        if next_landmarks is not None and status is not None:
            valid = status.ravel() == 1
            if valid.sum() > 5:
                motion_vectors = next_landmarks[valid].reshape(-1, 2) - \
                               self.prev_landmarks[valid].reshape(-1, 2)
                
                # Compute motion energy (magnitude)
                motion_energy = float(np.mean(np.linalg.norm(motion_vectors, axis=1)))
                
                # High-frequency component (micro-motion)
                # Low-pass filter to remove gross head motion
                self.prev_gray = gray
                self.prev_landmarks = landmarks[:68].astype(np.float32)
                
                return motion_energy
        
        self.prev_gray = gray
        self.prev_landmarks = landmarks[:68].astype(np.float32)
        return 0.0
    
    def analyze_motion_signature(self, motion_history: np.ndarray) -> Dict[str, float]:
        """
        Analyze motion signature for liveness indicators.
        
        Args:
            motion_history: T - Micro-motion energy over time
            
        Returns:
            Dictionary of motion features
        """
        if len(motion_history) < self.fps * 2:
            return {'score': 0.0, 'hr_estimate': 0.0, 'naturalness': 0.0}
        
        motion = np.array(motion_history)
        
        # Remove gross motion trend
        detrended = motion - np.convolve(motion, np.ones(30)/30, mode='same')
        
        # Frequency analysis
        freqs = np.fft.rfftfreq(len(detrended), 1/self.fps)
        fft_mag = np.abs(np.fft.rfft(detrended))
        
        # Natural micro-motion: 0.5-2 Hz (involuntary sway)
        natural_mask = (freqs >= 0.5) & (freqs <= 2.0)
        natural_energy = fft_mag[natural_mask].sum()
        
        # Screen refresh artifacts: 30, 60, 120 Hz harmonics
        screen_mask = (freqs >= 25) & (freqs <= 150)
        screen_energy = fft_mag[screen_mask].sum()
        
        # Total energy
        total_energy = fft_mag.sum()
        
        # Naturalness score
        naturalness = natural_energy / (total_energy + 1e-6)
        screen_artifact = screen_energy / (total_energy + 1e-6)
        
        # Liveness score: high naturalness, low screen artifacts
        liveness_score = naturalness * (1 - screen_artifact)
        
        return {
            'score': float(np.clip(liveness_score, 0, 1)),
            'naturalness': float(naturalness),
            'screen_artifact': float(screen_artifact),
            'total_energy': float(total_energy)
        }


# =============================================================================
# Depth Consistency Checker
# =============================================================================
class DepthConsistencyChecker:
    """
    Check geometric depth consistency using monocular depth estimation.
    
    Live face: Consistent 3D geometry across frames
    2D Print: Flat depth map
    3D Mask: Inconsistent curvature, unnatural depth discontinuities
    Video Replay: Temporal depth flickering, screen plane detection
    """
    
    def __init__(self, use_simple: bool = True):
        self.use_simple = use_simple
        self.depth_model = None
        if not use_simple:
            self._load_model()
    
    def _load_model(self):
        """Load depth estimation model (MiDaS small)."""
        try:
            self.depth_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
            self.depth_model.eval()
            self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
        except Exception:
            self.depth_model = None
            self.use_simple = True
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from single frame.
        
        Args:
            frame: BGR image (H, W, 3)
            
        Returns:
            Depth map (H, W) - normalized 0-1
        """
        if self.use_simple or self.depth_model is None:
            # Fast fallback: gradient-based pseudo-depth
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Use Laplacian for edge-aware depth proxy
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
            magnitude = np.abs(laplacian)
            # Smooth
            magnitude = cv2.GaussianBlur(magnitude, (5, 5), 0)
            depth = 1.0 - (magnitude / (magnitude.max() + 1e-6))
            return depth.astype(np.float32)
        
        # Use MiDaS
        try:
            input_batch = self.transform(frame).unsqueeze(0)
            with torch.no_grad():
                prediction = self.depth_model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth = prediction.cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            return depth.astype(np.float32)
        except Exception:
            # Fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
            magnitude = np.abs(laplacian)
            depth = 1.0 - (magnitude / (magnitude.max() + 1e-6))
            return depth.astype(np.float32)
    
    def check_consistency(self, depth_maps: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze depth consistency across frames.
        
        Args:
            depth_maps: List of depth maps (H, W) over time
            
        Returns:
            Consistency metrics
        """
        if len(depth_maps) < 3:
            return {'score': 0.5, 'flatness': 0.0, 'temporal_flicker': 0.0, 'curvature': 0.0}
        
        depths = np.stack(depth_maps)  # T x H x W
        
        # 1. Flatness check (2D print = flat)
        mean_depth = depths.mean(axis=0)
        flatness = 1.0 - (mean_depth.std() / (mean_depth.mean() + 1e-6))
        flatness = np.clip(flatness, 0, 1)
        
        # 2. Temporal flicker (video replay = screen refresh flicker)
        temporal_diff = np.abs(np.diff(depths, axis=0)).mean()
        temporal_flicker = np.clip(temporal_diff * 10, 0, 1)
        
        # 3. Curvature naturalness (3D mask = unnatural curvature)
        # Compute surface normals and check for discontinuities
        dy, dx = np.gradient(mean_depth)
        normal_magnitude = np.sqrt(dx**2 + dy**2 + 1)
        normals = np.stack([-dx, -dy, np.ones_like(dx)], axis=-1) / normal_magnitude[..., None]
        
        # Normal consistency (smooth surface = consistent normals)
        normal_var = np.var(normals.reshape(-1, 3), axis=0).sum()
        curvature_score = 1.0 / (1.0 + normal_var * 100)
        
        # Combined score: not flat, low flicker, natural curvature
        score = (1 - flatness) * 0.4 + (1 - temporal_flicker) * 0.3 + curvature_score * 0.3
        
        return {
            'score': float(np.clip(score, 0, 1)),
            'flatness': float(flatness),
            'temporal_flicker': float(temporal_flicker),
            'curvature': float(curvature_score)
        }


# =============================================================================
# Cross-Attention Fusion Module
# =============================================================================
class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion of multi-modal physiological signals.
    """
    
    def __init__(self, signal_dim: int = 64, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.signal_dim = signal_dim
        
        # Temporal encoders for each modality
        self.rppg_encoder = nn.Sequential(
            nn.Conv1d(1, signal_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(signal_dim),
            nn.ReLU(),
            nn.Conv1d(signal_dim, signal_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(signal_dim),
            nn.ReLU(),
        )
        
        self.motion_encoder = nn.Sequential(
            nn.Conv1d(1, signal_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(signal_dim),
            nn.ReLU(),
            nn.Conv1d(signal_dim, signal_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(signal_dim),
            nn.ReLU(),
        )
        
        self.depth_encoder = nn.Sequential(
            nn.Conv1d(1, signal_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(signal_dim),
            nn.ReLU(),
            nn.Conv1d(signal_dim, signal_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(signal_dim),
            nn.ReLU(),
        )
        
        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=signal_dim * 3,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(signal_dim * 3, signal_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(signal_dim * 2, signal_dim),
            nn.ReLU(),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(signal_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # live, spoof
        )
        
        # Attention map generator (for explainability)
        self.attention_decoder = nn.Sequential(
            nn.ConvTranspose2d(signal_dim, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, rppg: torch.Tensor, motion: torch.Tensor, depth: torch.Tensor):
        """
        Args:
            rppg: (B, 1, T) - rPPG signal
            motion: (B, 1, T) - Micro-motion energy
            depth: (B, 1, T) - Depth consistency scores
            
        Returns:
            logits: (B, 2)
            attention_map: (B, 1, H, W)
            fused_features: (B, signal_dim)
        """
        B, _, T = rppg.shape
        
        # Encode each modality
        rppg_feat = self.rppg_encoder(rppg)          # B x D x T
        motion_feat = self.motion_encoder(motion)    # B x D x T
        depth_feat = self.depth_encoder(depth)       # B x D x T
        
        # Concatenate for cross-attention
        combined = torch.cat([rppg_feat, motion_feat, depth_feat], dim=1)  # B x 3D x T
        combined = combined.permute(0, 2, 1)  # B x T x 3D
        
        # Self-attention over time
        attended, attn_weights = self.cross_attention(combined, combined, combined)
        attended = attended.permute(0, 2, 1)  # B x 3D x T
        
        # Global pooling
        pooled = attended.mean(dim=-1)  # B x 3D
        
        # Fusion
        fused = self.fusion(pooled)  # B x D
        
        # Classification
        logits = self.classifier(fused)  # B x 2
        
        # Generate spatial attention map (upsample from temporal features)
        # Use last layer features as spatial proxy
        spatial_feat = fused.unsqueeze(-1).unsqueeze(-1)  # B x D x 1 x 1
        attention_map = self.attention_decoder(spatial_feat)  # B x 1 x H x W
        
        return logits, attention_map, fused, attn_weights


# =============================================================================
# Main PhysioFusion Pipeline
# =============================================================================
class PhysioFusionPipeline:
    """
    Complete PhysioFusion pipeline for real-time liveness detection.
    """
    
    def __init__(self, device: str = 'cpu', model_path: Optional[str] = None):
        self.device = torch.device(device)
        
        # Extractors
        self.rppg_extractor = RPPExtractor(fps=30, window_sec=8.0)
        self.motion_extractor = MicroMotionExtractor(fps=30)
        self.depth_checker = DepthConsistencyChecker()
        
        # Fusion model
        self.fusion_model = CrossAttentionFusion().to(self.device)
        
        if model_path is None:
            ckpt_dir = Path(__file__).parent / 'checkpoints'
            for fn in ['physiofusion_best.pt', 'physiofusion_last.pt']:
                fp = ckpt_dir / fn
                if fp.exists():
                    model_path = str(fp)
                    break
        
        if model_path:
            self.load_model(model_path)
        
        self.fusion_model.eval()
        
        # Buffers for temporal signals
        self.rppg_buffer = []
        self.motion_buffer = []
        self.depth_buffer = []
        self.landmark_buffer = []
        self.max_buffer = 300  # 10 seconds at 30 fps
        
        # Face detector (MediaPipe or OpenCV)
        self.face_detector = self._init_face_detector()
    
    def _init_face_detector(self):
        """Initialize face detector using OpenCV Haar cascade (works out of the box)."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print("Warning: Haar cascade not loaded, face detection disabled")
            return 'none'
        return 'haar'
    
    def detect_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect face and return approximate facial landmarks from bounding box."""
        if self.face_detector == 'haar':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )
            if len(faces) == 0:
                return None
            
            x, y, w, h = faces[0]
            # Generate 68 approximate landmarks from bounding box
            landmarks = np.zeros((68, 2))
            # Face contour (jawline): 0-16
            for i in range(17):
                angle = np.pi * i / 16
                landmarks[i] = [x + w//2 + int(w*0.45*np.cos(np.pi - angle)),
                               y + int(h*0.85*(1 - i/16))]
            # Left eyebrow: 17-21
            for i in range(5):
                landmarks[17+i] = [x + int(w*(0.2 + 0.15*i)), y + int(h*0.2)]
            # Right eyebrow: 22-26
            for i in range(5):
                landmarks[22+i] = [x + int(w*(0.55 + 0.15*i)), y + int(h*0.2)]
            # Nose bridge: 27-30, nose tip: 31-35
            for i in range(4):
                landmarks[27+i] = [x + w//2, y + int(h*(0.25 + 0.1*i))]
            for i in range(5):
                landmarks[31+i] = [x + w//2 + int(w*0.1*np.cos(0.5*(i-2))),
                                  y + int(h*(0.55 + 0.05*i))]
            # Left eye: 36-41
            cx, cy = x + int(w*0.3), y + int(h*0.35)
            for i in range(6):
                angle = np.pi * i / 3
                landmarks[36+i] = [cx + int(w*0.08*np.cos(angle)),
                                  cy + int(h*0.04*np.sin(angle))]
            # Right eye: 42-47
            cx, cy = x + int(w*0.7), y + int(h*0.35)
            for i in range(6):
                angle = np.pi * i / 3
                landmarks[42+i] = [cx + int(w*0.08*np.cos(angle)),
                                  cy + int(h*0.04*np.sin(angle))]
            # Mouth outer: 48-59
            cx, cy = x + w//2, y + int(h*0.65)
            for i in range(12):
                angle = np.pi * (i / 6 - 0.5)
                landmarks[48+i] = [cx + int(w*0.25*np.cos(angle)),
                                  cy + int(h*0.08*np.sin(angle))]
            # Mouth inner: 60-67
            for i in range(8):
                angle = np.pi * (i / 4 - 0.5)
                landmarks[60+i] = [cx + int(w*0.18*np.cos(angle)),
                                  cy + int(h*0.05*np.sin(angle))]
            return landmarks
        return None

    def _mediapipe_to_68(self, points_468: np.ndarray) -> np.ndarray:
        """Map MediaPipe 468 landmarks to standard 68."""
        # Key indices mapping (simplified)
        indices_68 = [
            162, 127, 234, 93, 132, 58, 172, 136, 150, 149,  # Jaw
            176, 148, 152, 377, 400, 378, 379, 365, 397, 288,  # Jaw cont.
            361, 323, 454, 356, 389, 251, 284, 332, 297, 338,  # Cheeks
            10, 109, 67, 103, 54, 21, 162,                      # Nose
            33, 246, 161, 160, 159, 158, 157, 173,              # Left eye
            33, 7, 163, 144, 145, 153, 154, 155,                # Right eye
            468, 473, 471, 470, 469,  # Left iris
            473, 478, 476, 475, 474,  # Right iris
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,  # Mouth outer
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,   # Mouth inner
        ]
        # Simplified - just return first 68 points
        return points_468[:68]
    
    def process_frame(self, frame: np.ndarray) -> Optional[LivenessResult]:
        """
        Process single frame and return liveness result.
        
        Args:
            frame: BGR image
            
        Returns:
            LivenessResult or None if no face detected
        """
        landmarks = self.detect_landmarks(frame)
        if landmarks is None:
            return None
        
        # Extract physiological signals
        # 1. rPPG
        skin_rgb = self.rppg_extractor.extract_skin_pixels(frame, landmarks)
        self.rppg_buffer.append(skin_rgb)
        if len(self.rppg_buffer) > self.max_buffer:
            self.rppg_buffer.pop(0)
        
        # 2. Micro-motion
        motion_energy = self.motion_extractor.extract(frame, landmarks)
        self.motion_buffer.append(motion_energy)
        if len(self.motion_buffer) > self.max_buffer:
            self.motion_buffer.pop(0)
        
        # 3. Depth
        depth_map = self.depth_checker.estimate_depth(frame)
        self.depth_buffer.append(depth_map)
        if len(self.depth_buffer) > self.max_buffer:
            self.depth_buffer.pop(0)
        
        # For single images (buffer < 30 frames), use static analysis
        # Temporal rPPG/motion requires 90+ frames (3 sec video)
        if len(self.rppg_buffer) < 30:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Texture analysis: real faces have natural texture variance
            laplacian_var = cv2.Laplacian(frame_gray, cv2.CV_64F).var()
            # Typical live face Laplacian variance: 50-500
            # Spoof (blurry/printed/flat): < 20 or > 1000
            texture_score = np.clip((laplacian_var - 20) / 200, 0, 1)
            
            # Depth analysis: check flatness of estimated depth
            depth_flatness = self.depth_checker.check_consistency([depth_map])
            depth_score = depth_flatness['score']
            
            # Combined static score
            is_live = texture_score > 0.3 and depth_score > 0.3
            confidence = (texture_score * 0.5 + depth_score * 0.5)
            
            return LivenessResult(
                is_live=bool(is_live),
                confidence=float(confidence),
                rppg_score=float(texture_score),
                motion_score=0.0,
                depth_score=float(depth_score),
                attention_map=np.zeros(frame.shape[:2]),
                explanation=f"Static analysis (single frame) — texture={texture_score:.2f}, depth={depth_score:.2f}"
            )
        
        # For video streams (30+ frames), use temporal analysis
        # Process signals
        rppg_signal = np.array(self.rppg_buffer)  # T x 3
        pulse = self.rppg_extractor.pos_algorithm(rppg_signal)
        hr = self.rppg_extractor.estimate_hr(pulse)
        
        # rPPG quality score
        rppg_score = 0.0
        if 45 <= hr <= 130:
            rppg_score = 1.0
        if len(pulse) > 30:
            pulse_snr = np.std(pulse[-90:]) / (np.mean(np.abs(pulse[-90:])) + 1e-6)
            rppg_score = float(np.clip(pulse_snr / 5.0, 0, 1))
        
        # Motion analysis
        motion_analysis = self.motion_extractor.analyze_motion_signature(
            np.array(self.motion_buffer)
        )
        motion_score = motion_analysis['score']
        
        # Depth consistency
        depth_analysis = self.depth_checker.check_consistency(self.depth_buffer)
        depth_score = depth_analysis['score']
        
        # Fusion inference
        with torch.no_grad():
            signal_len = min(90, len(self.rppg_buffer))
            rppg_tensor = torch.from_numpy(pulse[-signal_len:]).float().unsqueeze(0).unsqueeze(0).to(self.device)
            motion_tensor = torch.from_numpy(np.array(self.motion_buffer)[-signal_len:]).float().unsqueeze(0).unsqueeze(0).to(self.device)
            depth_tensor = torch.from_numpy(np.array([d.mean() for d in self.depth_buffer[-signal_len:]])).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            logits, attention_map, _, _ = self.fusion_model(
                rppg_tensor, motion_tensor, depth_tensor
            )
            probs = F.softmax(logits, dim=1)
            is_live = probs[0, 1].item() > 0.5
            confidence = probs[0, 1].item()
        
        # Explanation
        explanation = self._generate_explanation(
            rppg_score, motion_score, depth_score, hr, motion_analysis, depth_analysis
        )
        
        return LivenessResult(
            is_live=bool(is_live),
            confidence=float(confidence),
            rppg_score=float(rppg_score),
            motion_score=float(motion_score),
            depth_score=float(depth_score),
            attention_map=attention_map[0, 0].cpu().numpy(),
            explanation=explanation
        )
    
    def _generate_explanation(self, rppg: float, motion: float, depth: float,
                             hr: float, motion_analysis: Dict, depth_analysis: Dict) -> str:
        """Generate human-readable explanation."""
        parts = []
        if rppg > 0.5:
            parts.append(f"✓ Pulse detected ({hr:.0f} BPM)")
        else:
            parts.append("✗ No physiological pulse")
        
        if motion > 0.5:
            parts.append("✓ Natural micro-motions")
        elif motion_analysis.get('screen_artifact', 0) > 0.3:
            parts.append("✗ Screen refresh artifacts detected")
        else:
            parts.append("✗ Suspicious motion pattern")
        
        if depth > 0.5:
            parts.append("✓ Natural 3D geometry")
        elif depth_analysis.get('flatness', 0) > 0.7:
            parts.append("✗ Flat 2D surface detected")
        else:
            parts.append("✗ Unnatural depth geometry")
        
        return " | ".join(parts)
    
    def load_model(self, path: str):
        """Load trained fusion model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.fusion_model.load_state_dict(checkpoint['model_state_dict'])
    
    def save_model(self, path: str):
        """Save fusion model."""
        torch.save({
            'model_state_dict': self.fusion_model.state_dict(),
        }, path)
    
    def export_onnx(self, path: str):
        """Export fusion model to ONNX for deployment."""
        dummy_rppg = torch.randn(1, 1, 90).to(self.device)
        dummy_motion = torch.randn(1, 1, 90).to(self.device)
        dummy_depth = torch.randn(1, 1, 90).to(self.device)
        
        torch.onnx.export(
            self.fusion_model,
            (dummy_rppg, dummy_motion, dummy_depth),
            path,
            input_names=['rppg', 'motion', 'depth'],
            output_names=['logits', 'attention_map', 'fused_features', 'attn_weights'],
            dynamic_axes={
                'rppg': {0: 'batch', 2: 'time'},
                'motion': {0: 'batch', 2: 'time'},
                'depth': {0: 'batch', 2: 'time'},
            },
            opset_version=14
        )


# =============================================================================
# Training Utilities
# =============================================================================
class PhysioFusionTrainer:
    """Training pipeline for PhysioFusion model."""
    
    def __init__(self, model: CrossAttentionFusion, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
    
    def train_step(self, rppg: torch.Tensor, motion: torch.Tensor, depth: torch.Tensor, labels: torch.Tensor):
        self.model.train()
        self.optimizer.zero_grad()
        
        logits, _, _, _ = self.model(rppg, motion, depth)
        loss = self.criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader) -> Dict[str, float]:
        self.model.eval()
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for rppg, motion, depth, labels in val_loader:
                rppg, motion, depth, labels = \
                    rppg.to(self.device), motion.to(self.device), depth.to(self.device), labels.to(self.device)
                
                logits, _, _, _ = self.model(rppg, motion, depth)
                probs = F.softmax(logits, dim=1)
                
                preds = probs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                all_probs.append(probs[:, 1].cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        accuracy = correct / total if total > 0 else 0
        return {'accuracy': accuracy}
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
        }, path)


# =============================================================================
# Demo / Testing
# =============================================================================
if __name__ == "__main__":
    # Quick test
    pipeline = PhysioFusionPipeline(device='cpu')
    print("PhysioFusion initialized")
    print(f"Fusion model params: {sum(p.numel() for p in pipeline.fusion_model.parameters()):,}")
    
    # Test with dummy frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = pipeline.process_frame(frame)
    if result:
        print(f"Liveness: {result.is_live}, Confidence: {result.confidence:.3f}")
        print(f"Scores - rPPG: {result.rppg_score:.3f}, Motion: {result.motion_score:.3f}, Depth: {result.depth_score:.3f}")
    else:
        print("No face detected in test frame")