"""
Deepfake detection module using ResNeXt50 + LSTM.
Uses centralized config for paths and device.

Inference accuracy features:
- Adaptive face-crop preprocessing (engages only on full-frame scenes;
  model was trained on face-only data)
- First-20-consecutive-frame sampling (measured best on a 24-video
  labeled DFDC sample: 100% vs 91.7% for strided multi-clip ensembling)
- Process-wide model caching (weights are loaded once, not per request)
"""
import torch
from torchvision import transforms
import numpy as np
import cv2
from torch import nn
from torchvision import models

from config import (
    DEVICE, DEEPFAKE_MODEL_PATH, SEQUENCE_LENGTH, IM_SIZE,
    MEAN, STD, OUTPUT_DIR
)

FACE_MARGIN = 0.35
FULLFRAME_FACE_RATIO = 0.25

sm = nn.Softmax(dim=1)
inv_normalize = transforms.Normalize(
    mean=-1 * np.divide(MEAN, STD),
    std=np.divide([1, 1, 1], STD)
)

inference_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


class Model(nn.Module):
    """ResNeXt50 + LSTM for video classification."""

    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT
        model = models.resnext50_32x4d(weights=weights)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


_cached_model = None


def get_model():
    global _cached_model
    if _cached_model is None:
        model = Model(2).to(DEVICE)
        model.load_state_dict(torch.load(DEEPFAKE_MODEL_PATH, map_location=DEVICE))
        model.eval()
        _cached_model = model
    return _cached_model


class FaceCropper:
    """MediaPipe face detector with full-frame fallback."""

    def __init__(self):
        self.detector = None
        try:
            import mediapipe as mp
            self.detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
        except Exception:
            self.detector = None

    def crop(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.detector is None:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        results = self.detector.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return frame_bgr

        box = max(
            results.detections,
            key=lambda d: d.location_data.relative_bounding_box.width
                          * d.location_data.relative_bounding_box.height,
        ).location_data.relative_bounding_box

        # Already face-cropped input: keep the frame as-is
        if box.width * box.height >= FULLFRAME_FACE_RATIO:
            return frame_bgr

        mx, my = box.width * FACE_MARGIN, box.height * FACE_MARGIN
        x1 = int(max(0, (box.xmin - mx) * w))
        y1 = int(max(0, (box.ymin - my) * h))
        x2 = int(min(w, (box.xmin + box.width + mx) * w))
        y2 = int(min(h, (box.ymin + box.height + my) * h))
        if x2 - x1 < 20 or y2 - y1 < 20:
            return frame_bgr
        return frame_bgr[y1:y2, x1:x2]


def read_clip(video_path: str, cropper: FaceCropper):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < SEQUENCE_LENGTH:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(inference_transforms(cropper.crop(frame)))
    cap.release()

    if not frames:
        return None
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1])
    return torch.stack(frames)


def predict_deepfake(video_path):
    """
    Predict if a video is a deepfake.

    Args:
        video_path: Path to video file

    Returns:
        tuple: (prediction_str, confidence, image_path)
    """
    model = get_model()
    clip = read_clip(video_path, FaceCropper())
    if clip is None:
        raise ValueError(f"Could not decode frames from {video_path}")
    clip = clip.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fmap, logits = model(clip)
        prob = sm(logits)[0]
        prediction = int(torch.argmax(prob).item())
        confidence = prob[prediction].item() * 100

    image_path = save_attention_overlay(model, fmap, clip)
    prediction_result = "REAL" if prediction == 1 else "FAKE"
    return prediction_result, confidence, image_path


def save_attention_overlay(model, fmap, clip):
    if fmap is None or clip is None:
        return None
    weight = model.linear1.weight.detach().cpu().numpy()
    bz, nc, h, w = fmap.shape
    out = np.dot(
        fmap[-1].detach().cpu().numpy().reshape((nc, h * w)).T,
        weight[0, :].T
    )
    predict_map = out.reshape(h, w)
    predict_map = predict_map - np.min(predict_map)
    max_val = np.max(predict_map)
    if max_val > 0:
        predict_map = predict_map / max_val
    predict_img = np.uint8(255 * predict_map)
    heatmap = cv2.resize(predict_img, (IM_SIZE, IM_SIZE))

    image = clip[:, -1, :, :, :].to("cpu").clone().detach().squeeze()
    image = inv_normalize(image).numpy().transpose(1, 2, 0).clip(0, 1)
    base = np.uint8(image * 255)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base, 0.7, heatmap_color, 0.3, 0)
    out_path = OUTPUT_DIR / 'result.jpg'
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return str(out_path)
