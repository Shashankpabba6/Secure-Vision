"""
Deepfake detection module using ResNeXt50 + LSTM.
Uses centralized config for paths and device.
"""
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch import nn
from torchvision import models

# Local config
from config import (
    DEVICE, DEEPFAKE_MODEL_PATH, SEQUENCE_LENGTH, IM_SIZE, 
    MEAN, STD, OUTPUT_DIR
)

sm = nn.Softmax()
inv_normalize = transforms.Normalize(
    mean=-1 * np.divide(MEAN, STD),
    std=np.divide([1, 1, 1], STD)
)


class ValidationDataset(Dataset):
    """Dataset for video deepfake detection."""
    
    def __init__(self, video_names, sequence_length=20, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        
        for i, frame in enumerate(self.frame_extract(video_path)):
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        """Extract frames from video."""
        vid_obj = cv2.VideoCapture(path)
        success = True
        while success:
            success, image = vid_obj.read()
            if success:
                yield image


# Transforms
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


class Model(nn.Module):
    """ResNeXt50 + LSTM for video classification."""
    
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        # Use new weights API (torchvision >= 0.13)
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


def im_convert(tensor):
    """Convert tensor to image for visualization."""
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    cv2.imwrite(str(OUTPUT_DIR / '2.png'), image * 255)
    return image


def predict_deepfake(video_path):
    """
    Predict if a video is a deepfake.
    
    Args:
        video_path: Path to video file
        
    Returns:
        tuple: (prediction_str, confidence, image_path)
    """
    video_dataset = ValidationDataset([video_path], sequence_length=20, transform=train_transforms)
    model = Model(2).to(DEVICE)
    
    # Load model weights
    model.load_state_dict(torch.load(DEEPFAKE_MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    prediction, confidence, image_path = predict(model, video_dataset[0], OUTPUT_DIR)
    
    if prediction == 1:
        prediction_result = "REAL"
    else:
        prediction_result = "FAKE"
    
    return prediction_result, confidence, image_path


def predict(model, img, path):
    """Run inference on a single video."""
    with torch.no_grad():
        fmap, logits = model(img.to(DEVICE))
        logits = sm(logits)
        _, prediction = torch.max(logits, 1)
        confidence = logits[:, int(prediction.item())].item() * 100
        
        # Generate attention map
        idx = np.argmax(logits.detach().cpu().numpy())
        bz, nc, h, w = fmap.shape
        out = np.dot(
            fmap[-1].detach().cpu().numpy().reshape((nc, h * w)).T,
            model.linear1.weight.detach().cpu().numpy()[idx, :].T
        )
        predict_map = out.reshape(h, w)
        predict_map = predict_map - np.min(predict_map)
        predict_img = predict_map / np.max(predict_map)
        predict_img = np.uint8(255 * predict_img)
        out = cv2.resize(predict_img, (IM_SIZE, IM_SIZE))
        
        img = im_convert(img[:, -1, :, :, :])
        result = img * 0.8 * 255
        cv2.imwrite(str(path / 'result.jpg'), result)
        
        return int(prediction.item()), confidence, str(path / 'result.jpg')