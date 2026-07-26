"""
Deepfake detection model: ResNeXt50 + LSTM.
Uses centralized config for device management.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch import nn
from torchvision import models
from config import DEVICE


class Model(nn.Module):
    """ResNeXt50 + LSTM for video deepfake detection."""
    
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        # Use weights parameter instead of deprecated pretrained=True
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT
        model = models.resnext50_32x4d(weights=weights)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
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


def load_model(model_path=None, num_classes=2):
    """Load a trained model from checkpoint."""
    if model_path is None:
        from config import DEEPFAKE_MODEL_PATH
        model_path = DEEPFAKE_MODEL_PATH
    
    model = Model(num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


if __name__ == "__main__":
    # Quick test
    import torch
    model = Model(2)
    x = torch.randn(1, 20, 3, 112, 112)
    fmap, out = model(x)
    print(f"Feature map shape: {fmap.shape}")
    print(f"Output shape: {out.shape}")
    print("Model architecture OK")