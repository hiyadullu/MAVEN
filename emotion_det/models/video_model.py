import torch
import torch.nn as nn
from models.backbone import get_model

class VideoEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = get_model()

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Linear(768, 4)

    def forward(self, x):
        # x shape: (B, 8, 3, 224, 224)

        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)

        features = self.backbone.swin(x).last_hidden_state[:, 0, :]
        # shape: (B*T, 768)

        features = features.view(B, T, -1)  # (B, 8, 768)

        features = features.permute(0, 2, 1)  # (B, 768, 8)

        pooled = self.temporal_pool(features).squeeze(-1)  # (B, 768)

        out = self.classifier(pooled)

        return out