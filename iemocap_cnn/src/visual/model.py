import torch
import torch.nn as nn
import torchvision.models as models

class ResNetLSTM(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetLSTM, self).__init__()

        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=True)

        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # output: (B, 2048, 1, 1)

        # Freeze early layers (optional but recommended initially)
        for name, param in self.backbone.named_parameters():
            if "layer4" not in name:
                param.requires_grad = False

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )

        # Final classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.size()

        # Merge batch and time
        x = x.view(B * T, C, H, W)

        # Extract features using CNN
        features = self.backbone(x)  # (B*T, 2048, 1, 1)
        features = features.view(B, T, 2048)  # reshape back to sequence

        # LSTM
        lstm_out, _ = self.lstm(features)

        # Take last timestep
        out = lstm_out[:, -1, :]

        # Classification
        out = self.fc(out)

        return out


def get_model(num_classes=4):
    return ResNetLSTM(num_classes=num_classes)


