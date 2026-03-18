import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=4):
    model = models.resnet18(pretrained=True)

    # Freeze early layers
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


