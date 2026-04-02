import torch
from transformers import SwinForImageClassification

def get_model(num_classes=4):
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    return model