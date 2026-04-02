import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import SwinForImageClassification
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "data/processed/video_data"
MODEL_PATH = "checkpoints/stage2_best.pth"


# MODEL (same as training)
class VideoModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.backbone = SwinForImageClassification.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224",
            num_labels=4,
            ignore_mismatched_sizes=True
        )

        self.backbone.classifier = nn.Identity()

        self.temporal = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        b, f, c, h, w = x.shape
        x = x.view(-1, c, h, w)

        features = self.backbone.swin(x).last_hidden_state
        features = features.mean(dim=1)

        video_feat = features.view(b, f, -1).mean(dim=1)

        return self.temporal(video_feat)


# DATASET
class VideoDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)

            for video in os.listdir(label_path):
                self.samples.append((os.path.join(label_path, video), int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        frames = []
        for i in range(8):
            img = Image.open(os.path.join(path, f"frame_{i}.jpg")).convert("RGB")
            img = img.resize((224, 224))

            img = np.array(img) / 255.0
            img = torch.tensor(img).permute(2, 0, 1).float()

            frames.append(img)

        return torch.stack(frames), label


# EVALUATION
def evaluate():
    model = VideoModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    dataset = VideoDataset(DATA_PATH)

    # SAME SPLIT AS TRAINING
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    _, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(DEVICE)

            outputs = model(videos)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # REPORT
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds))

    # CONFUSION MATRIX
    cm = confusion_matrix(all_labels, all_preds)

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Angry","Happy","Sad","Neutral"],
                yticklabels=["Angry","Happy","Sad","Neutral"])

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("results/confusion_matrix_eval.png")
    plt.close()

    print("✅ Confusion matrix saved to results/confusion_matrix_eval.png")


if __name__ == "__main__":
    evaluate()
    