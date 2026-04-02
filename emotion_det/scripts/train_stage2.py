import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import SwinForImageClassification
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIG =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 2
LR = 1e-4

TRAIN_PATH = 'data/processed/train'
VAL_PATH = 'data/processed/val'
CHECKPOINT_PATH = "checkpoints/foundation_model.pth"

# ==========================================


# =============== MODEL =====================
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


# ========== LOAD STAGE A WEIGHTS ===========
def load_stage1_weights(model):
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}

    model.backbone.load_state_dict(state_dict, strict=False)

    print("✅ Stage A weights loaded correctly")


# ============== DATASET ====================
class VideoDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for label in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)

            if not os.path.isdir(label_path):
                continue

            for video in os.listdir(label_path):
                video_path = os.path.join(label_path, video)
                self.samples.append((video_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        frames = []
        for i in range(8):
            img_path = os.path.join(path, f"frame_{i}.jpg")
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))

            img = np.array(img) / 255.0
            img = torch.tensor(img).permute(2, 0, 1).float()

            frames.append(img)

        return torch.stack(frames), label


# ========== VALIDATION =====================
def validate(model, loader, final=False):
    model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in loader:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)

            outputs = model(videos)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if final:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    print(f"\nValidation Accuracy: {acc*100:.2f}%")

    if final:
        from sklearn.metrics import classification_report, confusion_matrix

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds))

        cm = confusion_matrix(all_labels, all_preds)

        os.makedirs("results", exist_ok=True)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Angry", "Happy", "Sad", "Neutral"],
            yticklabels=["Angry", "Happy", "Sad", "Neutral"]
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        plt.savefig("results/confusion_matrix.png")
        plt.close()

        print("✅ Confusion Matrix saved at results/confusion_matrix.png")

    return acc


# ============== TRAIN ======================
def train():
    model = VideoModel().to(DEVICE)
    load_stage1_weights(model)

    train_loader = DataLoader(
        VideoDataset(TRAIN_PATH),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0   # IMPORTANT (fix Windows crash)
    )

    val_loader = DataLoader(
        VideoDataset(VAL_PATH),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Class balancing
    class_weights = torch.tensor([1.2, 1.0, 1.5, 0.8]).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for videos, labels in loop:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"\nEpoch {epoch+1} Loss: {total_loss:.4f}")

        # Validation
        acc = validate(model, val_loader, final=(epoch == EPOCHS - 1))

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "checkpoints/stage2_best.pth")
            print("✅ Best model saved")

    print("\n🎯 Training Complete")


# ============ ENTRY ========================
if __name__ == "__main__":
    train()