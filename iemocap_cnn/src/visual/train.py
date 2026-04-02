import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

from dataset import IEMOCAPVisualDataset
from model import get_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "..", "..", "features", "visual_metadata.csv")

# =========================
# SETTINGS
# =========================
test_session = "Ses05"   # Change for other folds
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.0003
NUM_CLASSES = 4

device = torch.device("cpu")

# =========================
# LOAD CSV
# =========================
df = pd.read_csv(csv_path)

# =========================
# CREATE DATASETS (WITH TRANSFORMS)
# =========================
train_full_dataset = IEMOCAPVisualDataset(csv_path, train=True)
eval_full_dataset = IEMOCAPVisualDataset(csv_path, train=False)

train_indices = []
test_indices = []

for idx, row in df.iterrows():
    path = row["image_path"]
    folder = os.path.basename(os.path.dirname(path))
    session = folder[:5]

    if session == test_session:
        test_indices.append(idx)
    else:
        train_indices.append(idx)

train_dataset = Subset(train_full_dataset, train_indices)
test_dataset = Subset(eval_full_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"\nTraining Samples: {len(train_dataset)}")
print(f"Test Samples ({test_session}): {len(test_dataset)}")


# MODEL

model = get_model(NUM_CLASSES)
model.to(device)


# CLASS WEIGHTS (HANDLE IMBALANCE)

labels = [df.iloc[i]["label"] for i in train_indices]
class_counts = Counter(labels)

weights = []
for i in range(NUM_CLASSES):
    weights.append(1.0 / class_counts[i])

weights = torch.tensor(weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# TRAINING LOOP
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}%")

print("\nTraining complete.")


# TEST EVALUATION (REAL LOSO)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)

print(f"\nTest Accuracy on {test_session}: {test_acc:.2f}%")

# CLASSIFICATION REPORT

class_names = ["Angry", "Happy", "Neutral", "Sad"]

print("\n--- Detailed Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=class_names))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

plt.xlabel("Predicted")
plt.ylabel("True")

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("\nConfusion Matrix saved as confusion_matrix.png")


# SINGLE IMAGE PREDICTION FUNCTION

import torch.nn.functional as F
from PIL import Image

def predict_single_image(image_path):
    model.eval()

    transform = IEMOCAPVisualDataset(csv_path, train=False).transform
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    print("\n==============================")
    print(f"PREDICTION: {class_names[predicted.item()]}")
    print(f"Confidence: {confidence.item()*100:.2f}%")
    print("==============================")
    print("\nDetailed Probabilities:")

    for i, emotion in enumerate(class_names):
        print(f"- {emotion}: {probs[0][i].item()*100:.2f}%")

torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")
