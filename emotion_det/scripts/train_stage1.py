import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
  
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report
from models.backbone import get_model
from models.dataset_loader import get_fer_dataloaders
 
# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "data/raw/fer2013"
BATCH_SIZE = 32
EPOCHS = 10
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# LOAD DATA
# -----------------------
train_loader, test_loader = get_fer_dataloaders(DATA_DIR, BATCH_SIZE)

# -----------------------
# MODEL
# -----------------------
model = get_model(num_classes=4)
model.to(DEVICE)

# -----------------------
# LOSS & OPTIMIZER
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# -----------------------
# TRAIN FUNCTION
# -----------------------
def train():
    model.train()
    total_loss = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(pixel_values=images)
        loss = criterion(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# -----------------------
# EVAL FUNCTION
# -----------------------

def evaluate():
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(pixel_values=images)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = (sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)) * 100

    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=["Angry", "Happy", "Sad", "Neutral"]
    ))

    return acc
# -----------------------
# TRAIN LOOP
# -----------------------
best_acc = 0
losses = []
accuracies = []

for epoch in range(EPOCHS):
    loss = train()
    acc = evaluate()

    losses.append(loss)
    accuracies.append(acc)

    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "checkpoints/foundation_model.pth")
        print("✅ Model saved!")

# SAVE METRICS
with open("results/training_metrics.txt", "w") as f:
    f.write("Epoch,Loss,Accuracy\n")
    for i, (l, a) in enumerate(zip(losses, accuracies), 1):
        f.write(f"{i},{l},{a}\n")

# PLOT GRAPHS
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(range(1, EPOCHS+1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, EPOCHS+1), accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("results/training_curves.png")
plt.close()

print(f"\nFinal Best Accuracy: {best_acc:.2f}%")
print("✅ Training metrics saved to results/training_metrics.txt")
print("✅ Training curves saved to results/training_curves.png")
