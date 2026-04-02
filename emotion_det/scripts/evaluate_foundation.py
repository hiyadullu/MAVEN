import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import SwinForImageClassification
from torchvision import datasets, transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, roc_auc_score

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/raw/fer2013"
MODEL_PATH = "checkpoints/foundation_model.pth"

# MODEL
def get_model(num_classes=4):
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    return model

# DATASET
def get_test_loader(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader

# EVALUATION
def evaluate():
    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    test_loader = get_test_loader(DATA_DIR)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(pixel_values=images)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # REPORT
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=["Angry", "Happy", "Sad", "Neutral"]))

    # CONFUSION MATRIX
    cm = confusion_matrix(all_labels, all_preds)

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Angry","Happy","Sad","Neutral"],
                yticklabels=["Angry","Happy","Sad","Neutral"])

    plt.title("Confusion Matrix - Foundation Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("results/confusionmatrix.png")
    plt.close()

    print("✅ Confusion matrix saved to results/confusionmatrix.png")

    # ROC CURVE
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8,6))
    for i, emotion in enumerate(["Angry", "Happy", "Sad", "Neutral"]):
        plt.plot(fpr[i], tpr[i], label=f'{emotion} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Foundation Model')
    plt.legend()
    plt.savefig("results/roc_curve.png")
    plt.close()

    print("✅ ROC curve saved to results/roc_curve.png")

    # PRECISION-RECALL CURVE
    precision = {}
    recall = {}
    pr_auc = {}
    for i in range(4):
        precision[i], recall[i], _ = precision_recall_curve(all_labels == i, all_probs[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    plt.figure(figsize=(8,6))
    for i, emotion in enumerate(["Angry", "Happy", "Sad", "Neutral"]):
        plt.plot(recall[i], precision[i], label=f'{emotion} (AUC = {pr_auc[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Foundation Model')
    plt.legend()
    plt.savefig("results/pr_curve.png")
    plt.close()

    print("✅ Precision-Recall curve saved to results/pr_curve.png")

if __name__ == "__main__":
    evaluate()