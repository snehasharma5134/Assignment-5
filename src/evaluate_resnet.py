# ==========================================
# CIFAR-10 MODEL COMPARISON EVALUATION
# Compares: 6-Layer Custom CNN vs ResNet-18 (Transfer Learning)
# ==========================================

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score
)

# Import our custom 6-Layer CNN architecture
from model import CIFAR10_CNN


# ------------------------------------------
# 1. DEVICE CONFIGURATION (GPU / CPU)
# ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

os.makedirs("outputs", exist_ok=True)


# ------------------------------------------
# 2. HELPER FUNCTION TO EVALUATE ANY MODEL
# ------------------------------------------
def evaluate_model(model, test_loader, model_name):
    """Runs inference on test_loader and computes Accuracy, Macro F1, CM, and Report."""
    model.eval()
    all_predictions = []
    all_labels = []

    # Disable gradients for fast evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)

            # Get predicted class index
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())

    # Calculate official metrics
    test_acc = accuracy_score(all_labels, all_predictions) * 100
    macro_f1 = f1_score(all_labels, all_predictions, average="macro")
    weighted_f1 = f1_score(all_labels, all_predictions, average="weighted")

    print(f"\n[{model_name}] Results:")
    print(f"Test Accuracy : {test_acc:.2f}%")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Weighted F1   : {weighted_f1:.4f}")

    return all_labels, all_predictions, test_acc, macro_f1, weighted_f1


# ------------------------------------------
# 3. EVALUATE MODEL 1: 6-LAYER CUSTOM CNN
# ------------------------------------------
print("\n" + "=" * 50)
print("Evaluating: 6-Layer Custom CNN (From Scratch)")
print("=" * 50)

# Custom CNN expects 32x32 images with CIFAR-10 normalization
cnn_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

cnn_test_dataset = datasets.CIFAR10(root="./dataset", train=False, download=False, transform=cnn_transform)
cnn_test_loader = DataLoader(cnn_test_dataset, batch_size=64, shuffle=False)
class_names = cnn_test_dataset.classes

# Load 6-Layer CNN checkpoint
cnn_model = CIFAR10_CNN().to(device)
cnn_ckpt = torch.load("checkpoints/best_model_30.pth", map_location=device, weights_only=False)
cnn_model.load_state_dict(cnn_ckpt["model_state_dict"])

_, _, cnn_acc, cnn_macro_f1, cnn_weighted_f1 = evaluate_model(
    cnn_model, cnn_test_loader, "6-Layer Custom CNN"
)


# ------------------------------------------
# 4. EVALUATE MODEL 2: RESNET-18 (TRANSFER LEARNING)
# ------------------------------------------
print("\n" + "=" * 50)
print("Evaluating: ResNet-18 (Transfer Learning)")
print("=" * 50)

# ResNet-18 expects 128x128 images with ImageNet normalization
resnet_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

resnet_test_dataset = datasets.CIFAR10(root="./dataset", train=False, download=False, transform=resnet_transform)
resnet_test_loader = DataLoader(resnet_test_dataset, batch_size=64, shuffle=False)

# Rebuild exact ResNet-18 structure used in train_resnet.py
resnet_model = models.resnet18(weights=None)
num_features = resnet_model.fc.in_features
resnet_model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 10)
)
resnet_model = resnet_model.to(device)

# Load trained ResNet-18 checkpoint
resnet_ckpt = torch.load("checkpoints/resnet18_cifar10.pth", map_location=device, weights_only=False)
resnet_model.load_state_dict(resnet_ckpt["model_state_dict"])

res_labels, res_preds, res_acc, res_macro_f1, res_weighted_f1 = evaluate_model(
    resnet_model, resnet_test_loader, "ResNet-18 Transfer Learning"
)

# Save ResNet-18 Confusion Matrix
cm = confusion_matrix(res_labels, res_preds)
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45, values_format="d")
plt.title("CIFAR-10 Confusion Matrix - ResNet-18 (Transfer Learning)")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_ResNet18.png")
plt.close()

# Save ResNet-18 Classification Report
res_report = classification_report(
    res_labels, res_preds, target_names=class_names, digits=4, zero_division=0
)
with open("outputs/classification_report_ResNet18.txt", "w") as f:
    f.write("Model: ResNet-18 (Transfer Learning)\n")
    f.write(f"Test Accuracy: {res_acc:.2f}%\n")
    f.write(f"Macro F1: {res_macro_f1:.4f}\n\n")
    f.write(res_report)


# ------------------------------------------
# 5. SAVE SIDE-BY-SIDE COMPARISON SUMMARY
# ------------------------------------------
summary_text = f"""CIFAR-10 MODEL COMPARISON SUMMARY
==================================================

1. 6-Layer Custom CNN (Trained from Scratch - 30 Epochs)
   - Validation Accuracy : {cnn_ckpt['val_accuracy']:.2f}%
   - Official Test Acc   : {cnn_acc:.2f}%
   - Macro F1-Score      : {cnn_macro_f1:.4f}
   - Weighted F1-Score   : {cnn_weighted_f1:.4f}

2. ResNet-18 (Transfer Learning - 12 Epochs)
   - Validation Accuracy : {resnet_ckpt['val_accuracy']:.2f}%
   - Official Test Acc   : {res_acc:.2f}%
   - Macro F1-Score      : {res_macro_f1:.4f}
   - Weighted F1-Score   : {res_weighted_f1:.4f}
"""

print("\n" + summary_text)

with open("outputs/model_comparison.txt", "w") as f:
    f.write(summary_text)

print("Comparison summary & ResNet-18 evaluation files saved successfully!")