# ==========================================
# CIFAR-10 6-LAYER CNN MODEL EVALUATION
# ==========================================

import os
import torch
import matplotlib.pyplot as plt

# Scikit-learn metrics for confusion matrix, F1-score, and detailed report
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score
)

# Import our 6-layer CNN architecture and data loader function
from model import CIFAR10_CNN
from dataset import get_data_loaders

# ------------------------------------------
# 1. DEVICE CONFIGURATION (GPU / CPU)
# ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ------------------------------------------
# 2. LOAD OFFICIAL TEST DATASET (10,000 Images)
# ------------------------------------------
# We only need test_loader here, so train and val loaders are ignored with '_'
_, _, test_loader = get_data_loaders()
class_names = test_loader.dataset.classes  # List of 10 CIFAR-10 class names

# ------------------------------------------
# 3. LOAD TRAINED 6-LAYER MODEL CHECKPOINT
# ------------------------------------------
# Initialize empty 6-layer CNN architecture on the selected device
model = CIFAR10_CNN().to(device)

# Load saved best weights from training
checkpoint = torch.load(
    "checkpoints/best_model_30.pth",
    map_location=device,
    weights_only=False
)

# Load weights into the model
model.load_state_dict(checkpoint["model_state_dict"])

# Set model to evaluation mode (disables Dropout & locks BatchNorm)
model.eval()

print(f"Loaded Checkpoint Epoch: {checkpoint['epoch']}")
print(f"Validation Accuracy: {checkpoint['val_accuracy']:.2f}%")

# ------------------------------------------
# 4. RUN PREDICTIONS ON TEST SET
# ------------------------------------------
all_predictions = []  # Stores model's predicted class index for each image
all_labels = []       # Stores actual ground-truth label for each image
correct = 0           # Count of right predictions
total = 0             # Total test images processed

# Disable gradient calculation during testing to save memory and speed up inference
with torch.no_grad():
    for images, labels in test_loader:
        # Move test batch to GPU/CPU
        images, labels = images.to(device), labels.to(device)

        # Forward pass: get raw class scores (logits)
        outputs = model(images)

        # Pick the class index with the highest score
        _, predicted = torch.max(outputs, 1)

        # Update accuracy counters
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Save batch results to lists for Confusion Matrix & Classification Report
        all_predictions.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

# ------------------------------------------
# 5. CALCULATE OVERALL TEST ACCURACY & F1-SCORE
# ------------------------------------------
test_accuracy = 100 * correct / total
macro_f1 = f1_score(all_labels, all_predictions, average="macro")

print("\n================================")
print("FINAL TEST RESULTS (6-LAYER CNN)")
print("================================")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Macro F1-Score: {macro_f1:.4f}")

# ------------------------------------------
# 6. GENERATE & SAVE CONFUSION MATRIX IMAGE
# ------------------------------------------
os.makedirs("outputs", exist_ok=True)  # Create outputs folder if missing

# Compute 10x10 confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Plot confusion matrix with class labels
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45, values_format="d")
plt.title("CIFAR-10 Confusion Matrix (6-Layer CNN - 30 Epochs)")
plt.tight_layout()

# Save image (filename matches what app.py expects)
plt.savefig("outputs/confusion_matrix_Fine_Tuned_30.png")
plt.close()

# ------------------------------------------
# 7. GENERATE & SAVE CLASSIFICATION REPORT
# ------------------------------------------
# Precision, Recall, F1-Score, and Support for each of the 10 classes
report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\nCLASSIFICATION REPORT:\n")
print(report)

# Save report to text file (filename matches what app.py expects)
with open("outputs/classification_report_Fine_Tuned_30.txt", "w") as f:
    f.write("CIFAR-10 CLASSIFICATION REPORT (6-LAYER CNN)\n\n")
    f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
    f.write(f"Macro F1-Score: {macro_f1:.4f}\n\n")
    f.write(report)

print("Evaluation files saved successfully!")