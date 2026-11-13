# ==========================================
# CIFAR-10 6-LAYER CNN TRAINING SCRIPT
# ==========================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import CIFAR10_CNN
from dataset import get_data_loaders

# ------------------------------------------
# 1. DEVICE SETUP (Use RTX 2050 GPU if available, else CPU)
# ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ------------------------------------------
# 2. LOAD DATA (45k Train, 5k Validation, 10k Test)
# ------------------------------------------
train_loader, val_loader, test_loader = get_data_loaders()

# ------------------------------------------
# 3. INITIALIZE MODEL, LOSS, OPTIMIZER & SCHEDULER
# ------------------------------------------
model = CIFAR10_CNN().to(device)

# CrossEntropyLoss automatically applies Softmax + Categorical Cross-Entropy
criterion = nn.CrossEntropyLoss()

# Adam optimizer with weight_decay (L2 regularization) to avoid overfitting
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Total training passes over the full dataset
epochs = 30

# CosineAnnealingLR smoothly lowers the learning rate as epochs progress
# This helps the model settle into a much higher accuracy
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Lists to track loss & accuracy per epoch for plotting curves
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Create checkpoints folder if it doesn't exist
os.makedirs("checkpoints", exist_ok=True)
best_val_accuracy = 0.0

# ------------------------------------------
# 4. MAIN TRAINING & VALIDATION LOOP
# ------------------------------------------
for epoch in range(epochs):
    # --- TRAINING PHASE ---
    model.train()  # Enables Dropout and BatchNorm updates
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # Move batch to GPU/CPU
        images, labels = images.to(device), labels.to(device)

        # Clear old gradients from the previous step
        optimizer.zero_grad()

        # Forward pass: get model predictions
        outputs = model(images)

        # Calculate error (loss) between predictions and actual labels
        loss = criterion(outputs, labels)

        # Backward pass: compute gradients
        loss.backward()

        # Update model weights using Adam
        optimizer.step()

        # Track total loss and correct predictions
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)  # Pick class with highest score
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Average training loss and accuracy for this epoch
    epoch_train_loss = running_loss / total
    epoch_train_acc = 100 * correct / total

    # --- VALIDATION PHASE ---
    model.eval()  # Disables Dropout & locks BatchNorm for fair evaluation
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # No need to compute gradients during validation (saves memory & time)
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # Average validation loss and accuracy for this epoch
    epoch_val_loss = val_loss / val_total
    epoch_val_acc = 100 * val_correct / val_total

    # Update learning rate for the next epoch
    scheduler.step()

    # Save metrics to history lists
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_accuracies.append(epoch_train_acc)
    val_accuracies.append(epoch_val_acc)

    # Print epoch progress
    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
        f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%"
    )

    # --- SAVE BEST CHECKPOINT ---
    # Only overwrite the saved model if validation accuracy improves
    if epoch_val_acc > best_val_accuracy:
        best_val_accuracy = epoch_val_acc
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_accuracy
            },
            "checkpoints/best_model_30.pth"
        )
        print(f"--> New Best Model Saved! (Val Acc: {best_val_accuracy:.2f}%)")

# ------------------------------------------
# 5. PLOT & SAVE TRAINING CURVES
# ------------------------------------------
# Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss (6-Layer CNN)")
plt.legend()
plt.savefig("checkpoints/loss_curve_30.png")
plt.close()

# Accuracy Curve
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy (6-Layer CNN)")
plt.legend()
plt.savefig("checkpoints/accuracy_curve_30.png")
plt.close()

print("\nTraining Completed! Best Validation Accuracy:", f"{best_val_accuracy:.2f}%")
