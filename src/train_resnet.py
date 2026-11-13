# ==========================================
# CIFAR-10 TRANSFER LEARNING (RESNET-18)
# Fine-Tuning Pre-trained ImageNet Model
# ==========================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset

# ------------------------------------------
# 1. DEVICE SETUP (GPU / CPU)
# ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ------------------------------------------
# 2. DATA TRANSFORMS FOR RESNET-18
# Note: ImageNet models expect larger images (128x128) & ImageNet Mean/Std
# ------------------------------------------
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),          # Upscale from 32x32 so ResNet filters work well
    transforms.RandomHorizontalFlip(p=0.5), # Random flip for augmentation
    transforms.RandomRotation(10),          # Slight rotation
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# Validation / Test transforms (no random augmentation)
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# ------------------------------------------
# 3. LOAD CIFAR-10 DATASET (45k Train / 5k Val)
# ------------------------------------------
dataset_path = "./dataset"

full_train_aug = datasets.CIFAR10(root=dataset_path, train=True, download=False, transform=train_transform)
full_train_clean = datasets.CIFAR10(root=dataset_path, train=True, download=False, transform=val_transform)

# Same reproducible 45,000 / 5,000 split
train_indices, val_indices = random_split(
    range(len(full_train_aug)),
    [45000, 5000],
    generator=torch.Generator().manual_seed(42)
)

train_subset = Subset(full_train_aug, train_indices.indices)
val_subset = Subset(full_train_clean, val_indices.indices)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

# ------------------------------------------
# 4. LOAD PRE-TRAINED RESNET-18 & APPLY TRANSFER LEARNING
# ------------------------------------------
# Load ResNet-18 with official ImageNet weights
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all early layers so we keep ImageNet's edge & shape knowledge
for param in model.parameters():
    param.requires_grad = False

# Unfreeze Layer 4 (top conv block) for Fine-Tuning on CIFAR-10
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace the final 1000-class ImageNet head with a new 10-class CIFAR-10 head
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 10)
)

# Move model to GPU/CPU
model = model.to(device)

# ------------------------------------------
# 5. LOSS, OPTIMIZER & SCHEDULER
# ------------------------------------------
criterion = nn.CrossEntropyLoss()

# Only pass trainable parameters (layer4 + fc) to Adam with a small LR for fine-tuning
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0003,
    weight_decay=1e-4
)

epochs = 12  # Pre-trained models converge very fast (10-12 epochs is plenty!)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# ------------------------------------------
# 6. TRAINING & VALIDATION LOOP
# ------------------------------------------
train_losses, val_losses = [], []
train_accs, val_accs = [], []
os.makedirs("checkpoints", exist_ok=True)
best_val_acc = 0.0

for epoch in range(epochs):
    # --- Training Phase ---
    model.train()
    run_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        run_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    ep_train_loss = run_loss / total
    ep_train_acc = 100 * correct / total

    # --- Validation Phase ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    ep_val_loss = val_loss / val_total
    ep_val_acc = 100 * val_correct / val_total

    scheduler.step()

    train_losses.append(ep_train_loss)
    val_losses.append(ep_val_loss)
    train_accs.append(ep_train_acc)
    val_accs.append(ep_val_acc)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {ep_train_loss:.4f} | Train Acc: {ep_train_acc:.2f}% | "
        f"Val Loss: {ep_val_loss:.4f} | Val Acc: {ep_val_acc:.2f}%"
    )

    # Save best ResNet-18 checkpoint separately
    if ep_val_acc > best_val_acc:
        best_val_acc = ep_val_acc
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_accuracy": best_val_acc
            },
            "checkpoints/resnet18_cifar10.pth"
        )
        print(f"--> New Best ResNet-18 Saved! (Val Acc: {best_val_acc:.2f}%)")

# ------------------------------------------
# 7. SAVE TRAINING GRAPHS
# ------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet-18 Transfer Learning Loss")
plt.legend()
plt.savefig("checkpoints/resnet18_loss_curve.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(train_accs, label="Training Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("ResNet-18 Transfer Learning Accuracy")
plt.legend()
plt.savefig("checkpoints/resnet18_accuracy_curve.png")
plt.close()

print(f"\nResNet-18 Training Completed! Best Validation Accuracy: {best_val_acc:.2f}%")