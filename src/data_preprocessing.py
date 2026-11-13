
# ==========================================
# 1. IMPORT REQUIRED LIBRARIES
# ==========================================

# PyTorch: Used for tensor operations and deep learning
import torch

# Torchvision: Provides datasets and image processing tools
from torchvision import datasets, transforms

# Matplotlib: Used to visualize images and graphs
import matplotlib.pyplot as plt

# NumPy: Used for numerical operations and array handling
import numpy as np

# Import DataLoader and Subset for dataset preparation
from torch.utils.data import DataLoader, Subset

# ==========================================
# 2. LOAD CIFAR-10 TRAINING DATASET
# ==========================================

# Define the folder where CIFAR-10 dataset will be stored
dataset_path = "./dataset"

# Load the CIFAR-10 training dataset
train_dataset = datasets.CIFAR10(
    root=dataset_path,  # Location to store the dataset
    train=True,         # Load the 50,000 training images
    download=True       # Download if the dataset is not available
)

# Display a message after successful dataset loading
print("CIFAR-10 Training Dataset Loaded Successfully!")


# ==========================================
# 3. INSPECT THE TRAINING DATASET
# ==========================================

# Display the total number of training images
print("\nTotal Training Images:", len(train_dataset))

# Display the available class names
print("\nCIFAR-10 Classes:", train_dataset.classes)

# Access the first image and its corresponding label
image, label = train_dataset[0]

# Display the image information
print("\nFirst Image Information:")
print("Image Type:", type(image))
print("Image Size:", image.size)
print("Image Label:", label)

# Display the corresponding class name
print("Class Name:", train_dataset.classes[label])

# ==========================================
# 4. VISUALIZE 10 RANDOM CIFAR-10 IMAGES
# ==========================================

# Select 10 random image indices from the training dataset
random_indices = torch.randperm(len(train_dataset))[:10]

# Create a figure containing 2 rows and 5 columns
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

# Convert the 2D axes array into a 1D array for easy iteration
axes = axes.flatten()

# Loop through the 10 selected random images
for i, index in enumerate(random_indices):

    # Get the image and its numerical label
    image, label = train_dataset[index]

    # Display the image in the current subplot
    axes[i].imshow(image)

    # Convert the numerical label into its class name
    class_name = train_dataset.classes[label]

    # Display the class name above the image
    axes[i].set_title(class_name)

    # Hide the x-axis and y-axis
    axes[i].axis("off")

# Adjust spacing between images
plt.tight_layout()

# Display the complete image grid
plt.show()


# ==========================================
# TASK 4: IMAGE PREPROCESSING USING ToTensor()
# ==========================================

# Import the ToTensor transformation
from torchvision import transforms

# Select the first image from the existing dataset
image, label = train_dataset[0]

# Create a ToTensor transformation
to_tensor = transforms.ToTensor()

# Convert the PIL image into a PyTorch tensor
image_tensor = to_tensor(image)

# Display the converted image information
print("\nImage After ToTensor Conversion:")

# Check the tensor data type
print("Image Type:", type(image_tensor))

# Check the tensor dimensions
print("Image Shape:", image_tensor.shape)

# Check the numerical data type
print("Data Type:", image_tensor.dtype)

# Find the minimum pixel value
print("Minimum Pixel Value:", image_tensor.min().item())

# Find the maximum pixel value
print("Maximum Pixel Value:", image_tensor.max().item())

# Display the original image class
print("Class Name:", train_dataset.classes[label])


# Define image preprocessing transformations
preprocess_transform = transforms.Compose([

    # Convert image into tensor (0 to 1)
    transforms.ToTensor(),

    # Standardize image using CIFAR-10 statistics
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

# Load CIFAR-10 dataset with preprocessing
processed_dataset = datasets.CIFAR10(
    root=dataset_path,
    train=True,
    download=False,
    transform=preprocess_transform
)

# Get one preprocessed image
processed_image, label = processed_dataset[0]

# Verify preprocessing
print("\nAfter Standardization:")
print("Image Shape:", processed_image.shape)
print("Data Type:", processed_image.dtype)
print("Minimum Value:", processed_image.min().item())
print("Maximum Value:", processed_image.max().item())


# ==========================================
# TASK 5: DATA AUGMENTATION
# ==========================================

# Define transformations for training images
augmentation_transform = transforms.Compose([

    # Randomly flip images horizontally
    transforms.RandomHorizontalFlip(p=0.5),

    # Randomly rotate images between -15 and +15 degrees
    transforms.RandomRotation(15),

    # Apply small horizontal and vertical shifts
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)
    )
])

# Get one original image
image, label = train_dataset[0]

# Apply augmentation
augmented_image = augmentation_transform(image)

# Display original and augmented images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(augmented_image)
plt.title("Augmented Image")
plt.axis("off")

plt.tight_layout()
plt.show()


# ==========================================
# TASK 6: COMPLETE TRAINING TRANSFORM
# ==========================================

# Combine augmentation and preprocessing
train_transform = transforms.Compose([

    # Random horizontal flip
    transforms.RandomHorizontalFlip(p=0.5),

    # Random rotation up to 15 degrees
    transforms.RandomRotation(15),

    # Small horizontal and vertical shifts
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)
    ),

    # Convert image into tensor
    transforms.ToTensor(),

    # Standardize RGB channels
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

# Load CIFAR-10 with complete training transformations
augmented_train_dataset = datasets.CIFAR10(
    root=dataset_path,
    train=True,
    download=False,
    transform=train_transform
)

# Check one transformed image
image, label = augmented_train_dataset[0]

print("\nComplete Training Pipeline:")
print("Dataset Size:", len(augmented_train_dataset))
print("Image Shape:", image.shape)
print("Image Data Type:", image.dtype)
print("Class:", train_dataset.classes[label])


# ==========================================
# TASK 7: TRAIN, VALIDATION AND TEST SPLIT
# ==========================================

from torch.utils.data import DataLoader, random_split, Subset

# Define transformations for validation and testing
validation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

# Load validation dataset without random augmentation
validation_dataset = datasets.CIFAR10(
    root=dataset_path,
    train=True,
    download=False,
    transform=validation_transform
)

# Create reproducible training and validation indices
train_indices, val_indices = random_split(
    range(len(train_dataset)),
    [45000, 5000],
    generator=torch.Generator().manual_seed(42)
)

# Create training and validation subsets
train_subset = Subset(
    augmented_train_dataset, train_indices.indices
)

val_subset = Subset(
    validation_dataset, val_indices.indices
)

# Load official testing dataset separately
test_dataset = datasets.CIFAR10(
    root=dataset_path,
    train=False,
    download=False,
    transform=validation_transform
)

# Display dataset sizes
print("\nDataset Split:")
print("Training Images:", len(train_subset))
print("Validation Images:", len(val_subset))
print("Testing Images:", len(test_dataset))


# ==========================================
# TASK 8: CREATE DATALOADERS
# ==========================================

# Define batch size
batch_size = 64

# Create training DataLoader
train_loader = DataLoader(
    train_subset,
    batch_size=batch_size,
    shuffle=True
)

# Create validation DataLoader
val_loader = DataLoader(
    val_subset,
    batch_size=batch_size,
    shuffle=False
)

# Create testing DataLoader
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Verify one training batch
images, labels = next(iter(train_loader))

print("\nDataLoader Verification:")
print("Batch Image Shape:", images.shape)
print("Batch Label Shape:", labels.shape)
print("Batch Size:", batch_size)
