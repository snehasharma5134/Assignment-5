
# ==========================================
# CIFAR-10 DATASET AND DATALOADERS
# ==========================================

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset


def get_data_loaders():

    dataset_path = "./dataset"

    # CIFAR-10 channel statistics
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # Training transformations with augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Validation and testing transformations
    validation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load training dataset with augmentation
    train_dataset = datasets.CIFAR10(
        root=dataset_path,
        train=True,
        download=False,
        transform=train_transform
    )

    # Load training dataset without augmentation
    validation_dataset = datasets.CIFAR10(
        root=dataset_path,
        train=True,
        download=False,
        transform=validation_transform
    )

    # Split training data into 45,000 and 5,000
    train_indices, val_indices = random_split(
        range(len(train_dataset)),
        [45000, 5000],
        generator=torch.Generator().manual_seed(42)
    )

    # Create training and validation subsets
    train_subset = Subset(
        train_dataset, train_indices.indices
    )

    val_subset = Subset(
        validation_dataset, val_indices.indices
    )

    # Load official testing dataset
    test_dataset = datasets.CIFAR10(
        root=dataset_path,
        train=False,
        download=False,
        transform=validation_transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":

    train_loader, val_loader, test_loader = get_data_loaders()

    print("Dataset Loading Successful!")
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Testing batches:", len(test_loader))

    images, labels = next(iter(train_loader))

    print("Batch Shape:", images.shape)
    print("Labels Shape:", labels.shape)