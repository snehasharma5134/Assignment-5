# 🧠 CIFAR-10 Image Classification Using Deep Learning (Custom CNN & Transfer Learning)

## 📌 Project Overview

CIFAR-10 Image Classification is a Deep Learning project that classifies real-world color images across **10 distinct object categories** (`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, and `truck`).

The project implements and compares two complete PyTorch modeling approaches:
1. A **6-Layer VGG-Style Custom CNN** trained from scratch (`88.34%` Test Accuracy).
2. A **ResNet-18 Transfer Learning** model fine-tuned on CIFAR-10 (`92.92%` Test Accuracy).

An interactive **Streamlit Web Dashboard** (`app.py`) is built to let users switch live between both models, upload images via drag-and-drop, view Top-3 predictions with confidence scores, and inspect training curves, confusion matrices, and classification reports.

## 🎯 Project Objectives

- Load, inspect, and visualize the CIFAR-10 dataset (60,000 RGB images across 10 classes).
- Apply real-time data augmentation (horizontal flips, rotations, affine shifts) and channel normalization.
- Design and train a deeper 6-Layer CNN from scratch using Batch Normalization, Max Pooling, Dropout, and Cosine Annealing LR.
- Implement Transfer Learning using a pre-trained ResNet-18 backbone and compare performance against the custom CNN.
- Evaluate models using Accuracy, Macro F1-Score, Confusion Matrix, and Classification Reports.
- Deploy both trained models inside an interactive Streamlit web dashboard.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| PyTorch | Deep Learning Framework |
| Torchvision | Dataset Loading, Augmentation & Pre-trained ResNet-18 |
| CIFAR-10 | 10-Class RGB Image Dataset |
| Streamlit | Interactive Web Dashboard |
| Scikit-learn | Confusion Matrix, F1-Score & Classification Report |
| Pandas & NumPy | Data Handling & Numerical Operations |
| PIL (Pillow) | Image Loading & Preprocessing |
| Matplotlib | Training Curves & Confusion Matrix Plots |

## 📂 Project Structure

```text
CIFAR10_Image_Classification/
│
├── .streamlit/
│   └── config.toml
├── checkpoints/
│   ├── best_model_30.pth
│   ├── resnet18_cifar10.pth
│   ├── accuracy_curve_30.png
│   ├── loss_curve_30.png
│   ├── resnet18_accuracy_curve.png
│   └── resnet18_loss_curve.png
├── dataset/
│   └── cifar-10-batches-py/
├── outputs/
│   ├── confusion_matrix_Fine_Tuned_30.png
│   ├── confusion_matrix_ResNet18.png
│   ├── classification_report_Fine_Tuned_30.txt
│   ├── classification_report_ResNet18.txt
│   └── model_comparison.txt
├── src/
│   ├── data_preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── train_resnet.py
│   ├── evaluate.py
│   └── evaluate_resnet.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

**1. src/data_preprocessing.py**
- Loads and inspects the CIFAR-10 dataset.
- Plots a grid of 10 random labeled images and demonstrates `ToTensor()`, normalization, and augmentation transforms.

**2. src/dataset.py**
- Splits the 50,000 training images into **45,000 Training** and **5,000 Validation** subsets (with seed `42`).
- Applies real-time training augmentations (`RandomHorizontalFlip`, `RandomRotation(15)`, `RandomAffine`) and builds PyTorch `DataLoader` objects.

**3. src/model.py**
- Defines the **6-Layer VGG-Style Custom CNN** (`CIFAR10_CNN`) with 3 convolutional blocks (`64 → 128 → 256` filters), BatchNorm2d, ReLU, MaxPool2d, Dropout2d, and a Dense Classifier head.

**4. src/train.py**
- Trains the 6-Layer Custom CNN from scratch for **30 epochs** using `CrossEntropyLoss`, `Adam` (`lr=0.001`), and `CosineAnnealingLR`.
- Saves the best weights to `checkpoints/best_model_30.pth` and exports loss/accuracy curves.

**5. src/train_resnet.py**
- Implements **Transfer Learning** using ImageNet pre-trained **ResNet-18**.
- Freezes early convolutional layers, fine-tunes `layer4` and a custom 10-class head for **12 epochs** (`lr=0.0003`).
- Saves the best weights to `checkpoints/resnet18_cifar10.pth` and exports ResNet-18 training curves.

**6. src/evaluate.py & src/evaluate_resnet.py**
- `evaluate.py` evaluates the 6-Layer Custom CNN on the 10,000 test images and saves its confusion matrix and classification report.
- `evaluate_resnet.py` evaluates ResNet-18 and generates a side-by-side comparison (`outputs/model_comparison.txt`).

**7. app.py**
- Builds the Streamlit web dashboard with a **Sidebar Model Switcher** to toggle between **ResNet-18 (Transfer Learning)** and **6-Layer Custom CNN**.
- Handles image upload, center-crop preprocessing, Top-3 predictions, 10-class confidence bar chart, and model analytics visualization.

## 📊 Dataset Information

This project uses the **CIFAR-10** (Canadian Institute For Advanced Research) dataset.

| Property | Details |
|---|---|
| Training Images | 45,000 |
| Validation Images | 5,000 |
| Testing Images | 10,000 |
| Total Images | 60,000 |
| Native Dimensions | 32 × 32 pixels (RGB) |
| Number of Classes | 10 |
| Classes | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |

### Data Preprocessing & Augmentation

1. **Custom CNN (`32 × 32`):** Uses `Resize(32)` + `CenterCrop(32)` for inference, `ToTensor()`, and CIFAR-10 channel normalization (`mean=(0.4914, 0.4822, 0.4465)`, `std=(0.2470, 0.2435, 0.2616)`).
2. **ResNet-18 (`128 × 128`):** Uses `Resize(128)` + `CenterCrop(128)`, `ToTensor()`, and ImageNet normalization (`mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`).
3. **Training Augmentation:** Random horizontal flip (`p=0.5`), random rotation (`±15°`), and random affine shifts (`translate=(0.1, 0.1)`).

## 🧠 Model Architectures & Training Configuration

### 1. 6-Layer Custom CNN (`CIFAR10_CNN`)

| Block / Layer | Configuration |
|---|---|
| Conv Block 1 | Conv2d (`3 → 64`) + BN + ReLU + Conv2d (`64 → 64`) + BN + ReLU + MaxPool(2×2) + Dropout2d(0.2) |
| Conv Block 2 | Conv2d (`64 → 128`) + BN + ReLU + Conv2d (`128 → 128`) + BN + ReLU + MaxPool(2×2) + Dropout2d(0.3) |
| Conv Block 3 | Conv2d (`128 → 256`) + BN + ReLU + Conv2d (`256 → 256`) + BN + ReLU + MaxPool(2×2) + Dropout2d(0.4) |
| Dense Classifier | Flatten (`4096`) → Linear (`4096 → 512`) + BN1d + ReLU + Dropout(0.5) → Linear (`512 → 10`) |

### 2. ResNet-18 (Transfer Learning)

| Component | Configuration |
|---|---|
| Backbone | Pre-trained ImageNet `ResNet-18` (`conv1`, `bn1`, `layer1`–`layer3` frozen) |
| Fine-Tuned Layers | `layer4` unfrozen for domain adaptation |
| Custom Head | Dropout(0.3) → Linear (`512 → 10`) |

### Training Parameters

| Parameter | 6-Layer Custom CNN | ResNet-18 (Transfer Learning) |
|---|---:|---:|
| Optimizer | Adam | Adam |
| Loss Function | CrossEntropyLoss | CrossEntropyLoss |
| Learning Rate | 0.001 (`CosineAnnealingLR`) | 0.0003 (`CosineAnnealingLR`) |
| Weight Decay | 1e-4 | 1e-4 |
| Batch Size | 64 | 64 |
| Epochs | 30 | 12 |

## 📈 Model Performance

Both models were evaluated on the official **10,000 CIFAR-10 test images**:

| Model | Validation Accuracy | Official Test Accuracy | Macro F1-Score | Weighted F1-Score |
|---|---:|---:|---:|---:|
| **6-Layer Custom CNN (From Scratch)** | 89.12% | **88.34%** | 0.8819 | 0.8819 |
| **ResNet-18 (Transfer Learning)** | 93.28% | **92.92%** | 0.9291 | 0.9291 |

## 🖥️ Application Features

The Streamlit dashboard (`app.py`) provides:

- **Sidebar Model Switcher:** Instantly switch between **ResNet-18 (Transfer Learning)** and **6-Layer Custom CNN**.
- **Drag-and-Drop Image Uploader:** Upload JPG, JPEG, PNG, or WEBP images with live preview.
- **Aspect-Ratio Safe Preprocessing:** Automatic center-cropping prevents distortion on wide or tall photos.
- **Top-3 Candidates & Confidence Chart:** Displays the predicted class, Top-3 progress bars, and a 10-class softmax confidence distribution chart.
- **Model Analytics Page:** Interactive tabs for Loss/Accuracy curves, Confusion Matrices, Classification Reports, and Model Comparison.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.x
- pip
- VS Code or any Python IDE (GPU optional for inference, recommended for training)

### Step 1: Clone the Repository

```bash
git clone [https://github.com/snehasharma5134/Assignment-5.git](https://github.com/snehasharma5134/Assignment-5.git)
cd Assignment-5
```

### Step 2: Create & Activate Virtual Environment

```bash
python -m venv venv
```

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Train & Evaluate Models *(Optional)*

```bash
# Train & evaluate 6-Layer Custom CNN
python src/train.py
python src/evaluate.py

# Train & evaluate ResNet-18 Transfer Learning
python src/train_resnet.py
python src/evaluate_resnet.py
```

**Note:** Pre-trained checkpoints are already saved in `checkpoints/`, so retraining is not required to run the dashboard.

### Step 5: Run the Streamlit Application

```bash
streamlit run app.py
```

## 🔄 Project Workflow

```text
CIFAR-10 Dataset
      |
      v
Data Augmentation & Normalization
      |
      +-------------------------------+
      |                               |
      v                               v
6-Layer Custom CNN           ResNet-18 Transfer Learning
(Train 30 Epochs)            (Fine-Tune 12 Epochs)
      |                               |
      v                               v
best_model_30.pth            resnet18_cifar10.pth
      |                               |
      +---------------+---------------+
                      |
                      v
         Evaluation & Comparison
         (88.34% vs 92.92% Acc)
                      |
                      v
         Streamlit Web Dashboard
       (Sidebar Model Switcher UI)
```

## 📚 Learning Outcomes

Through this project, I learned:

- Loading, visualizing, and augmenting multi-class RGB datasets with Torchvision.
- Designing deeper CNN architectures with Batch Normalization, Max Pooling, and Dropout.
- Understanding why deeper layers (3 → 6 Conv layers) and Cosine Annealing LR boost representation capacity.
- Implementing Transfer Learning (Freezing & Fine-Tuning) using pre-trained ResNet-18.
- Evaluating models using Confusion Matrices, Precision, Recall, and Macro F1-Scores.
- Deploying a dual-model PyTorch inference pipeline inside a modern Streamlit dashboard.

## 👩‍💻 Author

**Sneha**

B.Tech – Computer Science and Engineering (AI & ML)

GitHub: https://github.com/snehasharma5134

---

⭐ If you find this project interesting, feel free to explore the repository!