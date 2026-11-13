# ============================================================
# CIFAR-10 AI VISION - MODERN STREAMLIT DASHBOARD
# Supports: 6-Layer Custom CNN (88.34%) & ResNet-18 Transfer Learning (92.92%)
# ============================================================

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from pathlib import Path

# Import our custom 6-Layer CNN architecture
try:
    from src.model import CIFAR10_CNN
except ImportError:
    from model import CIFAR10_CNN


# ============================================================
# 1. PAGE & PATH CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CIFAR-10 AI Vision | Dual-Model Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "outputs"

# Official 10 CIFAR-10 class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CLASS_ICONS = {
    "airplane": "✈️", "automobile": "🚗", "bird": "🐦", "cat": "🐱", "deer": "🦌",
    "dog": "🐶", "frog": "🐸", "horse": "🐴", "ship": "🚢", "truck": "🚚"
}

# Automatically pick GPU (CUDA) if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration dictionary for both trained models
MODEL_CONFIGS = {
    "ResNet-18 (Transfer Learning)": {
        "checkpoint": CHECKPOINT_DIR / "resnet18_cifar10.pth",
        "test_acc": "92.92%",
        "val_acc": "93.28%",
        "macro_f1": "0.9291",
        "epochs": "12",
        "input_size": "128 × 128 RGB",
        "acc_curve": "resnet18_accuracy_curve.png",
        "loss_curve": "resnet18_loss_curve.png",
        "cm_image": "confusion_matrix_ResNet18.png",
        "report_file": "classification_report_ResNet18.txt",
        "short_desc": "Pre-trained ImageNet ResNet-18 fine-tuned on CIFAR-10"
    },
    "6-Layer Custom CNN (From Scratch)": {
        "checkpoint": CHECKPOINT_DIR / "best_model_30.pth",
        "test_acc": "88.34%",
        "val_acc": "89.12%",
        "macro_f1": "0.8819",
        "epochs": "30",
        "input_size": "32 × 32 RGB",
        "acc_curve": "accuracy_curve_30.png",
        "loss_curve": "loss_curve_30.png",
        "cm_image": "confusion_matrix_Fine_Tuned_30.png",
        "report_file": "classification_report_Fine_Tuned_30.txt",
        "short_desc": "3-Block (6 Conv Layers) CNN trained from scratch"
    }
}


# ============================================================
# 2. CUSTOM CSS STYLING (MODERN & CLEAN UI)
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 6px;
        background: linear-gradient(90deg, #60A5FA, #A78BFA, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1rem;
        margin: 0;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.3);
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        font-weight: 600;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 4px 0;
    }
    .metric-caption {
        font-size: 0.78rem;
        color: #64748B;
    }

    .prediction-hero {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.15) 0%, rgba(124, 58, 237, 0.15) 100%);
        border: 1.5px solid rgba(96, 165, 250, 0.4);
        border-radius: 18px;
        padding: 24px;
        text-align: center;
        margin-top: 10px;
    }
    .pred-badge {
        display: inline-block;
        padding: 4px 14px;
        background: rgba(96, 165, 250, 0.2);
        color: #93C5FD;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .pred-class {
        font-size: 2.3rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 4px 0 10px 0;
    }
    .pred-confidence {
        font-size: 1.25rem;
        color: #38BDF8;
        font-weight: 700;
    }
    
    .badge-gpu {
        background: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-cpu {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. MODEL LOADING & PREPROCESSING PIPELINES
# ============================================================

# Cache both models separately based on model_name
@st.cache_resource(show_spinner=False)
def load_selected_model(model_name):
    ckpt_path = MODEL_CONFIGS[model_name]["checkpoint"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build the matching architecture
    if model_name == "ResNet-18 (Transfer Learning)":
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 10)
        )
    else:
        model = CIFAR10_CNN()

    # Load saved weights onto GPU/CPU
    model = model.to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Disable Dropout & lock BatchNorm for inference
    return model


def get_transform(model_name):
    """Returns the exact preprocessing transform expected by the selected model."""
    if model_name == "ResNet-18 (Transfer Learning)":
        # ResNet-18 expects 128x128 RGB with ImageNet normalization
        return transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        # Custom 6-Layer CNN expects 32x32 RGB with CIFAR-10 normalization
        return transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def render_metric_card(label, value, caption):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-val">{value}</div>
        <div class="metric-caption">{caption}</div>
    </div>
    """, unsafe_allow_html=True)


def find_file(filename):
    """Searches outputs/ first, then checkpoints/."""
    p1 = OUTPUT_DIR / filename
    if p1.exists():
        return p1
    p2 = CHECKPOINT_DIR / filename
    if p2.exists():
        return p2
    return None


def display_visual_asset(filename, caption):
    file_path = find_file(filename)
    if file_path:
        st.image(str(file_path), caption=caption, use_container_width=True)
    else:
        st.info(f"💡 Visual asset `{filename}` will appear here once generated.")


def clear_old_prediction():
    """Clears stale prediction when image or model changes."""
    if "inference_result" in st.session_state:
        del st.session_state["inference_result"]


# ============================================================
# 5. SIDEBAR NAVIGATION & MODEL SELECTOR
# ============================================================

with st.sidebar:
    st.markdown("### 🧠 CIFAR-10 AI Vision")
    st.caption("Dual-Model Deep Learning Dashboard")
    st.divider()

    # Model Switcher Dropdown
    selected_model = st.selectbox(
        "🤖 SELECT ACTIVE MODEL",
        list(MODEL_CONFIGS.keys()),
        index=0,
        on_change=clear_old_prediction
    )
    active_cfg = MODEL_CONFIGS[selected_model]

    st.divider()

    # Page Navigation
    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Overview",
            "🔍 Image Classification",
            "📊 Model Analytics",
            "ℹ️ About Project"
        ],
        index=1
    )

    st.divider()
    st.markdown("#### ⚙️ Active Model Specs")
    st.write(f"**Test Accuracy:** `{active_cfg['test_acc']}`")
    st.write(f"**Macro F1:** `{active_cfg['macro_f1']}`")
    st.write(f"**Input Size:** `{active_cfg['input_size']}`")
    st.write(f"**Checkpoint:** `{active_cfg['checkpoint'].name}`")

    st.divider()
    st.markdown("#### 🖥️ Hardware Runtime")
    if device.type == "cuda":
        st.markdown(f'<span class="badge-gpu">● GPU ACTIVE ({torch.cuda.get_device_name(0)})</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-cpu">● CPU MODE</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Internship Project • PyTorch + Streamlit")


# ============================================================
# 6. HEADER BANNER
# ============================================================

st.markdown(f"""
<div class="hero-card">
    <div class="hero-title">🧠 CIFAR-10 AI Vision Studio</div>
    <div class="hero-subtitle">Active Engine: <b>{selected_model}</b> • Test Accuracy: <b>{active_cfg['test_acc']}</b></div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "🏠 Overview":
    st.markdown("### 📌 Dual-Model Performance Overview")
    st.write("Compare our **6-Layer Custom CNN (trained from scratch)** against **ResNet-18 (Transfer Learning)**.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("ResNet-18 Test Acc", "92.92%", "Transfer Learning (12 Epochs)")
    with col2:
        render_metric_card("Custom CNN Test Acc", "88.34%", "6-Layer Scratch (30 Epochs)")
    with col3:
        render_metric_card("Dataset Size", "60,000", "50k Train + 10k Test Images")
    with col4:
        render_metric_card("Categories", "10", "Balanced CIFAR-10 Classes")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Supported Image Classes")

    c_cols = st.columns(5)
    for i, cls in enumerate(CLASS_NAMES):
        with c_cols[i % 5]:
            with st.container(border=True):
                st.markdown(f"<h3 style='text-align: center; margin: 0;'>{CLASS_ICONS[cls]}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-weight: 600; margin: 4px 0 0 0;'>{cls.capitalize()}</p>", unsafe_allow_html=True)


# ============================================================
# PAGE 2: IMAGE CLASSIFICATION
# ============================================================

elif page == "🔍 Image Classification":
    st.markdown("### 🔍 Live Image Classifier")
    st.caption(f"Currently predicting using **{selected_model}** ({active_cfg['short_desc']}).")

    col_upload, col_preview = st.columns([1, 1], gap="medium")
    uploaded_image = None

    with col_upload:
        with st.container(border=True):
            st.markdown("#### 📤 Upload Picture")
            file_buffer = st.file_uploader(
                "Drag and drop image here",
                type=["png", "jpg", "jpeg", "webp"],
                help="Supports JPG, PNG, WEBP files",
                on_change=clear_old_prediction
            )
            st.caption("💡 Switch between Custom CNN and ResNet-18 in the sidebar to compare their predictions!")

    with col_preview:
        with st.container(border=True):
            st.markdown("#### 🖼️ Image Preview")
            if file_buffer:
                uploaded_image = Image.open(file_buffer).convert("RGB")
                st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
            else:
                st.info("Uploaded preview will appear here.")

    if uploaded_image:
        st.markdown("<br>", unsafe_allow_html=True)
        classify_btn = st.button(f"⚡ Classify with {selected_model}", type="primary", use_container_width=True)

        if classify_btn:
            with st.spinner(f"Analyzing image with {selected_model}..."):
                try:
                    model = load_selected_model(selected_model)
                    transform_fn = get_transform(selected_model)

                    # Preprocess image & add batch dimension
                    tensor_img = transform_fn(uploaded_image).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(tensor_img)
                        probs = F.softmax(logits, dim=1)
                        top_probs, top_indices = torch.topk(probs, k=3, dim=1)

                    all_scores = probs[0].cpu().numpy() * 100
                    top_probs = top_probs[0].cpu().numpy() * 100
                    top_indices = top_indices[0].cpu().numpy()

                    st.session_state["inference_result"] = {
                        "model_used": selected_model,
                        "class": CLASS_NAMES[top_indices[0]],
                        "confidence": float(top_probs[0]),
                        "top_classes": [CLASS_NAMES[i] for i in top_indices],
                        "top_probs": top_probs.tolist(),
                        "all_scores": all_scores.tolist()
                    }
                except Exception as e:
                    st.error(f"Inference error: {str(e)}")

        if "inference_result" in st.session_state:
            res = st.session_state["inference_result"]
            st.divider()

            res_col1, res_col2 = st.columns([1.1, 1.4], gap="large")

            with res_col1:
                icon = CLASS_ICONS.get(res["class"], "🎯")
                st.markdown(f"""
                <div class="prediction-hero">
                    <span class="pred-badge">{res['model_used']}</span>
                    <div class="pred-class">{icon} {res['class'].upper()}</div>
                    <div class="pred-confidence">{res['confidence']:.2f}% Confidence</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 🏆 Top 3 Candidates")
                for rank, (cls_name, prob) in enumerate(zip(res["top_classes"], res["top_probs"]), start=1):
                    c_icon = CLASS_ICONS.get(cls_name, "▪️")
                    col_t1, col_t2 = st.columns([2, 1])
                    with col_t1:
                        st.write(f"**{rank}. {c_icon} {cls_name.capitalize()}**")
                    with col_t2:
                        st.write(f"`{prob:.2f}%`")
                    st.progress(min(max(float(prob) / 100.0, 0.0), 1.0))

            with res_col2:
                st.markdown("#### 📊 Confidence Score Distribution")
                df_chart = pd.DataFrame({
                    "Category": [c.capitalize() for c in CLASS_NAMES],
                    "Confidence (%)": res["all_scores"]
                }).sort_values(by="Confidence (%)", ascending=True)

                st.bar_chart(df_chart, x="Category", y="Confidence (%)", horizontal=True)
                st.caption("Softmax probability distribution across all 10 CIFAR-10 classes.")


# ============================================================
# PAGE 3: MODEL ANALYTICS
# ============================================================

elif page == "📊 Model Analytics":
    st.markdown(f"### 📊 Analytics: {selected_model}")
    st.caption("Switch models in the sidebar to inspect each model's training curves, confusion matrix, and report.")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Test Accuracy", active_cfg["test_acc"], "Official 10k Test Set")
    with m2:
        render_metric_card("Macro F1-Score", active_cfg["macro_f1"], "Balanced multi-class F1")
    with m3:
        render_metric_card("Training Epochs", active_cfg["epochs"], "CosineAnnealingLR")
    with m4:
        render_metric_card("Validation Acc", active_cfg["val_acc"], "Peak Validation Score")

    st.markdown("<br>", unsafe_allow_html=True)

    tab_curves, tab_cm, tab_report, tab_compare = st.tabs([
        "📈 Loss & Accuracy Curves",
        "🎯 Confusion Matrix",
        "📋 Classification Report",
        "⚖️ Model Comparison"
    ])

    with tab_curves:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Training & Validation Accuracy")
            display_visual_asset(active_cfg["acc_curve"], f"{selected_model} Accuracy Curve")
        with c2:
            st.markdown("##### Training & Validation Loss")
            display_visual_asset(active_cfg["loss_curve"], f"{selected_model} Loss Curve")

    with tab_cm:
        st.markdown(f"##### Confusion Matrix ({selected_model})")
        display_visual_asset(active_cfg["cm_image"], f"{selected_model} Confusion Matrix")

    with tab_report:
        st.markdown(f"##### Per-Class Report ({selected_model})")
        rep_file = find_file(active_cfg["report_file"])
        if rep_file:
            st.code(rep_file.read_text(encoding="utf-8"), language="text")
        else:
            st.info("Classification report file not found in outputs.")

    with tab_compare:
        st.markdown("##### 6-Layer Custom CNN vs ResNet-18 Transfer Learning")
        comp_file = find_file("model_comparison.txt")
        if comp_file:
            st.code(comp_file.read_text(encoding="utf-8"), language="text")
        else:
            st.info("Run `python src/evaluate_resnet.py` to generate the comparison summary.")


# ============================================================
# PAGE 4: ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":
    st.markdown("### ℹ️ Project Architecture & Methodology")

    with st.expander("🧠 Approach 1: 6-Layer Custom CNN (From Scratch)", expanded=True):
        st.markdown("""
        - **Architecture:** 3 Blocks × 2 Conv2D layers (`64 → 128 → 256` filters) + BatchNorm + MaxPool + Dropout (0.2–0.5).
        - **Training:** 30 Epochs, Adam Optimizer (`lr=0.001`), CosineAnnealingLR, Data Augmentation.
        - **Official Test Accuracy:** **88.34%** (Macro F1: **0.8819**)
        """)

    with st.expander("🚀 Approach 2: ResNet-18 (Transfer Learning)", expanded=True):
        st.markdown("""
        - **Architecture:** Pre-trained ImageNet ResNet-18 backbone; frozen early layers + fine-tuned `layer4` & custom 10-class head.
        - **Training:** 12 Epochs, Adam Optimizer (`lr=0.0003`), `128 × 128` input resolution.
        - **Official Test Accuracy:** **92.92%** (Macro F1: **0.9291**)
        """)

    with st.expander("🛠️ Tech Stack"):
        st.markdown("Python | PyTorch | Torchvision | Scikit-Learn | Pandas | Matplotlib | Streamlit")


# ============================================================
# FOOTER
# ============================================================

st.markdown("<br><hr style='border: 0.5px solid rgba(255, 255, 255, 0.1);'><br>", unsafe_allow_html=True)
f_left, f_right = st.columns(2)
with f_left:
    st.caption("🧠 **CIFAR-10 AI Vision** • Dual-Model Internship Project")
with f_right:
    st.caption("<div style='text-align: right;'>Powered by PyTorch & Streamlit</div>", unsafe_allow_html=True)