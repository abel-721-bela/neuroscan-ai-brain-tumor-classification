
import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

# --------------------------------------------------
# Streamlit user interface
# --------------------------------------------------

st.set_page_config(
    page_title="NeuroScan AI | MRI Classification",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
CONFIG_PATH = OUTPUT_DIR / "project_config.json"

DEVICE = torch.device("cpu")

MODEL_FILES = {
    "Custom CNN": "custom_cnn_final.pth",
    "MobileNetV3 Head": "mobilenet_v3_head_final.pth",
    "MobileNetV3 Fine-tuned": "mobilenet_v3_finetuned_final.pth",
}


# --------------------------------------------------
# Custom CNN architecture
# Must match the training notebook exactly
# --------------------------------------------------

class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------------------------------
# Load the selected model and its metadata
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    model_name = config["selected_model"]

    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown selected model: {model_name}")

    model_path = MODEL_DIR / MODEL_FILES[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    # Only load trusted project artifacts using safe loading.
    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]
    image_size = checkpoint["image_size"]

    architecture = checkpoint["architecture"]

    if architecture == "BrainTumorCNN":
        model = BrainTumorCNN(num_classes)

    elif architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features,
            num_classes
        )

    else:
        raise ValueError(
            f"Unsupported model architecture: {architecture}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    mean = checkpoint["normalization_mean"]
    std = checkpoint["normalization_std"]

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return (
        model,
        class_names,
        image_size,
        image_transform,
        model_name
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_image(image, model, class_names, image_transform):
    image = image.convert("RGB")
    tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())

    return (
        class_names[predicted_index],
        probabilities.cpu().tolist()
    )


st.markdown("""
<style>
    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 1.6rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #10233F, #174C66);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        margin-bottom: 0.35rem;
        font-size: 2.2rem;
    }
    .hero p {
        color: #D9EAF5;
        margin-bottom: 0;
    }
    .disclaimer {
        padding: 1rem;
        border: 1px solid #D97706;
        border-radius: 12px;
        background: rgba(217, 119, 6, 0.08);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🧠 NeuroScan AI</h1>
    <p>Brain MRI image classification powered by deep learning</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Upload a brain MRI image to explore the prediction made by "
    "the trained image-classification model."
)

try:
    model, class_names, image_size, image_transform, model_name = (
        load_model()
    )
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

with st.sidebar:
    st.header("About the model")
    st.write(f"**Architecture:** {model_name}")
    st.write(f"**Input size:** {image_size} × {image_size}")
    st.write(f"**Classes:** {len(class_names)}")
    st.divider()
    st.caption(
        "This educational prototype is not a medical diagnostic system."
    )

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Supported formats: JPG, JPEG, PNG, BMP and WebP."
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1.2], gap="large")

        with left:
            st.subheader("Uploaded image")
            st.image(image, use_container_width=True)

        with right:
            st.subheader("Classification result")

            with st.spinner("Analyzing image..."):
                predicted_class, probabilities = predict_image(
                    image,
                    model,
                    class_names,
                    image_transform
                )

            confidence = probabilities[class_names.index(predicted_class)]

            st.success("Prediction generated")
            st.metric(
                label="Predicted category",
                value=predicted_class.replace("_", " ").title()
            )
            st.metric(
                label="Model confidence",
                value=f"{confidence * 100:.2f}%"
            )

            st.caption(
                "Confidence is the model's softmax score, "
                "not a measure of clinical certainty."
            )

        st.divider()
        st.subheader("Class probability scores")

        ranked = sorted(
            zip(class_names, probabilities),
            key=lambda item: item[1],
            reverse=True
        )

        for class_name, probability in ranked:
            label = class_name.replace("_", " ").title()
            st.write(f"**{label}** — {probability * 100:.2f}%")
            st.progress(float(probability))

    except Exception as error:
        st.error(f"Could not process this image: {error}")

st.divider()

st.markdown("""
<div class="disclaimer">
<strong>Important medical disclaimer</strong><br><br>
This application is an educational machine-learning prototype.
It has not been validated for clinical use and must not be used to
diagnose, rule out, or make treatment decisions about a brain tumor.
Consult a qualified medical professional for medical interpretation.
</div>
""", unsafe_allow_html=True)

st.caption(
    "Brain Tumor MRI Image Classification"
)
