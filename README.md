# NeuroScan AI — Brain Tumor MRI Image Classification

A deep-learning project that classifies brain MRI images into four categories using a custom Convolutional Neural Network (CNN) and transfer learning with MobileNetV3.


## Overview

NeuroScan AI explores how deep learning and computer vision can be applied to brain MRI image classification. The project compares a custom CNN trained from scratch with pretrained MobileNetV3 models and provides a Streamlit interface for uploading an MRI image and viewing the predicted category and class probability scores.

**This is an educational prototype, not a clinically validated diagnostic system.**

## Classification Categories

- Glioma
- Meningioma
- No Tumor
- Pituitary

## Models Implemented

1. **Custom CNN:** A convolutional neural network built from scratch using convolution, batch normalization, ReLU activation, max pooling, adaptive average pooling and dropout.
2. **MobileNetV3 Head:** A pretrained MobileNetV3 Small feature extractor with a newly trained classification head.
3. **MobileNetV3 Fine-tuned:** A transfer-learning model further fine-tuned for the target classification task.

The selected model is MobileNetV3 Fine-tuned, chosen using validation macro F1-score.

## Project Workflow

- Dataset loading and class-label mapping
- Dataset exploration and class-distribution analysis
- Image resizing, normalization and data augmentation
- Custom CNN development and training
- Transfer learning and fine-tuning
- Model checkpoints and early stopping
- Evaluation using classification metrics and confusion matrices
- Model comparison
- Saved model artifacts and project configuration
- Streamlit application development

## Technology Stack

- Python
- PyTorch and Torchvision
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- Pillow
- Streamlit
- Jupyter Notebook

## Project Structure

```text
BTIC/
├── Data/                 # Local dataset; excluded from Git
├── models/               # Trained model checkpoints
├── outputs/              # Evaluation outputs and configuration
├── app.py                # Streamlit application
├── BTIC_Training.ipynb   # Model training and evaluation notebook
├── requirements.txt      # Dependencies
├── .gitignore
└── README.md
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone <https://github.com/abel-721-bela/neuroscan-ai-brain-tumor-classification.git>
cd BTIC
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Using the Application

1. Launch the Streamlit application.
2. Upload a supported brain MRI image in JPG, JPEG, PNG, BMP or WebP format.
3. View the predicted category.
4. Review the model's softmax probability scores for all four classes.

## Model Evaluation

The project compares the custom CNN, MobileNetV3 Head and MobileNetV3 Fine-tuned models using validation and test evaluation.

Reported evaluation metrics should be taken directly from the completed notebook and include accuracy, precision, recall, macro F1-score and confusion matrices, where available.

## Limitations

- Predictions depend on the training dataset and model generalization.
- Confidence scores are not clinical certainty.
- The application has not been validated for medical diagnosis.
- Predictions must not replace assessment by a qualified medical professional.

## Future Improvements

- Improve generalization and investigate class-specific errors.
- Add more robust model monitoring and input validation.
- Evaluate performance on an independent dataset.
- Explore deployment with a hosted Streamlit application.

## Acknowledgement

Developed as a practical project in deep learning, medical-image classification, transfer learning and application deployment.
