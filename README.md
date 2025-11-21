# FakeBuster - Advanced Deepfake Detection System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-%23000.svg?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**FakeBuster** is a robust deepfake detection framework capable of identifying hyper-realistic synthetic media. It employs a **Dual-Branch EfficientNet architecture** that fuses Spatial (RGB) and Frequency (FFT) domain features to detect manipulation artifacts that are often invisible to the naked eye.

## 🚀 Live Demo
Try the deployed model on Hugging Face Spaces:
**[🔗 Launch FakeBuster Space](https://huggingface.co/spaces/shreyankbr/FakeBuster)**

---

## 🧠 Architecture Overview

I designed the system using a "DualBranchEfficientNet" architecture to capture complementary evidence of forgery:

1. **Spatial Branch (RGB):** Uses **EfficientNet-B4** (pretrained on ImageNet) to detect semantic inconsistencies and blending artifacts in standard video frames.
2. **Frequency Branch (FFT):** Uses **EfficientNet-B0** on Fast Fourier Transform (FFT) representations to identify spectral irregularities and abnormal high-frequency noise patterns.

The features from both branches are aggregated via **Temporal Pooling** and fused into a 1024-dimensional vector before final classification.

```mermaid
graph TD
    Input[Input Video] --> S[Frame Sampling]
    S --> B{Dual-Branch Processing}
    B -->|Branch 1: Spatial| C[RGB Stream<br/>EfficientNet-B4]
    B -->|Branch 2: Frequency| D[FFT Stream<br/>EfficientNet-B0]
    C --> E[Feature Fusion<br/>Concatenate 1024D]
    D --> E
    E --> F[Temporal Pooling]
    F --> G[Classifier<br/>Sigmoid Activation]
    G --> Output[Real vs Fake Probability]
````

## 🔬 R\&D: Optimization & Generalization

My research for this project involved extensive experimentation to balance computational efficiency with detection accuracy.

### 1\. Frame Rate Optimization Strategy

Through rigorous testing of different frame counts (8, 12, 14, 16, 32), I engineered a split-sampling strategy:

  * **Training (12 Frames):** I trained on fewer frames to prevent overfitting and reduce noise sensitivity.
  * **Inference (18 Frames):** I increased sampling during deployment to maximize temporal coverage.
  * **Result:** This configuration yielded the highest Validation AUC (**0.9678**) compared to uniform sampling methods.

### 2\. Cross-Dataset Generalization

A major challenge in deepfake detection is handling "unseen" manipulation types. I tested FakeBuster against datasets and manipulation techniques **not** included in the training phase.

  * **Cross-Dataset AUC:** 0.8155
  * **Cross-Dataset F1-Score:** 0.8387
  * **Conclusion:** The model demonstrates strong robustness, proving it learns fundamental forgery artifacts rather than just memorizing specific dataset patterns.

-----

## ✨ Key Features

  - **🎭 Dual-Domain Analysis:** Simultaneously analyzes pixel integrity (RGB) and spectral consistency (FFT) for superior accuracy.
  - **📊 Visual Explainability:** Integrated **Grad-CAM** (Gradient-weighted Class Activation Mapping) provides heatmaps to interpret model decisions.
  - **🎯 Optimized Sampling:** Implements smart temporal sampling (12 frames for training, 18 for inference) to balance speed and accuracy.
  - **🔒 Secure Platform:** Features Firebase authentication and session management for secure user history.
  - **☁️ Scalable Deployment:** Containerized via Docker and capable of running on cloud environments or local GPUs.

-----

## 🛠️ Technology Stack

| Component | Technologies |
|-----------|--------------|
| **Deep Learning** | PyTorch, Timm (Image Models), Torch.cuda.amp (Mixed Precision) |
| **Computer Vision** | OpenCV, Pillow, Albumentations |
| **Backend** | Python 3.11+, Flask, Gunicorn |
| **Frontend** | HTML5, CSS3, JavaScript (Fetch API), Chart.js |
| **Infrastructure** | Docker, Firebase (Auth/DB), Hugging Face Spaces |

-----

## 📊 Performance Metrics

I trained and validated the model on the preprocessed version of **FaceForensics++** dataset (Real, DeepFakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures). (https://www.kaggle.com/datasets/adham7elmy/faceforencispp-extracted-frames)

### Validation Results

Achieved state-of-the-art performance on the validation set using the champion configuration (12-frame training / 18-frame inference):

| Metric | Score |
|--------|-------|
| **AUC** | **0.9678** |
| **F1-Score** | **0.9672** |
| **Accuracy** | **94.58%** |
| **Precision** | **97.56%** |
| **Recall** | **92.86%** |

-----

## ⚙️ Installation & Setup

### Prerequisites

  * Python 3.11+
  * Firebase Account
  * CUDA-enabled GPU (Recommended for training)

### 1\. Clone the Repository

```bash
git clone [https://github.com/shreyankbr/FakeBuster-DeepFake-Detection-System.git](https://github.com/shreyankbr/FakeBuster-DeepFake-Detection-System.git)
cd FakeBuster-DeepFake-Detection-System
```

### 2\. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3\. Configuration

1.  Set up a **Firebase Project** and enable Email/Password Authentication.
2.  Create a **Firestore Database**.
3.  Update `static/js/firebase-config.js` with your credentials.

### 4\. Run the Application

```bash
python app.py
```

Access the application at `http://localhost:5000`.

-----

## 🔮 Future Roadmap

  * **Edge Computing:** Optimization for mobile and embedded devices (e.g., NPU integration) to enable on-device detection.
  * **Blockchain Integration:** Implementing distributed ledger technology (DLT) for immutable content provenance and authenticity tracking.
  * **Real-Time Stream Analysis:** Enhanced optimization for processing live video feeds with minimal latency.

-----

## Citation

A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies and M. Nießner, "FaceForensics++: Learning to Detect Manipulated Facial Images," in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2019.

-----

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
