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

The system utilizes a "DualBranchEfficientNet" design to capture complementary evidence of forgery:

1.  [cite_start]**Spatial Branch (RGB):** Uses **EfficientNet-B4** (pretrained on ImageNet) to detect semantic inconsistencies and blending artifacts in standard video frames[cite: 97].
2.  [cite_start]**Frequency Branch (FFT):** Uses **EfficientNet-B0** on Fast Fourier Transform (FFT) representations to identify spectral irregularities and checking for abnormal high-frequency noise patterns[cite: 98].

[cite_start]These features are aggregated via **Temporal Pooling** and fused into a 1024-dimensional vector before final classification[cite: 112].

```mermaid
graph TD
    Input[Input Video] --> S[Frame Sampling<br/>18 frames/video]
    S --> B{Dual-Branch Processing}
    B -->|Branch 1: Spatial| C[RGB Stream<br/>EfficientNet-B4]
    B -->|Branch 2: Frequency| D[FFT Stream<br/>EfficientNet-B0]
    C --> E[Feature Fusion<br/>Concatenate 1024D]
    D --> E
    E --> F[Temporal Pooling]
    F --> G[Classifier<br/>Sigmoid Activation]
    G --> Output[Real vs Fake Probability]
````

## 📸 Screenshots

*The Analysis Dashboard showing a detected deepfake with high confidence.*

[cite\_start]*GradCAM heatmaps visualizing the specific facial regions influencing the model's decision[cite: 161].*

-----

## ✨ Key Features

  - [cite\_start]**🎭 Dual-Domain Analysis:** Simultaneously analyzes pixel integrity (RGB) and spectral consistency (FFT) for superior accuracy[cite: 66].
  - [cite\_start]**📊 Visual Explainability:** Integrated **Grad-CAM** (Gradient-weighted Class Activation Mapping) provides heatmaps to interpret model decisions[cite: 161].
  - [cite\_start]**🎯 Optimized Sampling:** Implements smart temporal sampling (12 frames for training, 18 for inference) to balance speed and accuracy[cite: 144].
  - [cite\_start]**🔒 Secure Platform:** Features Firebase authentication and session management for secure user history[cite: 162].
  - [cite\_start]**☁️ Scalable Deployment:** Containerized via Docker and capable of running on cloud environments or local GPUs[cite: 163].

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

[cite\_start]The model was trained and validated on the **FaceForensics++** dataset (Real, DeepFakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures)[cite: 78].

### Validation Results

[cite\_start]Achieved state-of-the-art performance on the validation set using the champion configuration (12-frame training / 18-frame inference)[cite: 144, 218]:

| Metric | Score |
|--------|-------|
| **AUC** | **0.9678** |
| **F1-Score** | **0.9672** |
| **Accuracy** | **94.58%** |
| **Precision** | **97.56%** |
| **Recall** | **92.86%** |

### Robustness

[cite\_start]The model demonstrates strong generalization with a **Cross-Dataset AUC of 0.8155**, proving efficacy against manipulation types not seen during training[cite: 155].

-----

## ⚙️ Installation & Setup

### Prerequisites

  * Python 3.11+
  * Firebase Account
  * CUDA-enabled GPU (Recommended for training)

### 1\. Clone the Repository

```bash
git clone [https://github.com/shreyankbr/fakebuster.git](https://github.com/shreyankbr/fakebuster.git)
cd fakebuster
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

  * [cite\_start]**Edge Computing:** Optimization for mobile and embedded devices (e.g., NPU integration) to enable on-device detection[cite: 252].
  * [cite\_start]**Blockchain Integration:** Implementing distributed ledger technology (DLT) for immutable content provenance and authenticity tracking[cite: 256].
  * [cite\_start]**Real-Time Stream Analysis:** Enhanced optimization for processing live video feeds with minimal latency[cite: 254].

-----

## 📄 License

Distributed under the MIT License. See `LICENSE` for more details.
