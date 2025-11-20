# FakeBuster - Advanced Deepfake Detection System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-%23000.svg?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**FakeBuster** is a web-based application leveraging deep learning to detect deepfake videos and images with high accuracy. Utilizing a Dual-Branch architecture, the system analyzes both RGB data and Frequency domain features (FFT) to spot manipulation artifacts invisible to the naked eye.

## 🚀 Live Demo
Try the model implementation on Hugging Face:
**[🔗 Launch FakeBuster Space](https://huggingface.co/spaces/shreyankbr/FakeBuster)**

---

## 🧠 Architecture Overview

```mermaid
graph TD
    Input[Input Video/Image] --> A[Frame Extraction]
    A --> B{Dual-Branch Processing}
    B -->|Branch 1| C[RGB Spatial Analysis<br/>EfficientNet-B4]
    B -->|Branch 2| D[Frequency Domain Analysis<br/>FFT + EfficientNet-B0]
    C --> E[Feature Concatenation]
    D --> E
    E --> F[Fully Connected Layers]
    F --> Output[Probability Score<br/>Real vs Fake]
````

## 📸 Screenshots

*The Analysis Dashboard showing a detected deepfake with 99.8% confidence.*

*GradCAM heatmap highlighting the manipulated facial regions.*

-----

## ✨ Key Features

  - **🎭 Multi-Format Detection:** Supports video files (MP4, AVI, MOV) and frame-based image analysis.
  - **🔍 Dual-Branch Architecture:** Combines **EfficientNet-B4** (Spatial/RGB) and **EfficientNet-B0** (Frequency/FFT) for robust detection.
  - **📊 Visual Explainability:** Generates GradCAM heatmaps to visualize exactly *where* the model detects manipulation.
  - **👤 User System:** Secure Firebase authentication with persistent analysis history.
  - **🎯 High Precision:** Validated AUC of **0.9790** on FaceForensics++.
  - **📱 Responsive Design:** Seamless experience across desktop and mobile devices.

-----

## 🛠️ Technology Stack

| Component | Technologies |
|-----------|--------------|
| **Frontend** | HTML5, CSS3, JavaScript, Chart.js |
| **Backend** | Python 3.11+, Flask |
| **Deep Learning** | PyTorch, Timm, OpenCV |
| **Architecture** | EfficientNet-B4 (RGB) + EfficientNet-B0 (Frequency) |
| **Database/Auth** | Firebase Firestore, Firebase Auth |

-----

## 📊 Model Performance & R\&D

### Dataset

The model was trained on a preprocessed version of the **FaceForensics++** dataset (5,995 videos):

  * **Real:** 999 YouTube videos
  * **Fake:** 4,996 videos (DeepFakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures)

### Frame Rate Optimization Strategy

We implemented a smart sampling strategy to maximize temporal information while maintaining inference speed.

| Metric | Training Configuration | Inference Configuration |
|--------|------------------------|-------------------------|
| **Frames** | 12 frames/video | 18 frames/video |
| **Sampling** | Random Temporal | Padding + Uniform |
| **Threshold** | N/A | **0.45** (Probability \> 0.45 = Fake) |

### Validation Metrics

  - **Validation AUC:** 0.9790 (Best Epoch)
  - **Accuracy:** 94.42%
  - **F1 Score:** 0.9660

> **Note:** For a quick start without training, download my pre-trained model `12f/5.pth` from the Hugging Face repository linked above.

-----

## ⚙️ Installation & Setup

### Prerequisites

  * Python 3.11+
  * Firebase Account
  * Visual Studio Code (Recommended)

### 1\. Clone the Repository

```bash
git clone [https://github.com/yourusername/fakebuster.git](https://github.com/yourusername/fakebuster.git)
cd fakebuster
```

### 2\. Environment Setup

Create and activate a virtual environment:

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3\. Firebase Configuration

1.  Create a project in the [Firebase Console](https://console.firebase.google.com/).
2.  Enable **Email/Password** in the Authentication tab.
3.  Create a Firestore Database and apply the security rules found in `Firestore rules.txt`.
4.  Update `static/js/firebase-config.js`:

<!-- end list -->

```javascript
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_AUTH_DOMAIN",
  databaseURL: "YOUR_DATABASE_URL",
  projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_STORAGE_BUCKET",
  messagingSenderId: "YOUR_SENDER_ID",
  appId: "YOUR_APP_ID"
};
```

### 4\. Run the Application

```bash
python app.py
```

Access the application at `http://localhost:5000`.

-----

## ⚠️ Limitations

This project is intended for educational and research purposes.

  * **Generalization:** Accuracy may decrease on manipulation techniques not present in FaceForensics++.
  * **Lighting:** Extreme lighting conditions or heavy compression can affect prediction confidence.
  * **Processing Time:** Analysis speed depends on client hardware (GPU recommended).

## 🔮 Future Improvements

  * [ ] Integration of Transformer-based models (ViT).
  * [ ] Support for GAN-based deepfake detection.
  * [ ] Real-time webcam stream analysis.
  * [ ] Dark Mode UI implementation.

-----

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
