# FakeBuster - Advanced Deepfake Detection System

FakeBuster is a web-based application that uses deep learning to detect deepfake videos and images with high accuracy. The system provides visual explanations of its predictions using GradCAM heatmaps and maintains user analysis history.

## Key Features

- 🎭 **Multi-Format Detection**: Supports both video files and image frames analysis
- 🔍 **Dual-Branch Architecture**: Combines RGB and FFT analysis for enhanced accuracy
- 📊 **Visual Explanations**: GradCAM heatmaps highlight manipulated regions
- 👤 **User Accounts**: Secure authentication and analysis history tracking
- 🎯 **Confidence Scoring**: Detailed probability scores for real/fake classification
- 📱 **Responsive Design**: Works seamlessly across desktop and mobile devices

## Technology Stack

### Frontend
- HTML, CSS, JavaScript
- Firebase Authentication
- Firebase Firestore (NoSQL database)
- Chart.js (Visualizations)

### Backend & AI
- Python 3.11+
- Flask
- PyTorch (Deep Learning)
- EfficientNet-B4 & B0 (Dual-Branch CNN Architecture)
- OpenCV (Video Processing)
- Timm (PyTorch Image Models)

## Dataset

The model was trained on the a preprocessed version of FaceForensics++ dataset containing 5,995 videos across real and fake categories:

- **999 real videos** from YouTube
- **4,996 fake videos** across 5 manipulation types:
  1. DeepFakes
  2. Face2Face
  3. FaceShifter
  4. FaceSwap
  5. NeuralTextures

The dataset I used is available on Kaggle (https://www.kaggle.com/datasets/adham7elmy/faceforencispp-extracted-frames/data). Any dataset from Kaggle can be taken, but be sure to make the necessary name changes in all of the files.

For those who only want the project, I have uploaded my trained model (12f/5.pth) in huggingface, you may use that model and skip preprocessing and model training.

## R&D: Frame Rate Optimization

**Training Configuration:**
- **12 frames per video** during training
- **Balanced sampling**: Temporal coverage with random sampling during training, uniform during validation
- **Frame extraction**: Smart sampling from videos to maximize temporal information

**Inference Configuration:**
- **18 frames per analysis** during inference
- **Optimized processing**: Frame extraction with padding for consistent input size
- **Threshold**: **0.45** probability threshold for fake/real classification

**Performance Metrics:**
- Validation AUC: **0.9790** (Best epoch)
- Accuracy: **94.42%**
- F1 Score: **0.9660**

## Installation

### Prerequisites
- Python 3.11+
- Firebase account
- Visual Studio Code Preferred

### Local Setup

1. Download the repository and keep the structure the same

2. Virtual Environment:

    - Set up a virtual environment in the repo folder 

    - Run pip install -r requirements.txt in cmd after changing into the repo folder

3. **Firebase Configuration**:

    - Create a new Firebase project
     
    - Enable Email/Password authentication in Firebase Console
     
    - Update firebaseConfig in static/js/firebase-config.js
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

    - Apply the rules given in Firestore rules.txt to firestore rules

4. Dataset Preprocessing:
  
    - Download FaceForensics++ dataset from Kaggle and make necessary changes in all files regarding names

    - Run the notebook for data preprocessing and model training

5. Model training:

    - Run the training notebook and train the model and select the best ones from the many epochs

6. Run:

    - Run app.py preferable in visual studio code in dedicated terminal and open the localhost link with the 5000 port
   
## Usage

- **Trial Mode**: Use without account (no history saving)
- **Registration/Login**: Create account for analysis history
- **Video Analysis**: Upload MP4, AVI, MOV files (max 100MB)
- **Frame Analysis**: Upload multiple image frames from videos
- **Analyze**: Click "Analyze for Deepfakes" button
- **View Results**: See predictions with confidence scores and heatmaps
- **History**: Track previous analyses in dashboard

## Limitations

⚠️ Important: This is a demonstration project only. The system has several limitations:
- Accuracy may vary with new manipulation techniques not in training data
- Performance depends on video quality and face visibility
- Should not be used for critical security applications without further validation
- Analysis may take longer depending on file processing

## Areas of Improvement

- Feel free to use any better CNN architectures or transformer models
- Add support for newer deepfake techniques like GAN-based manipulations
- Implement real-time video stream analysis
- Add multi-language support
- Improve model accuracy with more data
- Dark theme

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
