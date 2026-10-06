# 🔬 DermaAI — AI-Powered Skin Lesion Classification & Educational Assistant

> **AI for Healthcare | Deep Learning + Generative AI**

DermaAI is an AI-powered skin-lesion classification and educational assistance platform that combines **EfficientNetB3-based deep learning** with **Google Gemini Generative AI**.

The system analyzes suitable dermoscopic skin-lesion images, classifies them into **seven lesion categories**, provides a confidence score, and uses Generative AI to convert the technical prediction into understandable educational information.

The application is designed for **healthcare awareness and educational support** and is **not intended to replace professional medical diagnosis**.

---

## 🏆 Hackathon Track

### AI For Healthcare

DermaAI addresses healthcare accessibility and health-awareness challenges by combining computer vision and Generative AI into an accessible web-based application.

---

# 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Our Solution](#-our-solution)
- [Why DermaAI](#-why-dermaai)
- [How It Works](#-how-it-works)
- [AI Architecture](#-ai-architecture)
- [Dataset](#-dataset)
- [Seven Skin Lesion Classes](#-seven-skin-lesion-classes)
- [Deep Learning Model](#-deep-learning-model)
- [Generative AI](#-generative-ai)
- [Key Features](#-key-features)
- [Input Image Requirements](#-input-image-requirements)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Environment Variables](#-environment-variables)
- [Running the Application](#-running-the-application)
- [Model Training](#-model-training)
- [Application Workflow](#-application-workflow)
- [Results and Evaluation](#-results-and-evaluation)
- [Limitations](#-limitations)
- [Medical Safety](#-medical-safety)
- [Real-World Impact](#-real-world-impact)
- [Future Scope](#-future-scope)
- [Live Demo](#-live-demo)
- [Repository](#-repository)
- [Team / Project](#-team--project)
- [Disclaimer](#-disclaimer)

---

# 🎯 Problem Statement

Skin lesions can have visually similar characteristics, making image-based classification challenging.

Access to dermatology specialists may also be limited for some users. Even when an AI model produces a prediction, a technical output such as a class label and probability may not be easy for a general user to understand.

Therefore, there is a need for an accessible AI-assisted system that can:

- Analyze suitable skin-lesion images.
- Classify the lesion into relevant categories.
- Provide a confidence score.
- Explain the prediction in understandable language.
- Support multilingual healthcare education.
- Encourage users to seek professional medical evaluation when appropriate.

---

# 💡 Our Solution

**DermaAI** combines two major AI components:

### 1. Deep Learning

A trained **EfficientNetB3-based transfer-learning model** performs seven-class skin-lesion classification.

### 2. Generative AI

**Google Gemini** receives the classification result and generates an educational explanation that is easier for users to understand.

Instead of stopping at:


Prediction → Melanocytic Nevi
Confidence → 87%

DermaAI provides:

Skin Image
     ↓
Deep Learning Classification
     ↓
Predicted Category + Confidence
     ↓
Google Gemini
     ↓
Educational Explanation
     ↓
Multilingual Interaction
     ↓
Follow-up Questions

This creates a user-focused healthcare AI experience rather than a classification-only system.
🚀 Why DermaAI?Many image-classification applications stop after displaying a predicted class. DermaAI extends this workflow by adding a Generative AI educational layer.Computer Vision $\rightarrow$ understands the imageEfficientNetB3 $\rightarrow$ performs 7-class classificationConfidence Score $\rightarrow$ communicates model certaintyGoogle Gemini $\rightarrow$ provides accessible educational explanationsMultilingual Support $\rightarrow$ improves accessibility (English, Kannada, Hindi)Streamlit $\rightarrow$ provides an intuitive, user-friendly interfaceDeep Learning provides the prediction; Generative AI makes the prediction easier to understand.🔄 How It WorksUSER 
  │
  ▼
Upload Skin Image ──► Image Preprocessing
  │
  ▼
┌─────────────────────────┐
│     EfficientNetB3      │
│   Transfer Learning     │
└────────────┬────────────┘
             │
             ▼
      Seven-Class Output
             │
      ┌──────┴──────┐
      ▼             ▼
Predicted Class  Confidence Score
      │             │
      └──────┬──────┘
             ▼
       Google Gemini
             │
             ▼
Educational Explanation
             │
      ┌──────┼──────┐
      ▼      ▼      ▼
   English Kannada Hindi
             │
             ▼
   Follow-up Questions & Q&A
🔬 Seven Skin Lesion Classes (HAM10000 Dataset)The project is trained on the HAM10000 dataset (10,015 dermoscopic images across 7 categories):CodeSkin Lesion CategoryNumber of ImagesnvMelanocytic Nevi6,705melMelanoma1,113bklBenign Keratosis-like Lesions1,099bccBasal Cell Carcinoma514akiecActinic Keratoses / Intraepithelial Carcinoma327vascVascular Lesions142dfDermatofibroma115TotalHAM10000 Dataset10,015🧠 AI ArchitectureLayer 1 — Deep Learning ClassifierArchitecture: EfficientNetB3 (using transfer learning, feature extraction, and fine-tuning).Pipeline: Input Image $\rightarrow$ Preprocessing $\rightarrow$ EfficientNetB3 $\rightarrow$ Feature Extraction $\rightarrow$ Softmax Probabilities $\rightarrow$ Predicted Class + Confidence Score.Saved Model: best_skin_model_phase2.kerasLayer 2 — Generative AI Assistant (Google Gemini)The CNN prediction and confidence score are passed as context to Google Gemini.Gemini acts as an educational assistant to explain the classification in clear terms and answer user follow-up questions in multiple languages.🛠️ Technology StackProgramming Language: PythonDeep Learning: TensorFlow / Keras (EfficientNetB3)Generative AI: Google Gemini APIWeb Application: StreamlitImage Processing: Pillow, NumPyEnvironment Management: python-dotenv📁 Project StructurePlaintextskin-disease/
│
├── app.py                     # Main Streamlit application
├── best_skin_model_phase2.keras # Trained EfficientNetB3 model
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── notebooks/                 
│   └── skin_lesion_training.ipynb # Model training & evaluation notebook
└── .gitignore
⚙️ Installation & Running LocallyClone the repository:Bashgit clone [https://github.com/Pavankm70/skin-disease.git](https://github.com/Pavankm70/skin-disease.git)
cd skin-disease
Create and activate a virtual environment:Bashpython -m venv venv
venv\Scripts\activate  # On Windows
Install dependencies:Bashpip install -r requirements.txt
Configure Environment Variables:Create a .env file in the root directory and add your Gemini API key:Code snippetGEMINI_API_KEY=YOUR_GEMINI_API_KEY
Run the Streamlit application:Bashstreamlit run app.py
⚠️ Medical Safety & LimitationsNot a Diagnosis: DermaAI is designed strictly for educational purposes, healthcare awareness, and demonstrating AI-assisted image classification. It is not a clinically validated diagnostic tool.Consult Professionals: Users should always consult a qualified dermatologist or healthcare professional for diagnosis, treatment, or medical concerns regarding any skin lesion.Distribution Limits: The model performs best on dermoscopic images similar to the training distribution. Out-of-distribution images, poor lighting, or heavy obstructions can reduce reliability.🌐 Links & Resources🚀 Streamlit Live Demo: View App💻 GitHub Repository: View Source Code🏆 Hackathon Track: AI For Healthcare

disease
