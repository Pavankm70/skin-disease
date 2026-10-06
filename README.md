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

```text
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

🚀 Why DermaAI?

Many image-classification applications stop after displaying a predicted class.

DermaAI extends the workflow by adding a Generative AI educational layer.

Our approach:

Computer Vision
→ understands the image

EfficientNetB3
→ performs classification

Confidence Score
→ communicates model confidence

Gemini
→ explains the prediction

Multilingual Support
→ improves accessibility

Streamlit
→ provides an easy-to-use interface

The key idea is:

Deep Learning provides the prediction; Generative AI makes the prediction easier to understand.

🔄 How It Works
                 USER
                   │
                   ▼
          Upload Skin Image
                   │
                   ▼
          Image Preprocessing
                   │
                   ▼
        ┌────────────────────┐
        │    EfficientNetB3  │
        │  Transfer Learning  │
        └──────────┬─────────┘
                   │
                   ▼
          Seven-Class Output
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
     Predicted Class    Confidence
          │                 │
          └────────┬────────┘
                   ▼
             Google Gemini
                   │
                   ▼
        Educational Explanation
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
     English    Kannada      Hindi
        │          │          │
        └──────────┼──────────┘
                   ▼
          Follow-up Questions
🧠 AI Architecture

The system contains two AI layers.

Layer 1 — Deep Learning Classifier

The uploaded image is processed and passed to a trained:

EfficientNetB3

The model uses transfer learning and fine-tuning to classify the image into seven skin-lesion categories.

Input Image
     ↓
Preprocessing
     ↓
EfficientNetB3
     ↓
Feature Extraction
     ↓
Classification Layer
     ↓
Softmax Probabilities
     ↓
Predicted Class
+
Confidence Score
Layer 2 — Generative AI Assistant

The CNN prediction is then provided as context to Google Gemini.

CNN Prediction
      +
Confidence
      ↓
Google Gemini
      ↓
Educational Explanation
      ↓
User Follow-up Questions

Gemini is used as an educational assistance layer.

It does not replace the trained image-classification model.

🗂️ Dataset

The project uses the:

HAM10000 Dataset

HAM10000 (Human Against Machine with 10000 training images) is a dermoscopic skin-lesion image dataset.

The dataset used in this project contains:

10,015 dermoscopic images
7 lesion categories
Significant class imbalance between categories

The dataset contains both common and less frequent lesion categories, making class balancing and appropriate preprocessing important during model development.

🔬 Seven Skin Lesion Classes

The trained model classifies images into the following seven categories:

Code	Skin Lesion Category
nv	Melanocytic Nevi
mel	Melanoma
bkl	Benign Keratosis-like Lesions
bcc	Basal Cell Carcinoma
akiec	Actinic Keratoses / Intraepithelial Carcinoma
vasc	Vascular Lesions
df	Dermatofibroma
Dataset Distribution
Class	Number of Images
Melanocytic Nevi	6,705
Melanoma	1,113
Benign Keratosis-like Lesions	1,099
Basal Cell Carcinoma	514
Actinic Keratoses / Intraepithelial Carcinoma	327
Vascular Lesions	142
Dermatofibroma	115
Total	10,015

The strong class imbalance makes techniques such as augmentation and class balancing important during training.

🧠 Deep Learning Model
EfficientNetB3

The project uses EfficientNetB3, not simply a generic CNN architecture.

The model is developed using:

Transfer learning
Feature extraction
Fine-tuning
Data augmentation
Class balancing
Seven-class classification
Why EfficientNetB3?

EfficientNet architectures provide a strong balance between:

Model capacity
Computational efficiency
Image feature extraction
Classification performance

A pretrained EfficientNetB3 backbone allows the model to start with learned visual representations and then adapt them to the skin-lesion classification task.

🔧 Model Training Pipeline
HAM10000 Dataset
       ↓
Data Cleaning / Preparation
       ↓
Image Preprocessing
       ↓
Data Augmentation
       ↓
Class Balancing
       ↓
EfficientNetB3
       ↓
Transfer Learning
       ↓
Fine-Tuning
       ↓
Seven-Class Classification
       ↓
Model Evaluation
       ↓
Saved Keras Model

The trained model is saved as:

best_skin_model_phase2.keras
📷 Input Image Requirements

The model is trained using dermoscopic skin-lesion images.

For the best possible prediction, users should provide:

Recommended
A clear image of a skin lesion.
The lesion should be clearly visible.
Good image quality.
Minimal obstruction.
A single primary lesion should be visible.
The image should be visually similar to the type of dermoscopic images represented in the training data.
Avoid
Extremely blurry images.
Images where the lesion is not visible.
Unrelated photographs.
Images containing multiple unrelated objects.
Very dark or severely overexposed images.
Images significantly different from the training data.
Important

The model is trained on a specific dataset and image distribution.

Therefore:

Images outside the training distribution may produce unreliable predictions even when the model provides a confidence score.

This is an important limitation of machine-learning-based medical image classification.

✨ Key Features
🧠 AI-Based Classification

Classifies suitable dermoscopic skin-lesion images into seven categories.

📊 Confidence Score

Displays the model's confidence associated with the predicted category.

📷 Image Upload

Users can upload a skin-lesion image directly through the Streamlit application.

✨ Gemini Educational Explanation

Google Gemini generates an understandable educational explanation based on the predicted category.

💬 Interactive Q&A

Users can ask follow-up questions about the predicted lesion category.

🌐 Multilingual Support

Educational interaction is designed to support:

English
Kannada
Hindi
⚠️ Medical Safety

The application clearly communicates that AI output is not a medical diagnosis.

🤖 Generative AI — Google Gemini

Google Gemini is integrated into the application as an educational assistant.

The workflow is:

CNN Prediction
      ↓
Predicted Lesion Category
      +
Confidence
      ↓
Gemini Prompt
      ↓
Educational Explanation
      ↓
User

Gemini helps explain the classification in a more accessible format.

It can also support follow-up questions from the user.

🖥️ Application

The web application is built using:

Streamlit

The application provides:

Image upload
Prediction display
Confidence information
Gemini explanation
Multilingual interaction
Follow-up Q&A
Medical disclaimer

The application is designed to provide a simple interface between the user and the AI pipeline.

🛠️ Technology Stack
Component	Technology
Programming Language	Python
Deep Learning	TensorFlow / Keras
Model Architecture	EfficientNetB3
Training Method	Transfer Learning + Fine-Tuning
Dataset	HAM10000
Numerical Processing	NumPy
Image Processing	Pillow
Generative AI	Google Gemini API
Web Application	Streamlit
Environment Variables	python-dotenv
Model Format	Keras .keras
Source Control	Git / GitHub
Deployment	Streamlit Community Cloud
📁 Project Structure
skin-disease/
│
├── app.py
│
├── best_skin_model_phase2.keras
│
├── requirements.txt
│
├── README.md
│
├── notebooks/
│   └── skin_lesion_training.ipynb
│
└── .gitignore
Important Files
app.py

Main Streamlit application.

Responsible for:

Loading the trained model
Processing uploaded images
Generating predictions
Displaying confidence
Calling Gemini
Providing the user interface
best_skin_model_phase2.keras

Trained EfficientNetB3-based skin-lesion classification model.

notebooks/skin_lesion_training.ipynb

Training notebook containing the model-development workflow.

It allows reviewers and judges to inspect the training methodology.

requirements.txt

Python dependencies required to run the application.

⚙️ Installation

Clone the repository:

git clone https://github.com/Pavankm70/skin-disease.git

Move into the project directory:

cd skin-disease

Create a virtual environment:

python -m venv venv

Activate the virtual environment.

Windows
venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
🔐 Environment Variables

The Gemini API key should never be committed to GitHub.

For local development, create:

.env

Add:

GEMINI_API_KEY=YOUR_GEMINI_API_KEY

The .env file should be included in .gitignore.

Never do this:
GEMINI_API_KEY = "actual-secret-key"

Do not publish API keys in source code, notebooks, screenshots, or GitHub.

▶️ Run the Application Locally

After installing the dependencies and configuring the Gemini API key:

streamlit run app.py

The application will open in your browser.

🧪 Model Training

The training notebook is available at:

notebooks/skin_lesion_training.ipynb

The notebook documents the development of the skin-lesion classifier, including:

Dataset loading
Dataset preparation
Image preprocessing
Data augmentation
Class balancing
EfficientNetB3 transfer learning
Fine-tuning
Model training
Validation
Evaluation
Model saving

The resulting trained model is integrated into the Streamlit application.

🔄 End-to-End Application Workflow
                ┌─────────────────┐
                │   User uploads  │
                │   skin image    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Image           │
                │ preprocessing   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ EfficientNetB3  │
                │ classification  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ 7-class         │
                │ prediction      │
                └────────┬────────┘
                         │
                ┌────────┴────────┐
                ▼                 ▼
           Class label       Confidence
                │                 │
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ Google Gemini   │
                │ educational AI  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Explanation +   │
                │ Follow-up Q&A   │
                └─────────────────┘
📊 Results and Evaluation

The model should be evaluated using multiple classification metrics rather than accuracy alone.

Recommended evaluation metrics include:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
Per-class performance
Model Performance

Add the exact values from the training notebook here. Do not use estimated or manually created values.

Example format:

Metric	Score
Accuracy	XX.XX%
Precision	XX.XX%
Recall	XX.XX%
F1-Score	XX.XX%
Confusion Matrix

The confusion matrix from the training notebook can be added here to demonstrate the performance of each of the seven classes.

⚠️ Limitations

DermaAI is an AI-assisted prototype and has important limitations.

1. Dataset Dependence

The model learns patterns from the HAM10000 dataset. Performance may decrease on images that differ substantially from the training distribution.

2. Class Imbalance

The dataset contains substantially different numbers of images across the seven categories.

Less represented classes may be more difficult for the model.

3. Visually Similar Lesions

Some skin-lesion categories can have visually similar characteristics, which may lead to misclassification.

4. Image Quality

Blurry, poorly framed, heavily obstructed, or unusual images can reduce prediction reliability.

5. Confidence Is Not Diagnosis

A high model confidence does not mean that the prediction is medically confirmed.

6. Clinical Validation

The project is a research/educational prototype and has not been presented as a clinically validated diagnostic system.

🏥 Medical Safety

DermaAI is designed for:

Educational purposes
Healthcare awareness
Demonstrating AI-assisted image classification
Helping users understand AI-generated information

It is **not designed to:

Diagnose a medical condition
Replace a dermatologist
Recommend treatment
Prescribe medication
Make clinical decisions

Users should consult a qualified healthcare professional for diagnosis, treatment, or concerns about an actual skin lesion.

🌍 Real-World Impact

DermaAI demonstrates how AI can be used to improve accessibility to healthcare information.

Potential applications include:

Healthcare Education

Users can learn about different categories of skin lesions.

Patient Awareness

The system can help users understand why professional medical evaluation may be important.

Multilingual Accessibility

English, Kannada, and Hindi interaction can make AI-generated educational information accessible to a wider group of users.

Telehealth Support

With appropriate clinical validation and integration, similar systems could potentially support telemedicine and remote healthcare workflows.

AI-Assisted Healthcare Workflows

The architecture demonstrates how predictive AI and Generative AI can work together in healthcare applications.

🚀 Future Scope

Future development can include:

Larger and more diverse datasets
External validation using independent datasets
Improved class balancing
Better confidence calibration
Explainable AI techniques such as Grad-CAM
Improved out-of-distribution image detection
Additional Indian languages
Mobile application support
Doctor-facing dashboards
Telemedicine integration
Secure patient history management
Privacy-preserving image processing
Clinical validation with medical professionals

The long-term goal is to develop a more robust and responsibly validated AI-assisted dermatology support system.

🎯 Innovation Summary

The primary innovation of DermaAI is the combination of:

Computer Vision
       +
Deep Learning
       +
EfficientNetB3
       +
Generative AI
       +
Multilingual Interaction
       +
Web Deployment

Instead of providing only a machine-learning classification label, the system adds a Generative AI layer that makes the result easier to understand.

In one sentence:

DermaAI transforms a deep-learning skin-lesion prediction into accessible, multilingual educational healthcare information through Generative AI.

🌐 Live Demo

🚀 Streamlit Demo:

https://skin-disease-8njbeyvqstmj8nkckf8gbr.streamlit.app/

💻 Source Code

📂 GitHub Repository:

https://github.com/Pavankm70/skin-disease

📓 Training Notebook

The model-development notebook is available in:

notebooks/skin_lesion_training.ipynb

The notebook is included so reviewers can inspect the model-training pipeline and understand how the trained EfficientNetB3 model was developed.

🏆 Hackathon Relevance
Challenge Track

AI For Healthcare

Healthcare Problem

Limited accessibility to understandable skin-lesion information and the complexity of interpreting technical AI predictions.

Proposed Solution

An AI-assisted platform combining deep-learning image classification with Generative AI educational assistance.

AI Technologies
EfficientNetB3
TensorFlow / Keras
Transfer Learning
Fine-Tuning
Google Gemini
Key Innovation

Combining predictive computer vision with Generative AI to provide understandable, multilingual healthcare education.

Real-World Potential

The architecture can potentially be extended toward healthcare education, telehealth support, and AI-assisted dermatology workflows after appropriate validation.

👥 Team / Project

Project: DermaAI
Track: AI For Healthcare
Repository: Pavankm70/skin-disease
