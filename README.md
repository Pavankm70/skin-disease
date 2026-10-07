https://www.google.com/imgres?q=ham10000%20dermoscopic%20images&imgurl=https%3A%2F%2Fmedia.springernature.com%2Ffull%2Fspringer-static%2Fimage%2Fart%253A10.1038%252Fs41598-022-22644-9%2FMediaObjects%2F41598_2022_22644_Fig1_HTML.png&imgrefurl=https%3A%2F%2Fwww.nature.com%2Farticles%2Fs41598-022-22644-9&docid=mu7tlT0GIEJrPM&tbnid=VV0MeyKGSq4wDM&vet=12ahUKEwjpyLKhmaeXAxXWmuEIHcQWK1MQnPAOegQIQBAA..i&w=1653&h=1404&hcb=2&ved=2ahUKEwjpyLKhmaeXAxXWmuEIHcQWK1MQnPAOegQIQBAA
7 classes of ham10000 dermoscopic dataset 


# 🔬 DermaAI — AI-Powered Skin Lesion Classification & Educational Assistant

> **AI for Healthcare | Deep Learning + Generative AI**

DermaAI is an AI-powered skin-lesion classification and educational assistance platform that combines **Deep Learning** and **Generative AI**.

The system uses an **EfficientNetB3-based deep learning model** trained on the **HAM10000 dataset** to classify suitable dermoscopic skin-lesion images into **seven categories**. After classification, the prediction and confidence score are provided as context to **Google Gemini**, which generates an understandable educational explanation.

The application supports **English, Kannada, and Hindi** and provides an interactive web interface using **Streamlit**.

> ⚠️ **Important:** DermaAI is an educational and research prototype. It is **not a medical diagnostic system** and should not be used as a substitute for professional medical advice.

---

## 🏆 Hackathon Track

### AI for Healthcare

DermaAI focuses on applying Artificial Intelligence to healthcare education and image-based skin-lesion analysis.

The project combines:

- Computer Vision
- Image Processing
- Deep Learning
- Transfer Learning
- Model Prediction
- Confidence Analysis
- Generative AI
- Multilingual Interaction
- Interactive Web Application

---

## 🎯 Problem Statement

Skin lesions can have visually similar characteristics, making image-based classification challenging.

For many users, even when an AI model produces a prediction, a technical output such as a class label and probability may not be easy to understand.

Therefore, DermaAI aims to create an accessible AI-assisted system that can:

- Analyze suitable dermoscopic skin-lesion images
- Classify images into seven lesion categories
- Provide a model confidence score
- Convert technical predictions into understandable educational information
- Support multiple languages
- Allow users to ask follow-up questions
- Encourage users to seek professional medical evaluation when required

---

## 💡 Our Solution

DermaAI uses two major AI components.

### 1. Deep Learning Classifier

An **EfficientNetB3** model using transfer learning and fine-tuning performs the seven-class skin-lesion classification.

### 2. Generative AI Assistant

**Google Gemini** receives the model prediction and confidence information and provides an educational explanation in a user-friendly format.

### Overall Concept

```text
Skin Image
    ↓
Image Processing
    ↓
EfficientNetB3
    ↓
Model Prediction
    ↓
Predicted Class + Confidence
    ↓
Google Gemini
    ↓
Educational Explanation
    ↓
English / Kannada / Hindi
    ↓
Follow-up Questions
```

Deep Learning provides the prediction, while Generative AI helps explain the prediction.

---

## 🚀 Why DermaAI?

Many image-classification applications stop after displaying a predicted class.

DermaAI extends this workflow by adding a Generative AI educational layer.

| Component | Purpose |
|---|---|
| 🖼️ Computer Vision | Processes the uploaded skin-lesion image |
| 🧠 EfficientNetB3 | Performs seven-class image classification |
| 📊 Confidence Score | Shows the model's predicted probability |
| 🤖 Google Gemini | Converts the technical prediction into understandable educational information |
| 🌐 Multilingual Support | Supports English, Kannada, and Hindi |
| 🖥️ Streamlit | Provides an interactive web interface |

---

## 🔄 How It Works

```text
                    USER
                      │
                      ▼
              Upload Skin Image
                      │
                      ▼
               Image Processing
                      │
                      ▼
              ┌─────────────────┐
              │  EfficientNetB3 │
              │ Deep Learning   │
              │ Classification  │
              └────────┬────────┘
                       │
                       ▼
                Prediction Output
                       │
               ┌───────┴────────┐
               ▼                ▼
        Predicted Class    Confidence Score
               │                │
               └───────┬────────┘
                       ▼
                  Google Gemini
                       │
                       ▼
             Educational Explanation
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
           English  Kannada    Hindi
                       │
                       ▼
                 Follow-up Q&A
```

---

## 🗂️ Dataset

The project uses the **HAM10000 (Human Against Machine with 10000 training images)** dataset.

The dataset contains **10,015 dermoscopic images** belonging to seven skin-lesion categories.

The dataset is highly imbalanced, with some classes containing significantly fewer images than others. Therefore, class distribution and balancing are important parts of the model-training process.

---

## 🔬 Seven Skin Lesion Classes

The model classifies images into the following seven categories:

| Code | Skin Lesion Category | Images |
|---|---|---:|
| `nv` | Melanocytic Nevi | 6,705 |
| `mel` | Melanoma | 1,113 |
| `bkl` | Benign Keratosis-like Lesions | 1,099 |
| `bcc` | Basal Cell Carcinoma | 514 |
| `akiec` | Actinic Keratoses / Intraepithelial Carcinoma | 327 |
| `vasc` | Vascular Lesions | 142 |
| `df` | Dermatofibroma | 115 |
| **Total** | | **10,015** |

---

## 📊 Data Analytics

Before training the model, the dataset can be analyzed to understand its characteristics.

Important areas include:

- Class distribution
- Number of images per class
- Class imbalance
- Image dimensions
- Image quality
- Training and validation distribution

Understanding class distribution is particularly important because the HAM10000 dataset contains significantly more images for some classes than others.

---

## 🖼️ Image Processing

The uploaded image goes through preprocessing before being provided to the trained model.

```text
Raw Image
    ↓
Image Loading
    ↓
Image Conversion
    ↓
Image Resizing
    ↓
Numerical Representation
    ↓
Model-Compatible Preprocessing
    ↓
EfficientNetB3
```

Image preprocessing helps ensure that the input is in a format compatible with the trained neural network.

---

## 🔄 Data Augmentation

During model training, suitable image augmentation techniques can be used to improve the model's ability to generalize to variations in image appearance.

Examples include:

- Rotation
- Horizontal flipping
- Vertical flipping
- Zooming
- Other suitable transformations

Augmentation helps create useful variations of training images while preserving important visual characteristics required for classification.

---

## ⚖️ Class Imbalance

HAM10000 contains an uneven number of images across the seven classes.

```text
Melanocytic Nevi        → 6705
Melanoma                → 1113
Benign Keratosis        → 1099
Basal Cell Carcinoma    → 514
Actinic Keratoses       → 327
Vascular Lesions        → 142
Dermatofibroma          → 115
```

Because of this imbalance, model training needs to consider the distribution of classes so that minority classes are not ignored by the learning process.

---

# 🧠 Deep Learning Model

## EfficientNetB3

The project uses **EfficientNetB3** as the primary image-classification architecture.

The model uses **transfer learning followed by fine-tuning**.

### Model Pipeline

```text
Input Image
    ↓
Preprocessing
    ↓
EfficientNetB3
    ↓
Feature Extraction
    ↓
Fine-Tuning
    ↓
Classification Layer
    ↓
Softmax Probabilities
    ↓
Predicted Class
    ↓
Confidence Score
```

### Saved Model

The trained model is saved as:

```text
best_skin_model_phase2.keras
```

This trained model is loaded by the Streamlit application for inference.

---

## 🏋️ Model Training

The overall training process follows this workflow:

```text
HAM10000 Dataset
       ↓
Data Analysis
       ↓
Image Preprocessing
       ↓
Data Augmentation
       ↓
Class Balancing
       ↓
Train / Validation Split
       ↓
EfficientNetB3
       ↓
Transfer Learning
       ↓
Fine-Tuning
       ↓
Model Evaluation
       ↓
Best Model Selection
       ↓
Saved .keras Model
```

The training notebook is included in the repository so that the training and evaluation process can be inspected.

---

## 📈 Model Evaluation

Model performance can be evaluated using multiple classification metrics rather than accuracy alone.

The project considers:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The confusion matrix is particularly useful for understanding which skin-lesion classes are being confused with one another.

> **Note:** Actual evaluation values should be taken directly from the training notebook rather than manually estimated.

---

## 🔮 Model Prediction

When a user uploads an image:

1. The image is loaded.
2. The image is preprocessed.
3. The trained EfficientNetB3 model analyzes the image.
4. The model produces probabilities for the seven classes.
5. The class with the highest predicted probability is selected.
6. The corresponding confidence score is displayed.
7. The prediction is passed to Gemini for educational explanation.

### Prediction Flow

```text
Input Image
    ↓
EfficientNetB3
    ↓
7 Class Probabilities
    ↓
Highest Probability
    ↓
Predicted Class + Confidence
```

A confidence score represents the model's prediction probability. It should **not** be interpreted as medical certainty.

---

# 🤖 Generative AI — Google Gemini

Google Gemini is used as the educational assistant layer of the application.

**Gemini does not perform the primary image classification.**

The classification is performed by the trained **EfficientNetB3** model.

Gemini receives the classification context and can provide:

- General educational information
- Explanation of the predicted category
- User-friendly descriptions
- Multilingual responses
- Answers to follow-up questions

### Gemini Workflow

```text
EfficientNetB3
      ↓
Prediction + Confidence
      ↓
Google Gemini
      ↓
Educational Explanation
      ↓
User Follow-up Questions
```

---

# 🌐 Multilingual Support

DermaAI supports educational interaction in:

- 🇬🇧 English
- 🇮🇳 Kannada
- 🇮🇳 Hindi

This is intended to make AI-generated healthcare information more accessible to users who may prefer regional languages.

---

# ✨ Key Features

- 🧠 EfficientNetB3 deep-learning classification
- 🔬 Seven-class skin-lesion classification
- 🖼️ Image upload and processing
- 📊 Confidence score
- 🤖 Google Gemini educational assistant
- 🌐 English, Kannada, and Hindi support
- 💬 Follow-up questions and interaction
- 🖥️ Streamlit web interface
- 📓 Training notebook included
- 🔐 API key protection using environment variables
- ⚠️ Medical safety disclaimer

---

# 🖼️ Input Image Requirements

The model is designed for suitable dermoscopic skin-lesion images similar to its training data.

For better reliability, images should ideally be:

- Clear
- Well framed
- Properly illuminated
- Focused on the lesion
- Similar to the training image distribution

Performance may be reduced for:

- Blurry images
- Very dark or overexposed images
- Images with heavy obstruction
- Unrelated images
- Images significantly different from the training data

---

# 🛠️ Technology Stack

### Programming

- Python

### Deep Learning

- TensorFlow
- Keras
- EfficientNetB3
- Transfer Learning
- Fine-Tuning

### Dataset

- HAM10000

### Image Processing

- Pillow
- NumPy

### Generative AI

- Google Gemini API

### Web Application

- Streamlit

### Environment Management

- python-dotenv

### Model Format

- Keras `.keras`

### Version Control

- Git
- GitHub

### Deployment

- Streamlit Community Cloud

---

# 📁 Project Structure

```text
skin-disease/
│
├── app.py
├── best_skin_model_phase2.keras
├── requirements.txt
├── README.md
├── .gitignore
│
└── notebooks/
    └── skin_lesion_training.ipynb
```

## Important Files

### `app.py`

Main Streamlit application containing the user interface, model loading, image prediction, and Gemini integration.

### `best_skin_model_phase2.keras`

Trained EfficientNetB3-based classification model.

### `skin_lesion_training.ipynb`

Notebook containing the model-training and evaluation workflow.

### `requirements.txt`

Contains the Python dependencies required to run the application.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Pavankm70/skin-disease.git
cd skin-disease
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

> ⚠️ **Never commit your API key to GitHub.**

Make sure your `.gitignore` contains:

```text
.env
.streamlit/secrets.toml
venv/
```

---

# ▶️ Running the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

After starting the application, Streamlit will provide a local URL in the terminal.

Open that URL in your browser to use DermaAI.

---

# ☁️ Deployment

The application can be deployed using **Streamlit Community Cloud**.

The deployment requires:

- GitHub repository
- `app.py`
- `requirements.txt`
- Trained model file
- Gemini API key configured through Streamlit Secrets

For deployment, the Gemini API key should be stored securely as a secret rather than committed to the repository.

Example Streamlit secret:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

---

# 📓 Model Training Notebook

The model-training notebook is included in the repository:

```text
notebooks/skin_lesion_training.ipynb
```

The notebook allows judges and reviewers to inspect the machine-learning workflow, including data processing and model-training stages.

The notebook covers the project pipeline from dataset preparation through model training and evaluation.

---

# 🔄 Complete Application Workflow

```text
                    HAM10000 DATASET
                           │
                           ▼
                     Data Analysis
                           │
                           ▼
                   Image Processing
                           │
                           ▼
                   Data Augmentation
                           │
                           ▼
                    Class Balancing
                           │
                           ▼
                  EfficientNetB3 Model
                           │
                           ▼
                   Transfer Learning
                           │
                           ▼
                      Fine-Tuning
                           │
                           ▼
                   Model Evaluation
                           │
                           ▼
              best_skin_model_phase2.keras
                           │
                           ▼
                     Streamlit App
                           │
                           ▼
                  User Uploads Image
                           │
                           ▼
                    Model Prediction
                           │
                           ▼
                   Class + Confidence
                           │
                           ▼
                     Google Gemini
                           │
                           ▼
                Educational Explanation
                           │
                           ▼
                   Multilingual Q&A
```

---

# ⚠️ Limitations

Like any machine-learning system, DermaAI has limitations.

### Dataset Dependency

The model learns from the HAM10000 dataset. Its performance may change when presented with images that differ significantly from the training distribution.

### Class Imbalance

The dataset contains significantly different numbers of images for each class.

### Image Quality

Poor-quality, blurry, dark, overexposed, or obstructed images may reduce prediction reliability.

### Visual Similarity

Some skin-lesion categories can have visually similar characteristics, which can result in incorrect classification.

### Confidence Limitations

A high confidence score does not guarantee that a prediction is correct or medically accurate.

### Clinical Validation

This project is an educational/research prototype and has not been presented as a clinically validated diagnostic system.

---

# 🏥 Real-World Impact

Potential applications of the concept include:

- Healthcare education
- Skin-health awareness
- AI-assisted research
- Multilingual health information
- Educational support for students
- AI-assisted healthcare workflows
- Telehealth-related research

However, real-world clinical deployment would require extensive validation, appropriate healthcare oversight, privacy protection, and regulatory compliance.

---

# 🚀 Future Scope

Possible future improvements include:

- Larger and more diverse datasets
- External validation datasets
- Improved handling of class imbalance
- Probability calibration
- Out-of-distribution detection
- Explainable AI using Grad-CAM
- Additional Indian languages
- Mobile application
- Doctor-oriented dashboard
- Telemedicine integration
- Secure healthcare data handling
- Clinical validation
- More advanced AI-assisted healthcare workflows

---

# 💡 Project Innovation

The main innovation of DermaAI is the combination of **Deep Learning classification** with **Generative AI-based educational assistance**.

### Traditional Workflow

```text
Image
  ↓
Classification
  ↓
Result
```

### DermaAI Workflow

```text
Image
  ↓
Deep Learning Classification
  ↓
Prediction + Confidence
  ↓
Generative AI
  ↓
Educational Explanation
  ↓
Multilingual Interaction
  ↓
Follow-up Questions
```

This creates a more understandable and interactive AI experience.

> **Deep Learning for Classification. Generative AI for Understanding. AI for Healthcare.**

---

# 🏆 Hackathon Relevance

DermaAI demonstrates a complete AI workflow:

```text
Healthcare Problem
       ↓
Dataset
       ↓
Data Analytics
       ↓
Image Processing
       ↓
Data Augmentation
       ↓
Class Balancing
       ↓
Deep Learning
       ↓
Transfer Learning
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Prediction
       ↓
Generative AI
       ↓
Multilingual Assistance
       ↓
Web Application
       ↓
Healthcare Awareness
```

This makes the project relevant to the **AI for Healthcare** hackathon track.

---

# 🌐 Links & Resources

### 🚀 Live Demo

**Open DermaAI Live Demo**

> ### 🚀 Live Demo

**[Open DermaAI Live Demo 🚀](https://skin-disease-8njbeyvqstmj8nkckf8gbr.streamlit.app/)**

### 💻 GitHub Repository

**[View Source Code on GitHub](https://github.com/Pavankm70/skin-disease)**

### 📓 Training Notebook

**Open the training notebook**

```text
notebooks/skin_lesion_training.ipynb
```

---
# 👥 Team / Project

| Field | Details |
|---|---|
| **Project** | DermaAI |
| **Track** | AI for Healthcare |
| **Domain** | Healthcare + Artificial Intelligence |
| **Focus** | Skin Lesion Classification + Generative AI |
| **Live Demo** | [Open DermaAI 🚀](https://skin-disease-8njbeyvqstmj8nkckf8gbr.streamlit.app/) |

## 👥 Team Members

| Name | Role |
|---|---|
| **Pavan K M** | AI/ML Developer |
| **Srinivas** | Frontend Developer |
| **Yalaguresh** | Backend Developer |
| **Dilip** | Data & Analytics |
---

# ⚠️ Medical Safety & Disclaimer

DermaAI is an **educational and research prototype**.

It is **not a medical diagnostic tool**, and its predictions should not be used to make medical decisions.

The system may produce incorrect predictions, particularly for images that differ from its training data.

Users should consult a **qualified healthcare professional or dermatologist** for diagnosis, treatment, or concerns regarding a skin lesion.

The project is intended to demonstrate the application of **Deep Learning, Computer Vision, and Generative AI in healthcare**.

---

# 🔬 DermaAI

> **Deep Learning for Classification.**  
> **Generative AI for Understanding.**  
> **AI for Healthcare. 🚀**
