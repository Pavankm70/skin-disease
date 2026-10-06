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
