# 🔬 AI Skin Lesion Analyzer

An AI-powered skin lesion classification application using deep learning and Generative AI for educational purposes.

## 🚀 Features

- 🧠 CNN-based skin lesion classification
- 📷 Skin lesion image upload
- 📊 Predicted lesion category and confidence score
- ✨ Gemini AI-generated educational explanation
- 💬 Ask questions about the predicted result
- 🌐 English, Kannada and Hindi interface
- ⚠️ Medical disclaimer

## 🧠 Skin Lesion Classes

The model classifies images into seven categories:

- Actinic Keratoses
- Basal Cell Carcinoma
- Benign Keratosis-like Lesions
- Dermatofibroma
- Melanoma
- Melanocytic Nevi
- Vascular Lesions

## 🔄 Workflow

1. Upload a skin lesion image.
2. Convert the image to RGB.
3. Resize the image to 300 × 300 pixels.
4. Pass the image to the trained CNN model.
5. Display the predicted class and confidence.
6. Generate an educational explanation using Gemini.
7. Allow the user to ask additional questions.

## 🛠️ Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pillow
- Streamlit
- Google Gemini API
- python-dotenv

## ▶️ Run Locally

Install the required packages:

```bash
pip install -r requirements.txt