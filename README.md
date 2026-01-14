# 🩺 Chronic Kidney Disease (CKD) Prediction from CT Images Using Deep Learning

This repository contains a **deep learning–based medical image classification project** that predicts kidney conditions from CT scan images. A **Convolutional Neural Network (CNN)** is used to classify kidney CT images into four categories, and **Grad-CAM** is applied to provide explainable visual insights into model predictions.

The project is implemented using **TensorFlow/Keras** and is designed to run seamlessly on **Google Colab**.

---

## 📌 Project Overview

- **Problem Type:** Multi-class image classification  
- **Domain:** Medical Imaging / Healthcare AI  
- **Model Type:** Convolutional Neural Network (CNN)  
- **Explainability:** Grad-CAM  
- **Framework:** TensorFlow / Keras  

This project aims to assist in the **early detection and analysis of kidney-related diseases** using CT scan images.

---

## 🗂 Dataset Information

- **Dataset Name:** CT KIDNEY DATASET: Normal–Cyst–Tumor and Stone  
- **Source:** Kaggle  
- **Dataset Link:**  
  https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone  

### 📊 Classes
The dataset contains CT images categorized into four classes:
1. **Normal**
2. **Cyst**
3. **Tumor**
4. **Stone**

Images are organized by class, making them suitable for supervised deep learning.

---

## ⚙️ Project Workflow

1. **Dataset Collection**
   - Dataset downloaded directly from Kaggle using Kaggle API
   - `kaggle.json` is required for authentication

2. **Data Preprocessing**
   - Image resizing
   - Normalization
   - Label encoding
   - Train–test split

3. **Model Development**
   - CNN architecture built using Keras
   - Sparse Categorical Cross-Entropy loss function
   - Adam optimizer used for training

4. **Model Training & Evaluation**
   - Accuracy, Precision, Recall, and F1-score calculated
   - Confusion Matrix and Precision–Recall Curve generated

5. **Explainable AI**
   - Grad-CAM applied to visualize important image regions
   - Helps interpret model decision-making

6. **Model Saving**
   - Trained model saved in `.h5` format

---

## 🧠 Model Details

- **Input:** Kidney CT images (RGB)
- **Output:** 4-class probability prediction
- **Loss Function:** Sparse Categorical Cross-Entropy
- **Optimizer:** Adam

---

## 📈 Performance Evaluation

The model performance is evaluated using:
- Classification Accuracy
- Precision, Recall, and F1-Score
- Confusion Matrix
- Precision–Recall Curve
- Metric comparison bar charts

These results demonstrate effective classification across all kidney condition classes.

---

## 🔍 Model Explainability (Grad-CAM)

To improve transparency and trust:
- Grad-CAM highlights regions of CT images influencing predictions
- Useful for medical interpretation and validation
- Enhances model explainability for clinical research

---

## 💾 Saved Model

After training, the model is saved as:
CKD_CNN_Model.h5

This file can be reused for inference or deployment.

---

## ▶️ How to Run the Project

### Requirements
- Python 3.x
- Google Colab (recommended)
- Kaggle account

### Steps
1. Open the notebook in Google Colab
2. Upload `kaggle.json` when prompted
3. Run all cells sequentially
4. Dataset will download automatically
5. Model will train, evaluate, and save

---

## 🧪 Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- OpenCV  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 🚀 Future Improvements

- Transfer learning (ResNet, EfficientNet)
- Hyperparameter tuning
- Web-based diagnostic interface
- Cross-dataset validation
- Integration with clinical data

---

## ⚠️ Disclaimer

This project is **for academic and research purposes only**.  
It is **not intended for real-world clinical diagnosis** without medical expert validation.

---

## 📚 Citation

If you use this dataset, please credit the original author:

**Nazmul Hasan**, *CT Kidney Dataset: Normal–Cyst–Tumor and Stone*, Kaggle

---

## 🤝 Acknowledgements

- Kaggle community for dataset support
- Open-source deep learning libraries

---

⭐ If you find this project useful, consider starring the repository!
