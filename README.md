# 🧠 Fake vs Pristine Image Detection

A Deep Learning project that classifies images as **Pristine (Real)** or **Fake (Manipulated)** using:

- Custom CNN Model
- Transfer Learning (VGG16)
- Binary Classification

---

## 📌 Problem Statement

To build a deep learning model capable of distinguishing between real (pristine) and fake images using Convolutional Neural Networks and Transfer Learning techniques.

---

## 🏗️ Models Implemented

### 1️⃣ Custom CNN
- Multiple Conv2D + MaxPooling layers
- Flatten + Dense layers
- Binary Crossentropy loss
- Adam optimizer

### 2️⃣ Transfer Learning
- VGG16 (Pretrained on ImageNet)
- Frozen base layers
- Custom fully connected top layers
- Fine-tuned for binary classification

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve

---

## 🗂️ Dataset Structure

<img width="278" height="176" alt="image" src="https://github.com/user-attachments/assets/17f1d8a7-0206-4426-ae0d-0b711d6ab10b" />


Labels:
- 0 → Fake
- 1 → Pristine

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies:

pip install -r requirements.txt

3. Run `DetectionCNN.ipynb`

---

## 📈 Results

Both models were evaluated on the test dataset.
Transfer Learning showed improved performance over the custom CNN.

---

## 👨‍💻 Author

**Gaurav Kumar**  
B.Tech CSE (Machine Learning)  
Deep Learning | Computer Vision | AI

---

⭐ If you found this project useful, feel free to star the repository.

