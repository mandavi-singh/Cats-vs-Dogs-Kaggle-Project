# Cats-vs-Dogs-Kaggle-Project
# 🐱🐶 Cats vs. Dogs Image Classification using a CNN

This repository contains code and resources for a deep learning project focused on **binary image classification** — distinguishing between images of **cats and dogs** using a **Convolutional Neural Network (CNN)** built with TensorFlow.  
Developed by **Mandavi Singh** as part of a machine learning curriculum.

---

## 📚 Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Methodology](#methodology)  
  - [Data Preprocessing](#data-preprocessing)  
  - [Model Architecture](#model-architecture)  
  - [Training and Callbacks](#training-and-callbacks)  
- [Results and Performance](#results-and-performance)  
- [How to Use](#how-to-use)  
- [Future Work](#future-work)  
- [Contact](#contact)  

---

## 📌 Project Overview

This project tackles the classic **Cats vs. Dogs** classification challenge. A custom CNN was constructed and trained on a large dataset to learn distinguishing features of both animals. The entire pipeline was implemented in a **Kaggle Notebook**, leveraging **Tesla T4 GPU** for efficient training.

- **Tech Stack**: Python, TensorFlow (Keras), NumPy, Matplotlib  
- **Platform**: Kaggle Notebooks  
- **Hardware**: Tesla T4 GPU  

---

## 🐾 Dataset

- **Source**: Microsoft-sponsored Kaggle competition  
- **Total Images**: ~25,000 JPGs  
- **Classes**: `Cat`, `Dog`  
- **Balanced Data**: 50% cats, 50% dogs  

### 🔀 Data Split:
- **Training**: 80% (20,000 images)  
- **Validation**: 10% (2,500 images)  
- **Test**: 10% (2,500 images)  

---

## 🧪 Methodology

### 🔧 Data Preprocessing

- Removed corrupted images  
- Resized all images to **150x150 pixels**  
- Split into `train`, `val`, and `test` folders  
- Used TensorFlow’s `image_dataset_from_directory()` for efficient loading  

### 🧠 Model Architecture

A **Sequential CNN** designed to extract hierarchical features:

Conv2D(32) ➝ Conv2D(64) ➝ MaxPooling2D
Conv2D(64) ➝ Conv2D(128) ➝ MaxPooling2D
Conv2D(128) ➝ Conv2D(256) ➝ GlobalAveragePooling2D
Dense(1024) ➝ Dense(1, activation='sigmoid')

- **Total Parameters**: 837,121 (All trainable)

### ⚙️ Training and Callbacks

- **Optimizer**: Adam  
- **Loss Function**: Binary Crossentropy  
- **Batch Size**: 32  
- **Epochs**: Up to 50  
- **Callback**:  
  - `EarlyStopping` with `patience=3` based on `val_loss`  
  - Automatically restores best weights

---

## 📊 Results and Performance

- ✅ **Test Accuracy**: **84%**  
- 📉 Training and validation losses dropped from ~0.7 to < 0.2  
- 📈 Validation Accuracy peaked around **87%**, indicating minimal overfitting  

### 🧾 Classification Report:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Cat   | 0.86      | 0.81   | 0.83     |
| Dog   | 0.82      | 0.87   | 0.84     |

---

## ▶️ How to Use

1. **Clone the repository:**

```bash
git clone https://github.com/mandavi-singh/Cats-vs-Dogs-Classifier.git
cd Cats-vs-Dogs-Classifier
```



2. **Install dependencies:**
```bash
pip install tensorflow matplotlib numpy
```
3. **Download Dataset:**

Get the dataset from Kaggle and place it in the correct folder structure as referenced in the notebook.

4. **Run the Notebook:**

Open cats_vs_dogs_classifier.ipynb in Jupyter or Kaggle and run all cells.

5. **Make Predictions:**

Load the saved model (cats_dogs_classifier.h5) and run predictions on new images.

🔮 **Future Work**
🔄 Add Data Augmentation (rotation, flipping, zooming)

🔁 Apply Transfer Learning (VGG16, ResNet, EfficientNet)

🌐 Deploy as a web app using Streamlit or Flask

🙋‍♀️ **Contact**
Feel free to reach out for questions, collaborations, or feedback.

Author: Mandavi Singh
🔗 LinkedIn-: www.linkedin.com/in/mandaviofficial
<img width="793" height="88" alt="image" src="https://github.com/user-attachments/assets/b472af4a-a5c2-4ec1-a4dc-03e8bd90bb1e" />
📧 singhmandavi002@gmail.com

