# 🧠 Neural Storyteller – Image Caption Generator (Seq2Seq)

## 👩‍💻 Authors
- **F223616 – Ishrat Fatima**
- **22F-3617 – Sabahat Jahangir**

---

## 📌 Project Overview
This project implements a **multimodal deep learning model** that generates natural language descriptions for images using a **Sequence-to-Sequence (Seq2Seq)** architecture.

The system combines:
- **Computer Vision (CNN)** for image feature extraction
- **Natural Language Processing (RNN/LSTM)** for caption generation

Additionally, we developed a **Streamlit web application** to interactively generate captions for input images.

---

## 🎯 Objective
To build an image captioning system that:
- Understands visual content
- Generates meaningful textual descriptions
- Demonstrates real-world multimodal AI capability

---

## ⚙️ Environment Setup
- Platform: Kaggle Notebook
- GPU: Dual GPU (T4 x2)
- Framework: PyTorch
- Dataset: Flickr30k

---

## 📂 Dataset
We used the **Flickr30k dataset**, which contains:
- 30,000 images
- 5 captions per image

---

## 🧩 Project Pipeline

### 🔹 Part 1: Feature Extraction (CNN)
- Used **pre-trained ResNet50**
- Removed final classification layer
- Extracted **2048-dimensional feature vectors**
- Cached features into:flickr30k_features.pkl
  
✔ Benefit:
- Avoids expensive CNN training during Seq2Seq learning

---

### 🔹 Part 2: Text Preprocessing
- Loaded captions file
- Performed:
- Lowercasing
- Tokenization
- Vocabulary creation
- Special tokens:
  ```
  <start>, <end>, <pad>, <unk>
  ```

---

### 🔹 Part 3: Seq2Seq Architecture

#### 🧠 Encoder
- Linear layer:2048 → 512 (hidden size)
  - Converts image features into hidden representation

#### 📝 Decoder
- LSTM-based network
- Components:
- Word Embedding layer
- LSTM
- Fully connected output layer

#### 🔁 Flow:Image → ResNet50 → Feature Vector → Encoder → Decoder → Caption

---

### 🔹 Part 4: Training

- Loss Function: CrossEntropy Loss  
  (with padding ignored)
- Optimizer: Adam
- Teacher Forcing used during training

---

## 🔍 Inference Methods

### ✅ Greedy Search
- Selects most probable word at each step

### ✅ Beam Search
- Keeps multiple candidate sequences
- Produces better captions than greedy approach

---

## 📊 Evaluation Metrics

We evaluated the model using:

- **BLEU-4 Score**
- **Precision**
- **Recall**
- **F1 Score**

(Optional metrics like METEOR/ROUGE can also be applied)

---

## 📈 Results

### 🖼 Caption Examples
- Displayed test images with:
  - Ground truth captions
  - Generated captions

### 📉 Loss Curve
- Training and validation loss plotted over epochs

---

## 🌐 Streamlit Web App

We built an interactive web app using **Streamlit**.

### 💡 Features:
- Upload image
- Generate caption instantly
- User-friendly interface

### ▶️ Run the App
```bash
streamlit run app.py

---
### ▶️ Project Structure

├── app.py                      # Streamlit app
├── image_captioning_model.pth          
├── vocab.pkl                   # Cached features
├── requirements.txt            # Dependencies
└── README.md
