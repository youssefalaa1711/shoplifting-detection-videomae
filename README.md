# Shoplifting Detection Using Video Models

This project explores two different deep learning approaches for detecting shoplifting behavior from video clips:

1. **3D Convolutional Neural Networks (3D CNN)**
2. **VideoMAE (Masked Autoencoder) from Hugging Face Transformers**

The goal was to classify short video clips into two categories:
- `shop_lifter`
- `non_shop_lifter`

---

## 📂 Dataset
The dataset consisted of video clips labeled according to the above two classes.  
Each clip was preprocessed into a sequence of frames and resized to a fixed resolution for model input.

---

## 🧠 Methods

### 1️⃣ 3D CNN Approach
- **Architecture:** A custom 3D Convolutional Neural Network.
- **Input:** Stacked frames shaped as `(T, C, H, W)`.
- **Training:** Standard cross-entropy loss, Adam optimizer, early stopping based on validation accuracy.
- **Result:** Achieved **100% accuracy** on the validation set for the dataset used.

This model was computationally cheaper and faster to train due to its smaller architecture, making it well-suited for smaller datasets.

---

### 2️⃣ Hugging Face VideoMAE Approach
- **Base Model:** `MCG-NJU/videomae-base-finetuned-kinetics` from Hugging Face.
- **Modifications:**
  - Classifier head adapted for binary classification (`shop_lifter` vs `non_shop_lifter`).
  - Option to **freeze the backbone** for faster training.
- **Preprocessing:**
  - Uniformly sampled frames from each video.
  - Used `VideoMAEImageProcessor` for normalization and formatting.
- **Training:**
  - Mixed precision training with `torch.cuda.amp` for speed.
  - Early stopping after no improvement in validation loss for a set patience period.
- **Result:**  
  Achieved **98%+ accuracy** on the validation set:
  ```
  Epoch 10 | train_loss 0.0349 acc 0.989 | val_loss 0.0820 acc 0.980
  ```
  The model generalized well, with minimal overfitting due to proper regularization.

---

## ⚡ Key Optimizations
- **Freezing backbone** for initial experiments to speed up training.
- **Early stopping** to prevent overfitting.

---

## 📊 Results Summary

| Model         | Accuracy | Notes                                   |
|---------------|----------|-----------------------------------------|
| 3D CNN        | 100%     | Perfect fit on dataset                  |
| VideoMAE (HF) | 98%+     | Excellent generalization and robustness |

