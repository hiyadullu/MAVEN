# Emotion Detection using Swin Transformer (Image + Video)

A two-stage deep learning pipeline for facial emotion recognition, combining spatial learning from images and temporal understanding from videos.

---

## Overview

This project explores emotion recognition by first training a model on static facial images and then extending it to video-based emotion detection using temporal aggregation.

---

## ⚙️ Pipeline

**Stage 1 – Image Training**
- Dataset: FER-2013  
- Model: Swin Transformer  
- Goal: Learn facial feature representations  

**Stage 2 – Video Training**
- Datasets: IEMOCAP, RAVDESS  
- Input: 8 frames per video  
- Method: Temporal pooling over frame features  
- Goal: Learn emotion dynamics over time  

---

## 🧩 Model Architecture

- Backbone: Swin Transformer  
- Transfer Learning: Stage 1 → Stage 2  
- Temporal Layer: Fully connected head over pooled frame features  

---

## 📊 Results

### Stage 1 (FER-2013)
- Accuracy: ~74%  
- Balanced performance across classes  

### Stage 2 (Video Model)
- Observed high validation accuracy (~90%)  
- Identified data leakage due to improper dataset split  
- Proposed fix: video-level splitting to ensure independence  

---

## 📈 Visualizations

### Confusion Matrix
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/c38a5499-7e27-460b-a3f8-1520d0905637" />


### Accuracy Curve
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/c41aba52-8a01-4751-bc07-c476b82fb0b4" />


### Loss Curve
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/7485e058-8c8a-419f-8f27-84f5b77003b8" />


### ROC Curve
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/708fca2b-7e68-45df-ba72-0ad1c62c3bb2" />


### Precision-Recall Curve
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/a2784d60-cd1d-4e8b-9755-3de401834fbf" />


---

## 🛠️ Tech Stack

- PyTorch  
- Hugging Face Transformers  
- NumPy  
- Matplotlib, Seaborn  
- OpenCV  

---

## 📁 Project Structure

emotion_det/
├── data/
├── models/
├── scripts/
├── results/
├── checkpoints/
├── requirements.txt


---

## ⚠️ Notes

- Video model performance is affected by dataset leakage during validation  
- Proper evaluation requires strict separation at video level  
- Future work includes improving temporal modeling (LSTM / 3D CNN)

---

## 📌 Conclusion

This project demonstrates a structured approach to emotion recognition using transfer learning and highlights the importance of proper validation strategies in deep learning workflows.

