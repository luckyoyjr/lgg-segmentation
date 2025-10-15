# 🧠 Brain Tumor Image Segmentation (Hybrid Quantum–Classical U-Net)

This Streamlit web app demonstrates an end-to-end medical image segmentation workflow using a **Hybrid Quantum–Classical U-Net**.  
It loads MRI images, performs preprocessing, segmentation, and displays predicted tumor masks with metrics such as Dice coefficient and IoU.

---

## 🚀 Features
- Interactive upload and visualization of MRI slices  
- U-Net–based segmentation (classical / hybrid quantum bottleneck)  
- Real-time inference and overlay visualization  
- Caching for faster predictions  
- Lightweight CPU deployment (no GPU required)  

---

## 🧩 Model Overview
The segmentation model is trained on the **TCGA-LGG** dataset, combining:
- Classical convolutional encoder–decoder
- Quantum bottleneck layer (rotation gates, entanglement, and measurement)
- Evaluation using Dice, IoU, precision, recall, and F1 metrics  

Weights are stored externally (Hugging Face Hub / Google Drive) to keep the repository lightweight.

---

## 🖥️ Deployment Options

### 1️⃣ Streamlit Community Cloud  
1. Fork or upload this repo to GitHub  
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**  
3. Select your repo and `app.py` file  
4. Set environment variable for model URL under **Settings → Secrets**

### 2️⃣ Hugging Face Spaces  
1. Create a new Space → **SDK: Streamlit**  
2. Add the following files:  
   - `app.py`  
   - `requirements.txt`  
   - `README.md`  
3. Commit → the Space builds automatically and gives you a public URL  

---

## 📦 Requirements

```bash
streamlit>=1.33
torch==2.3.1+cpu
torchvision==0.18.1+cpu
--extra-index-url https://download.pytorch.org/whl/cpu
numpy
pillow
opencv-python-headless
scikit-image
matplotlib
pandas
