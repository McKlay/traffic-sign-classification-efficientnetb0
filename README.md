---
title: 🚦 Traffic Sign Recognition – EfficientNetB0 (GTSRB)
emoji: 🚗
colorFrom: blue
colorTo: red
sdk: gradio
app_file: app.py
license: mit
model_card: true
---

[![HF Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue?logo=huggingface&style=flat-square)](https://github.com/McKlay/traffic-sign-classification-efficientnetb0)
[![Gradio](https://img.shields.io/badge/Built%20with-Gradio-orange?logo=gradio&style=flat-square)](https://www.gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![GitHub last commit](https://img.shields.io/github/last-commit/McKlay/traffic-sign-classification-efficientnetb0)
![GitHub Repo stars](https://img.shields.io/github/stars/McKlay/traffic-sign-classification-efficientnetb0?style=social)
![GitHub forks](https://img.shields.io/github/forks/McKlay/traffic-sign-classification-efficientnetb0?style=social)
![MIT License](https://img.shields.io/github/license/McKlay/traffic-sign-classification-efficientnetb0)

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=McKlay.traffic-sign-classification-efficientnetb0)

# 🚦 Traffic Sign Recognition - Hugging Face Inference App

This project performs traffic sign classification using a fine-tuned EfficientNetB0 model on the GTSRB dataset.

---

## 🌐 Demo
Deployed on Hugging Face Spaces [Hugging Face Space](https://huggingface.co/spaces/McKlay/traffic-sign-classification-efficientnetb0)

---

## Folder Structure
```bash
9_TrafficSignRecognition-HF/
│ app.py
│ README.md
│ requirements.txt
│ utils.py
└───model/
    └── traffic_sign_model.pth
```

---

## Model Details

- **Architecture:** EfficientNetB0
- **Input Size:** 224×224 (ImageNet normalized)
- **Classes:** 43 official GTSRB categories (e.g., “Stop”, “Yield”, “Slippery Road”)
- **Training:** Last-layer fine-tuning on GTSRB with data augmentation
- **Accuracy:** ~99.9% validation accuracy on Kaggle (requires further generalization testing)

> [Kaggle Training Notebook](https://www.kaggle.com/code/claymarksarte/traffic-sign-recognition-with-efficientnetb0)


---

## How to Run Locally

```bash
git clone https://github.com/McKlay/traffic-sign-classification-efficientnetb0.git
cd traffic-sign-classification-efficientnetb0
pip install -r requirements.txt
python app.py
```

---

## Requirements

torch
torchvision
gradio
Pillow

---

## Notes

- Model trained on Kaggle using data augmentation and last-layer fine-tuning.

- Download the traffic_sign_model.pth file from [Kaggle Training Notebook](https://www.kaggle.com/code/claymarksarte/traffic-sign-recognition-with-efficientnetb0)

- To test generalization, deploy and evaluate on real-world traffic sign images.

---

## 🧑‍💻 Author

Clay Mark Sarte  
[GitHub](https://github.com/McKlay) | [LinkedIn](https://www.linkedin.com/in/clay-mark-sarte-283855147/)





