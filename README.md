<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
     <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology" width="400">
  </a>
</p>

<h1 align="center"><b>Aspect-based Sentiment Analysis (ABSA) for Education Reviews</b></h1>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-ffae00.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## 📖 Introduction
This repository contains the implementation of an **Aspect-based Sentiment Analysis (ABSA)** pipeline. The model specifically extracts aspects (e.g., *teachers*, *facilities*) from educational reviews and classifies their corresponding sentiment (Positive, Negative, or Neutral) using **BiLSTM-CRF** and **Attention** mechanisms.

---

## ✨ Key Features
- **Aspect Extraction:** Uses a Sequence Labeling approach (BiLSTM-CRF) to accurately extract Multi-word Aspects (`B-ASP`, `I-ASP`) and Opinions (`B-OPI`, `I-OPI`) from unstructured text.
- **Sentiment Classification:** Uses an Attention-based BiLSTM model to classify the overarching sentiment of the review sentence.
- **Interactive UI:** Provides a built-in Gradio web application for real-time inference with text highlighting.

---

## 🛠 Project Structure

```text
CS221.Q11-NATURAL-LANGUAGE-PROCESSING/
├── data/               # Datasets and raw data (.zip, .csv)
├── docs/               # Documentations, Reference Papers, Video Lectures
├── models/             # Pretrained weights (model_crf.pth, model_sent.pth)
├── notebooks/          # Training notebooks (Jupyter/Colab)
├── scripts/            # Debugging and evaluation scripts
├── app.py              # Gradio Web Interface
├── config.json         # Model hyperparameters & configurations
├── vocab.pkl           # Pickled vocabulary dictionary
├── README.md           # Project documentation
└── .gitignore          # Git exclusion file
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TuTTC/CS221.Q11-NATURAL-LANGUAGE-PROCESSING.git
   cd CS221.Q11-NATURAL-LANGUAGE-PROCESSING
   ```

2. **Create a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   Ensure you have PyTorch installed according to your system specs.
   ```bash
   pip install torch torchvision torchaudio
   pip install gradio pytorch-crf
   ```

---

## 🚀 Usage

To run the interactive Gradio demo locally:

1. Make sure `vocab.pkl`, `config.json`, `model_crf.pth`, and `model_sent.pth` are in the root directory.
2. Execute the `app.py` script:
   ```bash
   python app.py
   ```
3. Open the link provided in the terminal (usually `http://127.0.0.1:7860/`) in your web browser.
4. Input an educational review in English (e.g., *"The professors are knowledgeable but the canteen food is terrible"*) to see the extracted aspects and sentiment polarity.

---

## 👥 Course & Team Information

- **Course:** NATURAL LANGUAGE PROCESSING (CS221)
- **Class Code:** CS221.Q11
- **Instructor:** Dr. Nguyen Trong Chinh

### Team Members
| No | Student ID | Full Name       | Role   | Github                                 | Email                   |
|:--:|:----------:|:----------------|:------:|:---------------------------------------|:------------------------|
| 1  | 23521704   | Tran Thi Cam Tu | Member | [TuTTC](https://github.com/TuTTC)      | 23521704@gm.uit.edu.vn  |
| 2  |            | Dinh Hoang Phuc | Member |                                        |                         |

---

## 📚 References

This project is built and evaluated upon principles and datasets introduced in the following publications:

1. [EduRABSA: An Education Review Dataset for Aspect-based Sentiment Analysis Tasks](https://arxiv.org/abs/2508.17008)
2. [Data-Efficient Adaptation and a Novel Evaluation Method for Aspect-based Sentiment Analysis](https://arxiv.org/abs/2511.03034)
