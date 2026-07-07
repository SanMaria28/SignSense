# 🚦 Indian Traffic Sign Learning Assistant

An end-to-end ML application that **identifies Indian traffic signs** from photos using a CNN, then uses the **Groq LLM API** to deliver structured educational content about each sign — including its legal meaning, driving rules, and safety tips under the Indian Motor Vehicles Act.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔍 CNN Classifier | 4-block Keras CNN trained on 59 Indian traffic sign classes |
| 🤖 Groq AI Explanations | `llama-3.3-70b-specdec` generates legal & safety breakdowns |
| 🖥️ Gradio UI | Dark-mode web interface with live predictions |
| 📊 59 Sign Classes | Mandatory, Cautionary & Informatory signs per Indian MVA |

---

## 📁 Project Structure

```
Indian Traffic Sign Learning Assistant/
├── archive (5).zip          ← Dataset zip (already present)
├── .env                     ← Your secrets (not committed)
├── .env.example             ← Template for .env
├── requirements.txt         ← Python dependencies
├── README.md
│
├── ml_pipeline/
│   └── train.py             ← CNN training script
│
├── app/
│   └── app.py               ← Gradio web application
│
└── models/                  ← Created after training
    ├── traffic_model.h5     ← Saved CNN weights
    ├── class_map.json       ← {idx: class_name} mapping
    └── training_summary.json
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Windows PowerShell
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` and replace `your_groq_api_key_here` with your actual key from [console.groq.com](https://console.groq.com/).

---

## 🏋️ Step 1 – Train the CNN

The training script reads the dataset directly from the zip file — no manual extraction required.

```bash
# Run from the project root
python ml_pipeline/train.py
```

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--zip` | `../archive (5).zip` | Path to the dataset zip |
| `--output` | `../models` | Directory to save the model |

**Example with custom paths:**
```bash
python ml_pipeline/train.py --zip "archive (5).zip" --output models
```

Training will:
1. Extract the zip to a temporary directory (auto-cleaned up)
2. Load and resize all 13 000+ images to 64×64
3. Apply data augmentation (flip, rotate, zoom, brightness)
4. Train for up to 25 epochs with early stopping
5. Save `traffic_model.h5`, `class_map.json`, and `training_summary.json` to `models/`

> **Expected time:** ~10–30 minutes on CPU, ~3–5 minutes on GPU.

---

## 🌐 Step 2 – Launch the App

```bash
python app/app.py
```

The Gradio app will open automatically at **http://localhost:7860**.

**How to use:**
1. Click **Upload Indian Traffic Sign** and choose a photo
2. Click **🔍 Analyse Sign**
3. Read the CNN prediction and the Groq AI educational breakdown

---

## 📋 Traffic Sign Classes (59 total)

| Range | Category | Examples |
|---|---|---|
| 0–22 | **Mandatory** | Give way, No entry, Speed limits, No parking |
| 23–49 | **Cautionary** | Steep descent, Narrow road, Crossroads, Level crossing |
| 50–58 | **Informatory** | Parking, Bus stop, Hospital, Restaurant, Hotel |

---

## 🔧 Requirements

- Python 3.9+
- TensorFlow 2.13+
- Gradio 4.20+
- Groq Python SDK
- Pillow, NumPy, python-dotenv

---

## 📄 License

Educational use. Dataset sourced from the **Indian Traffic Sign Dataset** on Kaggle.
