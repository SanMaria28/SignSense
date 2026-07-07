# 🚦 SignSense: Indian Traffic Sign Learning Assistant

An end-to-end ML web application that identifies Indian traffic signs and delivers highly structured educational content in multiple regional languages. 

SignSense uses a **Hybrid CNN + Vision LLM pipeline**: A lightweight CNN provides real-time prediction hints, and a Groq-powered Vision LLM (Llama Scout) performs the final robust classification and generates contextual legal advice based on the Indian Motor Vehicles Act.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **Hybrid Pipeline** | Uses a custom Keras CNN alongside Groq Vision AI for near 100% accuracy on both real-world photos and internet clipart. |
| 🌍 **Multilingual Support** | Outputs rules, guidelines, and fine structures in 11 Indian languages (English, Hindi, Telugu, Marathi, Tamil, Bengali, etc.). |
| 💬 **Contextual Chatbot** | Talk directly to the AI about the currently identified sign. Ask about challans, exact sections of the Motor Vehicles act, and defensive driving. |
| 🎨 **Modern Glassmorphism UI** | Built on Gradio 4 with custom CSS animated traffic lights, responsive columns, and dark-mode glass styling. |

---

## 📁 Project Structure

```
SignSense/
├── .env                     ← Your API keys (not committed)
├── .env.example             ← Template for .env
├── requirements.txt         ← Python dependencies
├── README.md                ← Project documentation
│
├── app/
│   └── app.py               ← Core Gradio web application
│
└── models/                  
    ├── traffic_model.h5     ← Pre-trained CNN weights
    └── class_map.json       ← Index to class name mapping
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Copy the `.env.example` file to `.env`:

```bash
# Windows PowerShell
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` and replace the placeholder with your actual key from [console.groq.com](https://console.groq.com/).

### 3. Launch the App

```bash
python app/app.py
```

The Gradio app will open automatically at **http://localhost:7860**.

**How to use:**
1. Click **Upload Sign** and choose a photo.
2. Select your preferred **Output Language** from the dropdown.
3. Click **Analyse Sign**.
4. Read the detailed Groq AI educational breakdown (Overview, Core Rules, Defensive Driving, Legal Implications).
5. Switch to the **Ask Questions** tab to chat with the AI about the detected sign!

---

## 🔧 Requirements

- Python 3.9+
- TensorFlow 2.13+ (for running the CNN hint generator)
- Gradio 4.0+
- Groq Python SDK
- Pillow, NumPy, python-dotenv

---

## 📄 License

For educational use. Dataset originally sourced from the **Indian Traffic Sign Dataset** on Kaggle.
