# 🚦 SignSense: Indian Traffic Sign Learning Assistant

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://signsense-assistant.streamlit.app/)

An end-to-end ML web application that identifies Indian traffic signs and delivers highly structured educational content in multiple regional languages. 

SignSense uses a **Hybrid CNN + Vision LLM pipeline**: A lightweight CNN (TensorFlow Lite) provides real-time prediction hints, and a Groq-powered Vision LLM (Llama Scout) performs the final robust classification and generates contextual legal advice based on the Indian Motor Vehicles Act.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **Hybrid Pipeline** | Uses a custom TFLite CNN alongside Groq Vision AI for near 100% accuracy on both real-world photos and internet clipart. |
| 🌍 **Multilingual Support** | Outputs rules, guidelines, and fine structures in 11 Indian languages (English, Hindi, Telugu, Marathi, Tamil, Bengali, etc.). |
| 💬 **Contextual Chatbot** | Talk directly to the AI about the currently identified sign. Ask about challans, exact sections of the Motor Vehicles act, and defensive driving. |
| 🎨 **Modern Glassmorphism UI** | Built on Streamlit with custom CSS animated traffic lights, responsive columns, and dark-mode glass styling. |

---

## 📁 Project Structure

```
SignSense/
├── .env                     ← Your API keys (not committed)
├── .env.example             ← Template for .env
├── requirements.txt         ← Python dependencies
├── README.md                ← Project documentation
├── app.py                   ← Core Streamlit web application
│
└── models/                  
    ├── traffic_model.tflite ← Pre-trained TFLite CNN weights
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
streamlit run app.py
```

The Streamlit app will open automatically at **http://localhost:8501**.

**How to use:**
1. Browse or upload an image of a traffic sign.
2. Select your preferred **Output Language** from the dropdown.
3. Click **Analyze Sign**.
4. Read the detailed Groq AI educational breakdown (Overview, Core Rules, Defensive Driving, Legal Implications).
5. Use the chat interface to ask questions about the detected sign!

---

## 🔧 Requirements

- Python 3.9+
- Streamlit
- AI Edge LiteRT (TensorFlow Lite runtime)
- Groq Python SDK
- Pillow, NumPy, python-dotenv

---

## 📄 License

For educational use. Dataset originally sourced from the **Indian Traffic Sign Dataset** on Kaggle.
