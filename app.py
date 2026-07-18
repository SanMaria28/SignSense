"""
Indian Traffic Sign Learning Assistant – Gradio App (Vision LLM + CNN + Chat)
=============================================================================
Run:
    python app/app.py
"""

import os
import io
import re
import json
import base64
import logging
from pathlib import Path

import time
import concurrent.futures
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
MODELS_DIR     = BASE_DIR / "models"
CLASS_MAP_PATH = MODELS_DIR / "class_map.json"
SUMMARY_PATH   = MODELS_DIR / "training_summary.json"

# ── Load training summary ─────────────────────────────────────────────────────
ARCH       = "custom_cnn"
IMG_SIZE   = 96
PREPROCESS = "mobilenet_v2"

if SUMMARY_PATH.exists():
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        _summary = json.load(f)
    ARCH       = _summary.get("architecture", ARCH)
    IMG_SIZE   = _summary.get("image_size", IMG_SIZE)
    PREPROCESS = _summary.get("preprocessing", PREPROCESS)

# ── Load TFLite model ─────────────────────────────────────────────────────────
def load_tflite_model():
    try:
        import ai_edge_litert.interpreter as tflite
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                log.error("ai_edge_litert or tflite_runtime or tensorflow not installed.")
                return None

    path = MODELS_DIR / "traffic_model.tflite"
    if not path.exists():
        log.error(f"TFLite model not found at {path}")
        return None

    try:
        interpreter = tflite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        log.info(f"TFLite loaded: {path.name}  input={input_details[0]['shape']}  classes={output_details[0]['shape'][-1]}")
        return interpreter
    except Exception as exc:
        log.error(f"Failed to load {path.name}: {exc}")
        return None

def load_class_map() -> dict[str, str]:
    if not CLASS_MAP_PATH.exists():
        return {}
    with open(CLASS_MAP_PATH, encoding="utf-8") as f:
        return json.load(f)

cnn_model = load_tflite_model()
class_map  = load_class_map()
ALL_CLASSES = list(class_map.values())

# ── Groq client ───────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client  = None

if GROQ_API_KEY:
    try:
        from groq import Groq
        import httpx
        groq_client = Groq(api_key=GROQ_API_KEY,
                           http_client=httpx.Client(verify=False))
        log.info("Groq client initialised.")
    except ImportError:
        log.warning("groq / httpx not installed.")
else:
    log.warning("GROQ_API_KEY not set.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def pil_to_base64(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    rgb = pil_image.convert("RGB")
    rgb.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def composite_on_white(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode in ("RGBA", "LA") or (
        pil_image.mode == "P" and "transparency" in pil_image.info
    ):
        rgba = pil_image.convert("RGBA")
        bg   = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba)
    return pil_image

def cnn_top3(pil_image: Image.Image) -> list[tuple[str, float]]:
    if cnn_model is None:
        return []
    try:
        input_details = cnn_model.get_input_details()[0]
        output_details = cnn_model.get_output_details()[0]
        
        h, w = input_details['shape'][1], input_details['shape'][2]
        img  = composite_on_white(pil_image).convert("RGB").resize((w, h))
        arr  = np.array(img, dtype=np.float32)
        
        is_gtsrb = output_details['shape'][-1] == 43
        if PREPROCESS == "mobilenet_v2" and not is_gtsrb:
            # mobilenet_v2 preprocess_input manually
            arr = (arr / 127.5) - 1.0
        else:
            arr = arr / 255.0
            
        arr = np.expand_dims(arr, 0)
        
        cnn_model.set_tensor(input_details['index'], arr)
        cnn_model.invoke()
        preds = cnn_model.get_tensor(output_details['index'])[0]
        
        top3  = np.argsort(preds)[::-1][:3]
        return [(class_map.get(str(i), f"class_{i}"), float(preds[i]) * 100) for i in top3]
    except Exception as exc:
        log.error(f"CNN error: {exc}")
        return []

# ── Single-call Vision LLM: identify + explain ───────────────────────────────
def vision_identify_and_explain(b64_image: str, cnn_hints: list[tuple[str, float]], language: str = "English") -> tuple[str, str]:
    classes_str = ", ".join(ALL_CLASSES)
    hint_str = ""
    if cnn_hints:
        hint_str = "\nLocal CNN suggestions (may be inaccurate):\n" + "\n".join(
            f"  {i+1}. {n} ({c:.1f}%)" for i, (n, c) in enumerate(cnn_hints)
        )

    system_prompt = (
        "You are an expert in Indian road traffic signs (Motor Vehicles Act / IRC standards).\n"
        f"Possible sign names: {classes_str}\n\n"
        f"Task: Look at the image, identify the sign, then explain it. You must generate the explanation section entirely in {language}.\n"
        "Respond in this EXACT format (no extra text before or after):\n\n"
        "SIGN: <exact sign name in English from the list above>\n"
        "---\n"
        f"<Markdown explanation. The entire explanation, INCLUDING the 4 headings, MUST be translated into {language}. Do NOT append '(in {language})' to the headings.>\n"
        "The 4 headings you must translate and use are:\n"
        "## [Translated: Sign Overview & Category]\n"
        "## [Translated: Core Meaning & Driving Rule]\n"
        "## [Translated: Defensive Driving Advice]\n"
        "## [Translated: Legal & Fine Implications in India]\n\n"
        "Rules:\n"
        "- The SIGN line must be EXACTLY one name from the list IN ENGLISH.\n"
        f"- The Markdown explanation must be completely written in {language}.\n"
        "- Keep explanation concise and use bullet points."
    )

    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
        {"type": "text", "text": f"Identify this Indian traffic sign and explain it.{hint_str}"},
    ]

    resp = groq_client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
        temperature=0.1,
        max_tokens=4096,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    sign_name = cnn_hints[0][0] if cnn_hints else "Unknown"
    explanation = raw
    if raw.startswith("SIGN:"):
        lines = raw.split("\n", 2)
        raw_name = lines[0][5:].strip()
        explanation = "\n".join(lines[2:]).strip() if len(lines) > 2 else ""
        raw_lower = raw_name.lower()
        for name in ALL_CLASSES:
            if name.lower() == raw_lower:
                sign_name = name
                break
        else:
            for name in ALL_CLASSES:
                if raw_lower in name.lower() or name.lower() in raw_lower:
                    sign_name = name
                    break
            else:
                sign_name = raw_name

    log.info(f"Vision LLM identified: {sign_name!r}")
    return sign_name, explanation

# ── Main prediction function ──────────────────────────────────────────────────
def predict_sign(pil_image, language="English"):
    if pil_image is None:
        return "No image provided.", "Please upload an image to get started.", "Unknown"

    t0 = time.time()

    top3 = cnn_top3(pil_image)
    b64 = pil_to_base64(pil_image)

    if top3:
        log.info(f"CNN top-1: {top3[0][0]} [{top3[0][1]:.1f}%]  ({time.time()-t0:.2f}s)")

    top3_md = ""
    if top3:
        top3_md = "\n\n---\n**CNN top-3 candidates:** " + " | ".join(f"`{n}` {c:.0f}%" for n, c in top3)

    if groq_client is None:
        sign_name = top3[0][0] if top3 else "Unknown"
        return sign_name, f"## {sign_name}\n\n⚠️ Groq API key not set.{top3_md}", sign_name

    try:
        sign_name, explanation = vision_identify_and_explain(b64, top3, language)
        elapsed = time.time() - t0
        log.info(f"Total prediction time: {elapsed:.2f}s")
        markdown_response = explanation + top3_md
    except Exception as exc:
        log.error(f"Vision LLM error: {exc}")
        sign_name = top3[0][0] if top3 else "Unknown"
        markdown_response = f"## {sign_name}\n\n⚠️ LLM call failed:\n```\n{exc}\n```{top3_md}"

    return sign_name, markdown_response, sign_name

# ── Chatbot function ──────────────────────────────────────────────────────────
def get_chat_response(user_message, current_sign, history):
    if not user_message or not user_message.strip():
        return "Please enter a message."
        
    if groq_client is None:
        return "⚠️ Groq API key is not configured."

    context = f"The user is asking about traffic signs. The currently identified sign on the screen is '{current_sign}'. " if current_sign and current_sign != "Unknown" else "The user is asking about Indian traffic signs."
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant expert in Indian traffic signs, driving rules, and road safety. Keep your answers concise, informative, and formatted with Markdown. " + context},
    ]
    
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
            
    messages.append({"role": "user", "content": user_message})

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        return resp.choices[0].message.content
    except Exception as exc:
        log.error(f"Chat LLM error: {exc}")
        return f"⚠️ API Error: {exc}"

# ── Streamlit UI ─────────────────────────────────────────────────────────────────
def build_ui():
    import streamlit as st

    # --- Page Config & CSS ---
    st.set_page_config(page_title="SignSense", page_icon="🚦", layout="wide")
    
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    :root {
        --clr-accent1:  #f97316;
        --clr-accent2:  #3b82f6;
    }
    
    body {
        font-family: 'Inter', system-ui, sans-serif !important;
    }
    
    #hero-banner {
        background: linear-gradient(135deg, rgba(26,10,0,0.8) 0%, rgba(13,17,23,0.8) 40%, rgba(0,16,58,0.8) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border-radius: 16px;
        padding: 2.2rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        color: #e6edf3;
    }
    
    #hero-title {
        font-size: clamp(1.8rem, 5vw, 3rem);
        font-weight: 800;
        background: linear-gradient(90deg, var(--clr-accent1), #fb923c, var(--clr-accent2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.75rem;
    }
    
    .status-pill {
        display: inline-block; padding: 6px 16px; border-radius: 999px; font-size: 0.85rem;
        font-weight: 600; background: rgba(59,130,246,0.15); color: var(--clr-accent2);
        border: 1px solid rgba(59,130,246,0.3);
        backdrop-filter: blur(4px);
    }
    .status-pill.green { background: rgba(34,197,94,0.15); color: #22c55e; border-color: rgba(34,197,94,0.3); }

    /* Animated CSS Traffic Light */
    .traffic-light {
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        justify-content: space-between;
        background: #222;
        padding: 4px;
        border-radius: 12px;
        border: 2px solid #111;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5), inset 0 0 4px rgba(255,255,255,0.1);
        height: 48px;
        width: 18px;
        vertical-align: middle;
        margin-right: 12px;
        margin-bottom: 6px;
    }
    
    .light {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #333;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.6);
        animation-duration: 4s;
        animation-iteration-count: infinite;
    }
    
    .light.red { animation-name: blink-red; }
    .light.yellow { animation-name: blink-yellow; }
    .light.green { animation-name: blink-green; }
    
    @keyframes blink-red {
        0%, 33% { background: #ef4444; box-shadow: 0 0 10px #ef4444, inset 0 1px 2px rgba(255,255,255,0.5); }
        34%, 100% { background: #333; box-shadow: inset 0 2px 4px rgba(0,0,0,0.6); }
    }
    @keyframes blink-yellow {
        0%, 33% { background: #333; box-shadow: inset 0 2px 4px rgba(0,0,0,0.6); }
        34%, 66% { background: #eab308; box-shadow: 0 0 10px #eab308, inset 0 1px 2px rgba(255,255,255,0.5); }
        67%, 100% { background: #333; box-shadow: inset 0 2px 4px rgba(0,0,0,0.6); }
    }
    @keyframes blink-green {
        0%, 66% { background: #333; box-shadow: inset 0 2px 4px rgba(0,0,0,0.6); }
        67%, 100% { background: #22c55e; box-shadow: 0 0 10px #22c55e, inset 0 1px 2px rgba(255,255,255,0.5); }
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Session State ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_sign" not in st.session_state:
        st.session_state.current_sign = "Unknown"
    if "groq_output" not in st.session_state:
        st.session_state.groq_output = "Upload a traffic sign image and click **Analyse Sign** to receive a detailed breakdown."

    # --- Header ---
    vision_ok = "Vision AI Active" if groq_client else "Vision AI Offline"
    cnn_ok = "CNN Loaded" if cnn_model else "CNN Not Loaded"
    
    st.markdown(f"""
    <div id="hero-banner">
        <h1 id="hero-title">
            <div class="traffic-light">
                <div class="light red"></div>
                <div class="light yellow"></div>
                <div class="light green"></div>
            </div>
            SignSense
        </h1>
        <p style="color: #8b949e; font-size: 1.1rem; margin: 0; line-height: 1.6;">
            Upload any Indian traffic sign — Vision AI identifies it instantly. <br/>
            Get detailed driving rules, legal implications, and ask follow-up questions in the chat!
        </p>
        <div style="margin-top:1.5rem; display:flex; gap:12px; justify-content:center; flex-wrap:wrap;">
            <span class="status-pill {'green' if groq_client else ''}">{vision_ok}</span>
            <span class="status-pill {'green' if cnn_model else ''}">{cnn_ok}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Layout ---
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.markdown("<p style='font-size: 0.8rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #8b949e;'>Upload Sign</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        lang = st.selectbox("Output Language", 
                           ["English", "Hindi", "Bengali", "Telugu", "Marathi", "Tamil", "Urdu", "Gujarati", "Kannada", "Odia", "Malayalam"], 
                           index=0, help="Choose the language for rules and guidelines.")
        
        st.markdown("<p style='font-size: 0.8rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #8b949e; margin-top: 1rem;'>Identified Sign</p>", unsafe_allow_html=True)
        st.info(f"**{st.session_state.current_sign}**", icon="ℹ️")
        
        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button("Analyse Sign", type="primary", use_container_width=True):
                if uploaded_file is not None:
                    with st.spinner("Analysing sign..."):
                        import io
                        pil_image = Image.open(io.BytesIO(uploaded_file.getvalue()))
                        sign_name, explanation, _ = predict_sign(pil_image, lang)
                        st.session_state.current_sign = sign_name
                        st.session_state.groq_output = explanation
                        st.rerun()
                else:
                    st.warning("Please upload an image first.")
        with c2:
            if st.button("Clear", use_container_width=True):
                st.session_state.current_sign = "Unknown"
                st.session_state.groq_output = "Upload a traffic sign image and click **Analyse Sign** to receive a detailed breakdown."
                st.session_state.chat_history = []
                st.rerun()

    with col2:
        tab1, tab2 = st.tabs(["AI Learning Guide", "Ask Questions"])
        
        with tab1:
            st.markdown(st.session_state.groq_output)
            
        with tab2:
            st.markdown("### Chat with SignSense")
            
            # Display chat messages
            chat_container = st.container(height=400)
            with chat_container:
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            # Voice Input
            audio_val = st.audio_input("Or ask your question using Voice")
            
            # Text Input
            if prompt := st.chat_input("Ask me anything about Indian traffic rules, fines, or driving advice..."):
                process_chat(prompt)
                
            if audio_val is not None:
                # Need to check if this specific audio has been processed to avoid re-triggering
                if "last_audio" not in st.session_state or st.session_state.last_audio != audio_val:
                    st.session_state.last_audio = audio_val
                    with st.spinner("Transcribing..."):
                        try:
                            # Save temporary file
                            with open("temp_audio.wav", "wb") as f:
                                f.write(audio_val.getbuffer())
                            
                            with open("temp_audio.wav", "rb") as file:
                                transcription = groq_client.audio.transcriptions.create(
                                    file=("temp_audio.wav", file.read()),
                                    model="whisper-large-v3-turbo",
                                )
                            process_chat(transcription.text)
                        except Exception as exc:
                            log.error(f"Whisper failed: {exc}")
                            st.error("Audio transcription failed.")

def process_chat(user_msg):
    import streamlit as st
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    
    with st.spinner("Thinking..."):
        reply = get_chat_response(user_msg, st.session_state.current_sign, st.session_state.chat_history[:-1])
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.rerun()

if __name__ == "__main__":
    build_ui()
