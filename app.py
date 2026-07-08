"""
Indian Traffic Sign Learning Assistant – Gradio App (Vision LLM + CNN + Chat)
=============================================================================
Run:
    python app/app.py
"""

import os
import io
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

# ── Load CNN model ────────────────────────────────────────────────────────────
def load_keras_model():
    try:
        import tensorflow as tf
    except ImportError:
        log.error("tensorflow not installed.")
        return None

    try:
        _orig = tf.keras.layers.Dense.from_config
        @classmethod
        def _patched(cls, config):
            config.pop("quantization_config", None)
            if isinstance(config.get("config"), dict):
                config["config"].pop("quantization_config", None)
            return _orig(config)
        tf.keras.layers.Dense.from_config = _patched
    except Exception:
        pass

    class CompatDense(tf.keras.layers.Dense):
        @classmethod
        def from_config(cls, config):
            config.pop("quantization_config", None)
            if isinstance(config.get("config"), dict):
                config["config"].pop("quantization_config", None)
            return super().from_config(config)

    for path in [
        MODELS_DIR / "traffic_model.h5",
        MODELS_DIR / "traffic_model.keras",
        MODELS_DIR / "best_checkpoint.h5",
        MODELS_DIR / "traffic_classifier_model.h5",
    ]:
        if not path.exists():
            continue
        try:
            m = tf.keras.models.load_model(str(path), custom_objects={"Dense": CompatDense})
            log.info(f"CNN loaded: {path.name}  input={m.input_shape}  classes={m.output_shape[-1]}")
            return m
        except Exception as exc:
            log.error(f"Failed to load {path.name}: {exc}")
    return None

def load_class_map() -> dict[str, str]:
    if not CLASS_MAP_PATH.exists():
        return {}
    with open(CLASS_MAP_PATH, encoding="utf-8") as f:
        return json.load(f)

cnn_model = load_keras_model()
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
        import tensorflow as tf
        h, w = cnn_model.input_shape[1], cnn_model.input_shape[2]
        img  = composite_on_white(pil_image).convert("RGB").resize((w, h))
        arr  = np.array(img, dtype=np.float32)
        is_gtsrb = cnn_model.output_shape[-1] == 43
        if PREPROCESS == "mobilenet_v2" and not is_gtsrb:
            arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        else:
            arr = arr / 255.0
        arr   = np.expand_dims(arr, 0)
        preds = cnn_model.predict(arr, verbose=0)[0]
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
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
        temperature=0.1,
        max_tokens=900,
    )
    raw = resp.choices[0].message.content.strip()

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

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        cnn_future = pool.submit(cnn_top3, pil_image)
        b64_future = pool.submit(pil_to_base64, pil_image)
        top3  = cnn_future.result()
        b64   = b64_future.result()

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
def chat_with_llm(user_message, audio_file, history, current_sign):
    if history is None:
        history = []
        
    # Handle Audio Transcription if audio_file is provided
    if audio_file is not None:
        try:
            with open(audio_file, "rb") as file:
                transcription = groq_client.audio.transcriptions.create(
                    file=(audio_file, file.read()),
                    model="whisper-large-v3-turbo",
                )
            user_message = transcription.text
        except Exception as exc:
            log.error(f"Whisper failed: {exc}")
            user_message = "⚠️ Audio transcription failed."

    if not user_message or not user_message.strip():
        return history, "", None
        
    if groq_client is None:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "⚠️ Groq API key is not configured."})
        return history, "", None

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
        reply = resp.choices[0].message.content
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply})
    except Exception as exc:
        log.error(f"Chat LLM error: {exc}")
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": f"⚠️ API Error: {exc}"})
        
    return history, "", None

# ── Gradio UI ─────────────────────────────────────────────────────────────────
_APP_CSS = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    :root {
        --clr-bg:       #0d1117;
        --clr-surface:  rgba(22, 27, 34, 0.7);
        --clr-card:     rgba(28, 35, 48, 0.6);
        --clr-border:   rgba(48, 54, 61, 0.5);
        --clr-accent1:  #f97316;
        --clr-accent2:  #3b82f6;
        --clr-green:    #22c55e;
        --clr-text:     #e6edf3;
        --clr-muted:    #8b949e;
        --radius:       16px;
        --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        --glass-border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    body, .gradio-container {
        background: linear-gradient(135deg, #0d1117 0%, #0a0a0f 50%, #111424 100%) !important;
        background-attachment: fixed !important;
        color: var(--clr-text) !important;
        font-family: 'Inter', system-ui, sans-serif !important;
    }
    
    /* Glassmorphism Containers */
    .glass-panel {
        background: var(--clr-surface) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: var(--glass-border) !important;
        box-shadow: var(--glass-shadow) !important;
        border-radius: var(--radius) !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-panel:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.45) !important;
    }
    
    #hero-banner {
        background: linear-gradient(135deg, rgba(26,10,0,0.8) 0%, rgba(13,17,23,0.8) 40%, rgba(0,16,58,0.8) 100%);
        backdrop-filter: blur(16px);
        border: var(--glass-border);
        box-shadow: var(--glass-shadow);
        border-radius: var(--radius);
        padding: 2.2rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    #hero-title {
        font-size: clamp(1.8rem, 5vw, 3rem);
        font-weight: 800;
        background: linear-gradient(90deg, var(--clr-accent1), #fb923c, var(--clr-accent2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.75rem;
        animation: pulseGradient 6s infinite alternate;
    }
    
    #hero-subtitle { color: var(--clr-muted); font-size: 1.1rem; margin: 0; line-height: 1.6; }
    
    #predict-btn {
        background: linear-gradient(135deg, var(--clr-accent1), #ea6c00) !important;
        border: none !important; border-radius: 12px !important; color: white !important;
        font-weight: 700 !important; font-size: 1.1rem !important; padding: 0.85rem 2rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(249, 115, 22, 0.4) !important;
    }
    
    #predict-btn:hover { 
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(249, 115, 22, 0.6) !important; 
    }
    
    #predict-btn:active {
        transform: translateY(1px) scale(0.98) !important;
    }
    
    #pred-label textarea, #pred-label input {
        background: rgba(22, 27, 34, 0.5) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(34, 197, 94, 0.3) !important;
        border-radius: 12px !important; color: #22c55e !important; font-size: 1.15rem !important;
        font-weight: 700 !important;
        text-align: center !important;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.2) !important;
    }
    
    #groq-output, #chat-container {
        background: rgba(28, 35, 48, 0.4); 
        backdrop-filter: blur(10px);
        border: var(--glass-border);
        border-radius: var(--radius); 
        padding: 1.5rem; 
        line-height: 1.75;
        animation: fadeInUp 0.6s ease-out backwards;
    }
    
    .status-pill {
        display: inline-block; padding: 6px 16px; border-radius: 999px; font-size: 0.85rem;
        font-weight: 600; background: rgba(59,130,246,0.15); color: var(--clr-accent2);
        border: 1px solid rgba(59,130,246,0.3);
        backdrop-filter: blur(4px);
        transition: transform 0.2s ease;
    }
    .status-pill:hover { transform: scale(1.05); }
    .status-pill.green { background: rgba(34,197,94,0.15); color: var(--clr-green); border-color: rgba(34,197,94,0.3); }
    
    .section-label {
        font-size: 0.8rem; font-weight: 700; letter-spacing: 1.5px;
        text-transform: uppercase; color: var(--clr-muted); margin-bottom: 0.75rem;
        display: flex; align-items: center; gap: 8px;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseGradient {
        0% { filter: hue-rotate(0deg); }
        100% { filter: hue-rotate(15deg); }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); border-radius: 4px; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25); }

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
"""

def build_ui():
    import gradio as gr

    vision_ok = "Vision AI Active" if groq_client else "Vision AI Offline"
    cnn_ok    = "CNN Loaded" if cnn_model else "CNN Not Loaded"

    with gr.Blocks(title="SignSense", theme=gr.themes.Base()) as demo:
        demo.load(None, js="() => { document.head.insertAdjacentHTML('beforeend', `<style>" + _APP_CSS.replace("`", "\\`") + "</style>`) }")
        
        # State to keep track of the current identified sign for the chatbot context
        current_sign_state = gr.State("Unknown")

        gr.HTML(f"""
        <div id="hero-banner">
            <h1 id="hero-title">
                <div class="traffic-light">
                    <div class="light red"></div>
                    <div class="light yellow"></div>
                    <div class="light green"></div>
                </div>
                SignSense
            </h1>
            <p id="hero-subtitle">
                Upload any Indian traffic sign — Vision AI identifies it instantly. <br/>
                Get detailed driving rules, legal implications, and ask follow-up questions in the chat!
            </p>
            <div style="margin-top:1.5rem; display:flex; gap:12px; justify-content:center; flex-wrap:wrap;">
                <span class="status-pill green">{vision_ok}</span>
                <span class="status-pill">{cnn_ok}</span>
            </div>
        </div>
        """)

        with gr.Row(equal_height=False):
            # Left Column (Input & Prediction)
            with gr.Column(scale=1, min_width=320, elem_classes="glass-panel", elem_id="left-col"):
                gr.HTML('<div style="padding: 1rem;"><p class="section-label">Upload Sign</p></div>')
                image_input = gr.Image(type="pil", height=280, show_label=False)
                
                lang_dropdown = gr.Dropdown(
                    choices=["English", "Hindi", "Bengali", "Telugu", "Marathi", "Tamil", "Urdu", "Gujarati", "Kannada", "Odia", "Malayalam"],
                    value="English",
                    label="Output Language",
                    info="Choose the language for rules and guidelines."
                )

                gr.HTML('<div style="padding: 1rem 1rem 0;"><p class="section-label">Identified Sign</p></div>')
                pred_label = gr.Textbox(
                    placeholder="Sign name will appear here…",
                    interactive=False,
                    elem_id="pred-label",
                    show_label=False,
                )

                with gr.Row(elem_classes="glass-panel", elem_id="btn-row"):
                    predict_btn = gr.Button("Analyse Sign", variant="primary", elem_id="predict-btn", scale=3)
                    clear_btn   = gr.Button("Clear", scale=1)

            # Right Column (Tabs for Guide and Chat)
            with gr.Column(scale=1, elem_classes="glass-panel"):
                with gr.Tabs():
                    with gr.TabItem("AI Learning Guide"):
                        groq_output = gr.Markdown(
                            value="Upload a traffic sign image and click **Analyse Sign** to receive a detailed breakdown.",
                            elem_id="groq-output",
                        )
                    
                    with gr.TabItem("Ask Questions"):
                        chatbot = gr.Chatbot(
                            elem_id="chat-container", 
                            height=400,
                            placeholder="Ask me anything about Indian traffic rules, fines, or driving advice..."
                        )
                        with gr.Row():
                            chat_input = gr.Textbox(
                                placeholder="Type your question here and press Enter...",
                                show_label=False,
                                scale=4
                            )
                            chat_submit = gr.Button("Send", variant="primary", scale=1)
                        with gr.Row():
                            audio_input = gr.Audio(
                                sources=["microphone"],
                                type="filepath",
                                label="Or ask your question using Voice",
                            )

        # Event listeners for Prediction
        predict_btn.click(
            fn=predict_sign,
            inputs=[image_input, lang_dropdown],
            outputs=[pred_label, groq_output, current_sign_state]
        )
        image_input.upload(
            fn=predict_sign, 
            inputs=[image_input, lang_dropdown],
            outputs=[pred_label, groq_output, current_sign_state]
        )
        clear_btn.click(
            fn=lambda: (None, "", "Upload a traffic sign image and click **Analyse Sign** to receive a detailed breakdown.", "Unknown", "English", []),
            inputs=None,
            outputs=[image_input, pred_label, groq_output, current_sign_state, lang_dropdown, chatbot],
        )

        # Event listeners for Chat
        chat_input.submit(
            fn=chat_with_llm,
            inputs=[chat_input, audio_input, chatbot, current_sign_state],
            outputs=[chatbot, chat_input, audio_input]
        )
        chat_submit.click(
            fn=chat_with_llm,
            inputs=[chat_input, audio_input, chatbot, current_sign_state],
            outputs=[chatbot, chat_input, audio_input]
        )
        audio_input.stop_recording(
            fn=chat_with_llm,
            inputs=[chat_input, audio_input, chatbot, current_sign_state],
            outputs=[chatbot, chat_input, audio_input]
        )

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False, inbrowser=True)
