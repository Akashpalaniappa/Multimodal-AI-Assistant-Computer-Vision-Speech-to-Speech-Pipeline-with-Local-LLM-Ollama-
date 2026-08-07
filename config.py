"""
Central configuration for the Multimodal AI Assistant.
Edit these values to match your local setup.
"""

# ---- Ollama / LLM ----
OLLAMA_MODEL = "qwen2.5:0.5b"     # the model you already pulled locally
OLLAMA_HOST = "http://localhost:11434"

# ---- Whisper (STT) ----
WHISPER_MODEL_SIZE = "base"       # tiny / base / small -> base is a good speed/accuracy tradeoff

# ---- YOLO (CV) ----
YOLO_MODEL_PATH = "yolov8n.pt"    # nano model, auto-downloads on first run
YOLO_CONFIDENCE = 0.45

# ---- TTS ----
TTS_RATE = 175                    # words per minute for pyttsx3
TTS_VOLUME = 1.0

# ---- Folders ----
UPLOAD_DIR = "temp_uploads"
