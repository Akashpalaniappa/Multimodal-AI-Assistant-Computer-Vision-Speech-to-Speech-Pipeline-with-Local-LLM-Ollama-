"""
Speech-to-Text module using OpenAI Whisper (runs 100% locally, no API key needed).
"""

import whisper
from config import WHISPER_MODEL_SIZE

_model = None


def load_stt_model():
    """Load Whisper model once and cache it (avoids reloading on every request)."""
    global _model
    if _model is None:
        print(f"[STT] Loading Whisper model: {WHISPER_MODEL_SIZE}")
        _model = whisper.load_model(WHISPER_MODEL_SIZE)
    return _model


def transcribe_audio(audio_path: str) -> str:
    """
    Convert a speech audio file (wav/mp3/m4a) into text.

    Args:
        audio_path: path to the audio file on disk

    Returns:
        transcribed text (str)
    """
    model = load_stt_model()
    result = model.transcribe(audio_path, fp16=False)
    text = result.get("text", "").strip()
    print(f"[STT] Transcribed: {text}")
    return text


if __name__ == "__main__":
    # quick manual test: python modules/stt_module.py path/to/audio.wav
    import sys
    if len(sys.argv) > 1:
        print(transcribe_audio(sys.argv[1]))
    else:
        print("Usage: python stt_module.py <audio_file_path>")
