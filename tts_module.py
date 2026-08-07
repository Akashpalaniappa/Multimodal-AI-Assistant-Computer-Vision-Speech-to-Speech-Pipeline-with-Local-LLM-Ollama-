"""
Text-to-Speech module using pyttsx3 — fully offline, works on Windows/Linux/Mac
without any API key. (Swap for Coqui TTS later if you want more natural voices.)
"""

import pyttsx3
from config import TTS_RATE, TTS_VOLUME


def text_to_speech_file(text: str, output_path: str = "response.mp3") -> str:
    """
    Convert text into a speech audio file saved on disk.

    Args:
        text: the LLM's response text
        output_path: where to save the generated audio

    Returns:
        path to the saved audio file
    """
    engine = pyttsx3.init()
    engine.setProperty("rate", TTS_RATE)
    engine.setProperty("volume", TTS_VOLUME)
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    print(f"[TTS] Saved audio response to: {output_path}")
    return output_path


def speak_directly(text: str):
    """Speak text out loud immediately (useful for local CLI testing)."""
    engine = pyttsx3.init()
    engine.setProperty("rate", TTS_RATE)
    engine.setProperty("volume", TTS_VOLUME)
    engine.say(text)
    engine.runAndWait()


if __name__ == "__main__":
    speak_directly("Hello, this is your local multimodal assistant speaking.")
