"""
LLM reasoning module using a local Ollama server running qwen2.5:0.5b.
No API key, no internet call — everything stays on your machine.
"""

import ollama
from config import OLLAMA_MODEL, OLLAMA_HOST

_client = ollama.Client(host=OLLAMA_HOST)

SYSTEM_PROMPT = (
    "You are a concise multimodal assistant. You are given a user's spoken "
    "question and a list of objects detected in their camera feed. Answer "
    "naturally and briefly, grounding your answer in the detected objects "
    "when relevant. If the objects are not relevant to the question, ignore "
    "them and just answer the question normally."
)


def query_llm(user_text: str, detected_objects_context: str) -> str:
    """
    Send the transcribed query + CV context to the local LLM and get a response.

    Args:
        user_text: transcribed speech from the STT module
        detected_objects_context: comma-separated object labels from the CV module

    Returns:
        the LLM's generated text response
    """
    prompt = (
        f"Detected objects in the camera view: {detected_objects_context}\n"
        f"User's question: {user_text}\n"
        f"Answer:"
    )

    response = _client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    answer = response["message"]["content"].strip()
    print(f"[LLM] Response: {answer}")
    return answer


if __name__ == "__main__":
    # quick manual test
    test_answer = query_llm(
        user_text="What's on the table?",
        detected_objects_context="laptop, coffee cup, notebook",
    )
    print(test_answer)
