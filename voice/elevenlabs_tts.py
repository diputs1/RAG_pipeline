"""
ElevenLabs text-to-speech helper for reading final RAG answers.
"""

from __future__ import annotations

from typing import Any

import requests


DEFAULT_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam premade voice; editable in UI.


def synthesize_speech(
    text: str,
    *,
    api_key: str,
    voice_id: str = DEFAULT_VOICE_ID,
    model_id: str = "eleven_flash_v2_5",
    output_format: str = "mp3_22050_32",
    stability: float = 0.35,
    similarity_boost: float = 0.85,
    speed: float = 1.0,
) -> bytes:
    """Convert text to MP3 bytes using ElevenLabs TTS HTTP API."""
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY is required for text-to-speech.")
    clean_text = (text or "").strip()
    if not clean_text:
        raise ValueError("Text is empty.")
    if len(clean_text) > 4500:
        clean_text = clean_text[:4500].rsplit(" ", 1)[0] + "..."

    endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload: dict[str, Any] = {
        "text": clean_text,
        "model_id": model_id,
        "voice_settings": {
            "stability": float(stability),
            "similarity_boost": float(similarity_boost),
            "speed": float(speed),
        },
    }
    response = requests.post(
        endpoint,
        params={"output_format": output_format},
        headers={
            "xi-api-key": api_key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        },
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    return response.content
