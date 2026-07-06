"""
ElevenLabs batch speech-to-text helper.

Uses the HTTP API directly so the app does not depend on SDK internals.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import requests


STT_ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"


def transcribe_audio(
    audio_bytes: bytes,
    *,
    api_key: str,
    filename: str = "question.wav",
    mime_type: str = "audio/wav",
    model_id: str = "scribe_v2",
    language_code: str = "vi",
) -> dict[str, Any]:
    """Return ElevenLabs transcription response as a dict."""
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY is required for speech-to-text.")
    if not audio_bytes:
        raise ValueError("Audio input is empty.")

    files = {"file": (filename, BytesIO(audio_bytes), mime_type)}
    data = {
        "model_id": model_id,
        "language_code": language_code,
        "tag_audio_events": "false",
    }
    response = requests.post(
        STT_ENDPOINT,
        headers={"xi-api-key": api_key},
        data=data,
        files=files,
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    payload.setdefault("text", "")
    return payload

