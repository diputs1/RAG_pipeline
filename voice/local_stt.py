"""
Local speech-to-text helpers.

Default backend is faster-whisper because it is easy to run from Python,
works offline after the model is downloaded, and supports Vietnamese well.
"""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def _get_faster_whisper_model(model_size: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing faster-whisper. Install it with: pip install faster-whisper"
        ) from exc
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_audio_local(
    audio_bytes: bytes,
    *,
    filename: str = "question.wav",
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "vi",
    beam_size: int = 1,
    vad_filter: bool = False,
) -> dict:
    """Transcribe audio bytes locally with faster-whisper."""
    if not audio_bytes:
        raise ValueError("Audio input is empty.")

    suffix = Path(filename or "question.wav").suffix or ".wav"
    model = _get_faster_whisper_model(model_size, device, compute_type)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        segments, info = model.transcribe(
            tmp.name,
            language=language or None,
            beam_size=beam_size,
            vad_filter=vad_filter,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())

    return {
        "text": text.strip(),
        "language": getattr(info, "language", language),
        "language_probability": getattr(info, "language_probability", None),
        "backend": "faster-whisper",
        "model_size": model_size,
        "device": device,
        "compute_type": compute_type,
        "beam_size": beam_size,
        "vad_filter": vad_filter,
    }
