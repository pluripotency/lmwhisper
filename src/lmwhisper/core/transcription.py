"""Speech-to-text abstraction layer."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence


@dataclass(slots=True)
class TranscriptSegment:
    text: str
    start: float | None = None
    end: float | None = None
    confidence: float | None = None


@dataclass(slots=True)
class TranscriptResult:
    text: str
    segments: Sequence[TranscriptSegment] = field(default_factory=tuple)
    language: str | None = None


class SpeechToTextClient(ABC):
    """Abstract whisper-like client."""

    @abstractmethod
    def transcribe(
        self, audio_stream: Iterable[bytes], *, language: str | None = None
    ) -> TranscriptResult:
        """Convert the provided audio chunks to text."""


class OpenAIWhisperClient(SpeechToTextClient):
    """Adapter for the official OpenAI Whisper API."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "whisper-1",
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The 'openai' package is required for the Whisper client."
            ) from exc

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def transcribe(
        self, audio_stream: Iterable[bytes], *, language: str | None = None
    ) -> TranscriptResult:
        audio_bytes = b"".join(audio_stream)
        if not audio_bytes:
            return TranscriptResult(text="", segments=())

        buffer = io.BytesIO(audio_bytes)
        buffer.name = "speech.wav"

        response = self._client.audio.transcriptions.create(
            model=self._model,
            file=(buffer.name, buffer.read(), "audio/wav"),
            language=language,
            response_format="verbose_json",
        )

        segments: List[TranscriptSegment] = []
        for item in response.segments or []:
            segments.append(
                TranscriptSegment(
                    text=item.get("text", ""),
                    start=item.get("start"),
                    end=item.get("end"),
                    confidence=item.get("confidence"),
                )
            )

        return TranscriptResult(
            text=response.text or "",
            segments=tuple(segments),
            language=response.language,
        )
