"""Speech-to-text abstraction layer."""

from __future__ import annotations

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


class LocalWhisperClient(SpeechToTextClient):
    """Adapter for running Whisper locally without the OpenAI API."""

    def __init__(self, *, model: str = "small") -> None:
        try:
            import whisper
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The 'openai-whisper' package is required for the Whisper client."
            ) from exc

        from whisper.audio import SAMPLE_RATE

        self._whisper = whisper.load_model(model)
        self._sample_rate = SAMPLE_RATE

    def transcribe(
        self, audio_stream: Iterable[bytes], *, language: str | None = None
    ) -> TranscriptResult:
        import numpy as np

        audio_bytes = b"".join(audio_stream)
        if not audio_bytes:
            return TranscriptResult(text="", segments=())

        sample_rate = self._sample_rate
        if audio_bytes.startswith(b"RIFF"):
            import io
            import wave

            with wave.open(io.BytesIO(audio_bytes)) as wav_file:
                sample_width = wav_file.getsampwidth()
                if sample_width != 2:
                    raise RuntimeError(
                        f"Unsupported WAV sample width: {sample_width * 8} bits"
                    )
                sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
            audio_array = np.frombuffer(frames, dtype=np.int16)
        else:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

        if audio_array.size == 0:
            return TranscriptResult(text="", segments=())

        audio = audio_array.astype(np.float32) / 32768.0

        if sample_rate != self._sample_rate:
            from whisper.audio import resample_audio

            audio = resample_audio(audio, sample_rate, self._sample_rate)

        result = self._whisper.transcribe(audio, language=language, fp16=False)

        segments: List[TranscriptSegment] = []
        for item in result.get("segments", []):
            confidence = item.get("avg_logprob")
            segments.append(
                TranscriptSegment(
                    text=item.get("text", ""),
                    start=item.get("start"),
                    end=item.get("end"),
                    confidence=float(confidence) if confidence is not None else None,
                )
            )

        return TranscriptResult(
            text=result.get("text", ""),
            segments=tuple(segments),
            language=result.get("language"),
        )
