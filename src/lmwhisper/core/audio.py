"""Audio capture abstractions.

This module exposes an abstract interface for microphone input alongside a
`PyAudio` based implementation.  The classes are intentionally lightweight so
that alternative input sources (files, network streams, etc.) can be plugged in
without touching the rest of the application.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(slots=True)
class AudioConfig:
    """Configuration shared by microphone implementations."""

    sample_rate: int = 16_000
    chunk_size: int = 1024
    channels: int = 1
    format: str = "int16"
    device_index: Optional[int] = None


class MicrophoneStream(ABC):
    """Context manager returning PCM audio chunks as ``bytes``."""

    def __init__(self, config: AudioConfig | None = None) -> None:
        self.config = config or AudioConfig()

    def __enter__(self) -> "MicrophoneStream":  # pragma: no cover - interface
        self.open()
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> None:  # pragma: no cover - interface
        self.close()

    @abstractmethod
    def open(self) -> None:
        """Allocate the underlying audio resource."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying audio resource."""

    @abstractmethod
    def chunks(self) -> Iterator[bytes]:
        """Yield PCM audio in discrete blocks."""


class PyAudioMicrophone(MicrophoneStream):
    """Microphone implementation backed by :mod:`pyaudio`.

    The dependency is imported lazily so environments without audio support can
    still use the rest of the project (e.g. automated tests).  Consumers should
    handle :class:`RuntimeError` which indicates that the backend is
    unavailable.
    """

    def __init__(self, config: AudioConfig | None = None) -> None:
        super().__init__(config)
        self._pyaudio = None
        self._stream = None

    def open(self) -> None:  # pragma: no cover - requires audio hardware
        try:
            import pyaudio
        except ImportError as exc:  # pragma: no cover - requires optional dep
            raise RuntimeError(
                "PyAudio is not installed. Install it to use the microphone backend."
            ) from exc

        self._pyaudio = pyaudio.PyAudio()
        format_map = {
            "int16": pyaudio.paInt16,
            "float32": pyaudio.paFloat32,
        }
        try:
            fmt = format_map[self.config.format]
        except KeyError as exc:
            raise RuntimeError(f"Unsupported audio format: {self.config.format}") from exc

        self._stream = self._pyaudio.open(
            format=fmt,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_size,
            input_device_index=self.config.device_index,
        )

    def close(self) -> None:  # pragma: no cover - requires audio hardware
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None

    def chunks(self) -> Iterator[bytes]:  # pragma: no cover - requires audio hardware
        if self._stream is None:
            raise RuntimeError("Microphone stream is not opened")
        while True:
            yield self._stream.read(self.config.chunk_size, exception_on_overflow=False)


class FileAudioStream(MicrophoneStream):
    """Fake microphone that iterates over PCM data from a file-like object."""

    def __init__(self, data: bytes, config: AudioConfig | None = None) -> None:
        super().__init__(config)
        self._data = data

    def open(self) -> None:
        # Nothing to do for in-memory audio.
        return None

    def close(self) -> None:
        return None

    def chunks(self) -> Iterator[bytes]:
        chunk_size = self.config.chunk_size
        for index in range(0, len(self._data), chunk_size):
            yield self._data[index : index + chunk_size]
