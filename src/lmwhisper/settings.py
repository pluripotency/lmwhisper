"""Configuration loading utilities for the application."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

ALLOWED_WHISPER_MODELS = {"tiny", "base", "small", "medium", "large"}


@dataclass(slots=True)
class WhisperSettings:
    model: str = "small"


@dataclass(slots=True)
class LMStudioSettings:
    base_url: str = "http://localhost:1234/v1"
    model: str = "lmstudio"
    temperature: float = 0.7
    max_tokens: int | None = None


@dataclass(slots=True)
class LoggingSettings:
    output_dir: Path = Path("logs")


@dataclass(slots=True)
class AppSettings:
    whisper: WhisperSettings
    lmstudio: LMStudioSettings
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AppSettings":
        whisper_data = data.get("whisper") or {}
        lmstudio_data = data.get("lmstudio") or {}
        logging_data = data.get("logging") or {}

        whisper_model = whisper_data.get("model", "small")
        if whisper_model not in ALLOWED_WHISPER_MODELS:
            raise ValueError(
                "Invalid Whisper model configured. Choose one of: "
                + ", ".join(sorted(ALLOWED_WHISPER_MODELS))
            )

        whisper = WhisperSettings(model=whisper_model)

        lmstudio = LMStudioSettings(
            base_url=lmstudio_data.get("base_url", "http://localhost:1234/v1"),
            model=lmstudio_data.get("model", "lmstudio"),
            temperature=float(lmstudio_data.get("temperature", 0.7)),
            max_tokens=lmstudio_data.get("max_tokens"),
        )

        logging = LoggingSettings(output_dir=Path(logging_data.get("output_dir", "logs")))

        return cls(whisper=whisper, lmstudio=lmstudio, logging=logging)


def load_settings(path: Path) -> AppSettings:
    with path.open("rb") as fp:
        data = tomllib.load(fp)
    return AppSettings.from_mapping(data)
