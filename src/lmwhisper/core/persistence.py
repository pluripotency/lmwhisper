"""TOML based persistence utilities."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from lmwhisper.core.conversation import Message


@dataclass(slots=True)
class ConversationTurn:
    user: Message
    assistant: Message


@dataclass(slots=True)
class TomlLoggerConfig:
    directory: Path

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)


class TomlLogger:
    """Append-only TOML logging for conversation transcripts."""

    def __init__(self, config: TomlLoggerConfig) -> None:
        self._config = config

    def _file_path(self, conversation_id: str) -> Path:
        return self._config.directory / f"{conversation_id}.toml"

    def write(
        self,
        conversation_id: str,
        *,
        system: Sequence[Message] = (),
        turns: Sequence[ConversationTurn],
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        """Persist a complete conversation to TOML."""

        import tomli_w

        doc: dict[str, object] = {
            "conversation": {
                "id": conversation_id,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "metadata": dict(metadata or {}),
            },
            "messages": [],
        }

        for message in system:
            doc["messages"].append(_message_to_dict(message))

        for turn in turns:
            doc["messages"].append(_message_to_dict(turn.user))
            doc["messages"].append(_message_to_dict(turn.assistant))

        path = self._file_path(conversation_id)
        with path.open("wb") as fp:
            tomli_w.dump(doc, fp)
        return path


def _message_to_dict(message: Message) -> dict[str, object]:
    payload: dict[str, object] = {
        "role": message.role,
        "content": message.content,
        "timestamp": message.timestamp.isoformat(),
    }
    if message.metadata:
        payload["metadata"] = dict(message.metadata)
    return payload
