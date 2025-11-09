"""Conversation orchestration and LLM integration."""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Mapping, MutableMapping, Optional, Sequence


@dataclass(slots=True)
class Message:
    role: str
    content: str
    timestamp: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    metadata: MutableMapping[str, object] = field(default_factory=dict)

    def as_chat_dict(self) -> Mapping[str, object]:
        data: dict[str, object] = {"role": self.role, "content": self.content}
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass(slots=True)
class GenerationConfig:
    temperature: float = 0.7
    max_tokens: int | None = None
    system_prompt: str | None = None


class LLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        *,
        config: GenerationConfig | None = None,
    ) -> Message:
        """Generate an assistant message from the given history."""


class LMStudioClient(LLMClient):
    """HTTP adapter for the LM Studio local server."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        timeout: float = 30.0,
    ) -> None:
        import httpx

        self._client = httpx.Client(base_url=base_url, timeout=timeout)
        self._model = model

    def generate(
        self,
        messages: Sequence[Message],
        *,
        config: GenerationConfig | None = None,
    ) -> Message:
        config = config or GenerationConfig()
        payload = {
            "model": self._model,
            "messages": [
                {"role": msg.role, "content": msg.content} for msg in messages
            ],
            "temperature": config.temperature,
        }
        if config.max_tokens is not None:
            payload["max_tokens"] = config.max_tokens
        if config.system_prompt:
            payload.setdefault("messages", [])
            payload["messages"].insert(0, {"role": "system", "content": config.system_prompt})

        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]["message"]
        assistant = Message(role=choice.get("role", "assistant"), content=choice.get("content", ""))
        usage = data.get("usage")
        if isinstance(usage, dict):
            assistant.metadata["usage"] = usage
        return assistant


@dataclass
class ConversationManager:
    llm: LLMClient
    config: GenerationConfig = field(default_factory=GenerationConfig)
    messages: List[Message] = field(default_factory=list)

    def add_user_message(
        self,
        content: str,
        *,
        metadata: Optional[MutableMapping[str, object]] = None,
    ) -> Message:
        message = Message(role="user", content=content, metadata=metadata or {})
        self.messages.append(message)
        return message

    def add_system_message(self, content: str) -> Message:
        message = Message(role="system", content=content)
        self.messages.append(message)
        return message

    def generate_reply(self) -> Message:
        assistant = self.llm.generate(self.messages, config=self.config)
        self.messages.append(assistant)
        return assistant

    def history(self) -> Sequence[Message]:
        return tuple(self.messages)
