"""Command line entry point for the lmwhisper application."""

from __future__ import annotations

import itertools
import uuid
from pathlib import Path
from typing import Sequence

import typer

from lmwhisper.core.audio import AudioConfig, FileAudioStream, MicrophoneStream, PyAudioMicrophone
from lmwhisper.core.conversation import ConversationManager, GenerationConfig, LMStudioClient
from lmwhisper.core.persistence import ConversationTurn, TomlLogger, TomlLoggerConfig
from lmwhisper.core.transcription import OpenAIWhisperClient, TranscriptResult
from lmwhisper.settings import load_settings


app = typer.Typer(add_completion=False, help=__doc__)


def _resolve_audio_source(audio_file: Path | None, config: AudioConfig) -> MicrophoneStream:
    if audio_file is not None:
        return FileAudioStream(audio_file.read_bytes(), config=config)
    return PyAudioMicrophone(config=config)


def _collect_audio(
    stream: MicrophoneStream, *, duration: float | None = None
) -> Sequence[bytes]:
    if duration is None:
        return list(stream.chunks())

    required_chunks = max(1, int(duration * stream.config.sample_rate / stream.config.chunk_size))
    return list(itertools.islice(stream.chunks(), required_chunks))


def _transcript_metadata(transcript: TranscriptResult) -> dict:
    return {
        "language": transcript.language,
        "segments": [
            {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
                "confidence": segment.confidence,
            }
            for segment in transcript.segments
        ],
    }


@app.command()
def chat(
    config: Path = typer.Option(..., "--config", exists=True, help="Path to the application TOML configuration."),
    audio_file: Path | None = typer.Option(None, "--audio-file", exists=True, help="Optional PCM WAV file used instead of the microphone."),
    conversation_id: str | None = typer.Option(None, "--conversation-id", help="Identifier for the conversation session."),
    duration: float | None = typer.Option(5.0, "--duration", help="Seconds to capture from the microphone (ignored for audio files)."),
    system_prompt: str | None = typer.Option(None, "--system-prompt", help="Optional system prompt injected before the conversation."),
) -> None:
    """Run a single-shot transcription and LM Studio completion."""

    settings = load_settings(config)

    audio_config = AudioConfig()
    microphone = _resolve_audio_source(audio_file, audio_config)

    whisper = OpenAIWhisperClient(
        settings.whisper.api_key,
        model=settings.whisper.model,
        base_url=settings.whisper.base_url,
    )

    llm_client = LMStudioClient(
        base_url=settings.lmstudio.base_url,
        model=settings.lmstudio.model,
    )

    generation = GenerationConfig(
        temperature=settings.lmstudio.temperature,
        max_tokens=settings.lmstudio.max_tokens,
        system_prompt=system_prompt,
    )

    manager = ConversationManager(llm=llm_client, config=generation)
    if system_prompt:
        manager.add_system_message(system_prompt)

    conv_id = conversation_id or uuid.uuid4().hex

    with microphone:
        audio_chunks = _collect_audio(microphone, duration=duration if audio_file is None else None)

    transcript = whisper.transcribe(audio_chunks)

    if not transcript.text:
        typer.echo("No speech detected.")
        raise typer.Exit(code=1)

    user_message = manager.add_user_message(
        transcript.text,
        metadata={"transcript": _transcript_metadata(transcript)},
    )

    assistant_message = manager.generate_reply()

    typer.echo("User: " + user_message.content)
    typer.echo("Assistant: " + assistant_message.content)

    logger = TomlLogger(TomlLoggerConfig(directory=settings.logging.output_dir))
    path = logger.write(
        conv_id,
        system=[msg for msg in manager.history() if msg.role == "system"],
        turns=[ConversationTurn(user=user_message, assistant=assistant_message)],
        metadata={"transcript": _transcript_metadata(transcript)},
    )

    typer.echo(f"Conversation saved to {path}")


def run() -> None:
    app()
