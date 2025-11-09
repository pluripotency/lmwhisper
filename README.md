# lmwhisper

Modular command line application that combines the open source Whisper model for
transcription with LM Studio for local large language model responses.  Conversation turns are
persisted to TOML files for further processing.

## Features

- Audio input abstraction with interchangeable backends (PyAudio microphone or
  file-based sources).
- Whisper transcription client decoupled behind an interface for easy engine
  substitution.
- Conversation manager coordinating prompts and LM Studio responses.
- TOML persistence that captures conversation metadata for later analysis.

## Configuration

Create a configuration file based on `config.example.toml` and fill in your
Whisper model choice and LM Studio settings.

```toml
[whisper]
model = "small" # tiny, base, small, medium, or large

[lmstudio]
base_url = "http://localhost:1234/v1"
model = "YourModel"
temperature = 0.7
max_tokens = 256

[logging]
output_dir = "logs"
```

## Usage

```bash
uv run python -m lmwhisper.main chat --config config.toml
```

Optional flags:

- `--audio-file`: Use a PCM WAV file instead of the microphone.
- `--duration`: Seconds to capture from the microphone (default: 5).
- `--system-prompt`: Provide a system instruction prepended to the dialogue.
- `--conversation-id`: Supply a deterministic identifier for the saved TOML
  file.

Each execution transcribes the captured audio, queries LM Studio, displays the
reply, and persists the full turn to a TOML file within the configured log
folder.
