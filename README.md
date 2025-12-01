# linux-voice

Voice-to-text dictation tool for Linux (X11) using OpenAI Whisper.

Hold Ctrl+Space to record speech, release to transcribe and type into the focused window.

## Installation

```bash
# System dependencies (Arch Linux)
sudo pacman -S xdotool

# Python dependencies
pip install pynput sounddevice numpy openai
```

Requires `OPENAI_API_KEY` environment variable.

## Usage

```bash
python linux-voice.py
```

- **Hold mode** (default): Hold Ctrl+Space while speaking, release to transcribe
- **Toggle mode**: Press Ctrl+Space to start recording, press again to stop

```bash
LINUX_VOICE_MODE=toggle python linux-voice.py
```

## Configuration

Create `~/.config/linux-voice/config.toml` to customize:

```toml
[hotkey]
key = "space"
modifiers = ["ctrl"]
mode = "hold"  # or "toggle"

[audio]
sample_rate = 48000

[transcription]
language = "en"
```

## Requirements

- Linux with X11
- PulseAudio/PipeWire
- Working microphone
- OpenAI API key
