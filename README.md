# linux-voice

Voice-to-text dictation tool for Linux (X11) using OpenAI Whisper.

Hold Ctrl+Space to record speech, release to transcribe and type into the focused window.

## Requirements

- Python 3.11+
- Linux with X11 (Wayland has limited support)
- PulseAudio/PipeWire
- Working microphone
- OpenAI API key

## Installation

```bash
# System dependencies (Arch Linux)
sudo pacman -S xdotool

# Python dependencies
pip install pynput sounddevice numpy openai
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

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

## Privacy and Security

**Audio is sent to OpenAI:** All recorded speech is transmitted to OpenAI's Whisper API for transcription. See [OpenAI's data usage policies](https://openai.com/policies/api-data-usage-policies).

**Text is typed into the focused window:** Be careful not to dictate sensitive information while password fields or sensitive applications are focused. The transcribed text is typed using xdotool into whatever window has focus.

## Cost

OpenAI Whisper API costs $0.006 per minute of audio. A typical 10-second dictation costs ~$0.001.

## License

MIT
