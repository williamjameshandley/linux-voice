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

### Ubuntu/Debian

```bash
# System dependencies
sudo apt install xdotool ffmpeg python3-pip python3-venv

# Create virtual environment (recommended)
python3 -m venv ~/.local/share/linux-voice
source ~/.local/share/linux-voice/bin/activate

# Install Python packages
pip install pynput sounddevice numpy openai

# Download and install script
curl -o ~/.local/bin/linux-voice https://raw.githubusercontent.com/williamjameshandley/linux-voice/main/linux-voice.py
chmod +x ~/.local/bin/linux-voice
```

### Fedora

```bash
# System dependencies
sudo dnf install xdotool ffmpeg python3-pip

# Create virtual environment (recommended)
python3 -m venv ~/.local/share/linux-voice
source ~/.local/share/linux-voice/bin/activate

# Install Python packages
pip install pynput sounddevice numpy openai

# Download and install script
curl -o ~/.local/bin/linux-voice https://raw.githubusercontent.com/williamjameshandley/linux-voice/main/linux-voice.py
chmod +x ~/.local/bin/linux-voice
```

### Arch Linux

```bash
# System dependencies
sudo pacman -S xdotool ffmpeg python-numpy python-pynput python-sounddevice python-openai

# Download and install script
curl -o ~/.local/bin/linux-voice https://raw.githubusercontent.com/williamjameshandley/linux-voice/main/linux-voice.py
chmod +x ~/.local/bin/linux-voice
```

Or build the package:

```bash
git clone https://github.com/williamjameshandley/linux-voice
cd linux-voice
makepkg -si
```

### pip (Any Distribution)

```bash
# Ensure system dependencies are installed first:
# - xdotool (for typing text)
# - ffmpeg (for audio compression, optional but recommended)

pip install pynput sounddevice numpy openai

# Download script
curl -o ~/.local/bin/linux-voice https://raw.githubusercontent.com/williamjameshandley/linux-voice/main/linux-voice.py
chmod +x ~/.local/bin/linux-voice
```

### API Key Setup

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):
```bash
export OPENAI_API_KEY="your-key-here"
```

For the systemd service, create `~/.config/environment.d/openai.conf`:
```
OPENAI_API_KEY=your-key-here
```

## Usage

### Manual

```bash
linux-voice
```

If using a virtual environment:
```bash
~/.local/share/linux-voice/bin/python ~/.local/bin/linux-voice
```

### Systemd Service (Auto-start)

Create `~/.config/systemd/user/linux-voice.service`:
```ini
[Unit]
Description=Linux Voice Dictation

[Service]
ExecStart=%h/.local/bin/linux-voice
Restart=on-failure

[Install]
WantedBy=default.target
```

If using a virtual environment, update `ExecStart`:
```ini
ExecStart=%h/.local/share/linux-voice/bin/python %h/.local/bin/linux-voice
```

Then enable:
```bash
systemctl --user daemon-reload
systemctl --user enable --now linux-voice

# Check status
systemctl --user status linux-voice

# View logs
journalctl --user -u linux-voice -f
```

### Modes

- **Hold mode** (default): Hold Ctrl+Space while speaking, release to transcribe
- **Toggle mode**: Press Ctrl+Space to start recording, press again to stop

```bash
LINUX_VOICE_MODE=toggle linux-voice
```

### Submit Hotkey

Use Ctrl+Shift+Space (default) to record and automatically press Enter after typing. Useful for command-line input, chat applications, or any context where you want to submit immediately.

Configure an alternative (e.g., Alt+Space) in `~/.config/linux-voice/config.toml`:
```toml
[hotkey_submit]
key = "space"
modifiers = ["alt"]
delay = 150  # ms delay before Enter (increase if unreliable)
```

### Edit Hotkey

Use Ctrl+Alt+Space (default) to correct the previous transcription using an LLM. This is useful when Whisper misinterprets a word:

1. Dictate normally with Ctrl+Space → "The function uses a cash mechanism"
2. Hold Ctrl+Alt+Space and say "change cash to cache"
3. The original text is backspaced and replaced with "The function uses a cache mechanism"

The edit uses the backend's chat model (gpt-4o-mini for OpenAI, llama-3.3-70b-versatile for Groq).

Configure in `~/.config/linux-voice/config.toml`:
```toml
[hotkey_edit]
key = "space"
modifiers = ["ctrl", "alt"]

[transcription]
# llm_model = "gpt-4o-mini"  # Override LLM model for corrections
```

## Configuration

Create `~/.config/linux-voice/config.toml` to customize:

```toml
[hotkey]
key = "space"
modifiers = ["ctrl"]
mode = "hold"  # or "toggle"

[hotkey_submit]
key = "space"
modifiers = ["ctrl", "shift"]  # Ctrl+Shift+Space by default
delay = 150                     # ms delay before Enter key

[hotkey_edit]
key = "space"
modifiers = ["ctrl", "alt"]    # Ctrl+Alt+Space by default

[audio]
sample_rate = 16000      # Whisper native rate (don't change unless needed)
silence_threshold = 150  # RMS threshold for silence detection

[transcription]
backend = "openai"  # or "groq"
language = "en"
# model = "whisper-1"  # or "whisper-large-v3-turbo" for groq
# llm_model = "gpt-4o-mini"  # or "llama-3.3-70b-versatile" for groq (edit mode)
# prompt = "Domain-specific vocabulary. Technical terms, jargon, names."
```

### Backend Options

| Backend | Model | Cost | Latency |
|---------|-------|------|---------|
| `openai` | whisper-1 | $0.006/min | ~1-2s |
| `groq` | whisper-large-v3-turbo | $0.04/hr | ~200ms |

For Groq, set `GROQ_API_KEY` instead of `OPENAI_API_KEY`.

The `prompt` helps Whisper with:
- British vs American spelling (colour, favour, organisation)
- Domain-specific vocabulary (your field's jargon)
- Consistent formatting and punctuation

### Text Replacements

Define regex replacements to convert spoken phrases into special characters or commands:

```toml
[replacements]
"^[Ss]lash " = "/"              # "slash compact" → "/compact"
"^[Ff]orward [Ss]lash " = "/"
"^(/.*)\\.\\s*$" = "\\1"        # strip period only from /commands
```

This allows saying "slash compact" to type `/compact` for Claude Code commands.

## Privacy and Security

**Audio is sent to OpenAI:** All recorded speech is transmitted to OpenAI's Whisper API for transcription. See [OpenAI's data usage policies](https://openai.com/policies/api-data-usage-policies).

**Text is typed into the focused window:** Be careful not to dictate sensitive information while password fields or sensitive applications are focused.

## Cost

OpenAI Whisper API costs $0.006 per minute of audio. A typical 10-second dictation costs ~$0.001.

## Troubleshooting

### Microphone not working

**Ubuntu/Debian:**
```bash
sudo apt install linux-firmware
```

**Fedora:**
```bash
sudo dnf install linux-firmware
```

**Arch Linux** (modern AMD laptops - Ryzen 6000+, Strix Point):
```bash
sudo pacman -S sof-firmware
```

Then reboot.

### Vim

Ctrl+Space (`C-@`) has a default behavior in vim insert mode (insert previously inserted text). Disable it by adding to your `.vimrc`:

```vim
" Disable Ctrl+Space insert mode behavior (for linux-voice)
inoremap <C-@> <Nop>
```

### Wayland

xdotool has limited Wayland support. Consider using X11 or switching to ydotool.

### Permission errors with pynput

On some systems, you may need to run as root or add your user to the `input` group:
```bash
sudo usermod -aG input $USER
```
Then log out and back in.

## License

MIT
