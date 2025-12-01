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

### Arch Linux (AUR)

```bash
# Install AUR dependencies first
paru -S python-pynput python-sounddevice

# Clone and build package
git clone https://github.com/williamjameshandley/linux-voice
cd linux-voice
makepkg -si
```

Or with an AUR helper that handles dependencies:
```bash
paru -S linux-voice  # once published to AUR
```

### Manual Installation

```bash
# System dependencies (Arch Linux)
sudo pacman -S xdotool ffmpeg python-numpy python-pynput python-sounddevice python-openai

# Or with pip (still need system packages for xdotool and ffmpeg)
sudo pacman -S xdotool ffmpeg
pip install pynput sounddevice numpy openai
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

### Systemd Service (Auto-start)

```bash
# Enable and start the service
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

## Configuration

Create `~/.config/linux-voice/config.toml` to customize:

```toml
[hotkey]
key = "space"
modifiers = ["ctrl"]
mode = "hold"  # or "toggle"

[audio]
sample_rate = 16000      # Whisper native rate (don't change unless needed)
silence_threshold = 150  # RMS threshold for silence detection

[transcription]
language = "en"
prompt = "British English academic dictation. Bayesian, cosmology, nested sampling."
```

The `prompt` helps Whisper with:
- British vs American spelling (colour, favour, organisation)
- Domain-specific vocabulary (your field's jargon)
- Consistent formatting and punctuation

## Privacy and Security

**Audio is sent to OpenAI:** All recorded speech is transmitted to OpenAI's Whisper API for transcription. See [OpenAI's data usage policies](https://openai.com/policies/api-data-usage-policies).

**Text is typed into the focused window:** Be careful not to dictate sensitive information while password fields or sensitive applications are focused.

## Cost

OpenAI Whisper API costs $0.006 per minute of audio. A typical 10-second dictation costs ~$0.001.

## Troubleshooting

### Microphone not working

On modern AMD laptops (Ryzen 6000+, Strix Point), you may need:
```bash
sudo pacman -S sof-firmware
```
Then reboot.

### Wayland

xdotool has limited Wayland support. Consider using X11 or switching to ydotool.

## License

MIT
