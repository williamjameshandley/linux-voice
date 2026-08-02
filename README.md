# linux-voice

Voice-to-text dictation for Linux/X11 and macOS. It records while a hotkey is
held, sends the recording to OpenAI or Groq, and inserts the transcription into
the window that was focused when recording began.

Correct, minimal documentation is best. Omission is preferable to an
unsupported or obsolete claim. Incorrect documentation is worst.

## Linux installation

The Arch package is the maintained Linux installation. It installs the program
as `/usr/bin/linux-voice` and the packaged user unit as
`/usr/lib/systemd/user/linux-voice.service`.

```sh
makepkg -si
systemctl --user enable --now linux-voice.service
```

Provide the API key to the user service through the systemd user manager, for
example with `~/.config/environment.d/linux-voice.conf`:

```ini
OPENAI_API_KEY=...
```

Then reload the manager environment and restart the service:

```sh
systemctl --user daemon-reload
systemctl --user restart linux-voice.service
```

The Linux implementation requires X11, `xdotool`, a working microphone, and an
OpenAI key by default. Wayland is detected but text injection through `xdotool`
is not reliable there. `ffmpeg` is used for compressed uploads; without it the
program sends WAV audio.

## macOS installation

Install `ffmpeg` and `uv`, then install the project with the macOS extra and the
extra for the selected backend:

```sh
brew install ffmpeg uv
uv sync --extra macos --extra groq
```

Grant Accessibility permission to `/bin/zsh` and microphone permission when
prompted. Generate the LaunchAgent from the active environment and start it:

```sh
uv run python linux-voice.py --install-agent
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.linux-voice.agent.plist
```

The generated agent runs the current Python and checkout paths. Remove it with:

```sh
uv run python linux-voice.py --uninstall-agent
```

## Configuration

Configuration is read from `~/.config/linux-voice/config.toml`. API keys may be
provided as `OPENAI_API_KEY` or `GROQ_API_KEY`, or as `api_key` in the
`[transcription]` table.

```toml
[transcription]
backend = "openai"              # or "groq"
language = "en"
# model = "whisper-1"
# llm_model = "gpt-4o-mini"
# prompt = "Names and domain vocabulary"

[audio]
sample_rate = 16000

[hotkey]
key = "space"
modifiers = ["ctrl"]
mode = "hold"                  # or "toggle"

[hotkey_submit]
key = "space"
modifiers = ["ctrl", "shift"]
delay = 150

[hotkey_edit]
key = "space"
modifiers = ["ctrl", "alt"]

[replacements]
"^[Ss]lash " = "/"
```

`LINUX_VOICE_MODE` overrides `hotkey.mode`.

Default hotkeys are:

| Action | Linux | macOS |
| --- | --- | --- |
| Record and insert | Ctrl+Space | Cmd+Shift+Space |
| Record, insert, and press Enter | Ctrl+Shift+Space | Cmd+Shift+Ctrl+Space |
| Correct the previous insertion | Ctrl+Alt+Space | Cmd+Alt+Space |

Edit mode transcribes a correction instruction and applies it to the last text
inserted by the current process. The correction model defaults to
`gpt-4o-mini` for OpenAI and `llama-3.3-70b-versatile` for Groq.

## Recovery and evidence

If connectivity fails, the program preserves one recording at
`/tmp/linux-voice-recovery.wav`. Say `recover`, or run
`linux-voice --recover`, after connectivity returns. A second failed recording
does not overwrite the first.

Every transcription outcome is appended to
`$XDG_STATE_HOME/linux-voice/ledger.jsonl`, or
`~/.local/state/linux-voice/ledger.jsonl` when `XDG_STATE_HOME` is unset. It
includes the captured window title and whether text was typed. This is the
recovery record when text reaches the wrong window or an insertion fails.

## Operations

On Linux:

```sh
systemctl --user status linux-voice.service
journalctl --user -u linux-voice.service -f
```

On macOS:

```sh
tail -f ~/Library/Logs/linux-voice.log
launchctl kickstart -k gui/$(id -u)/com.linux-voice.agent
```

Audio is sent to the configured transcription provider. Text is injected into
the focused application, so secure-input fields and applications that reject
synthetic input can prevent insertion.

## License

MIT
