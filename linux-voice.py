#!/usr/bin/env python3
"""
linux-voice: Voice-to-text dictation tool for Linux (X11) and macOS

Hold Ctrl+Space to record, release to transcribe and type.
Toggle mode: Press Ctrl+Space to start, press again to stop.

Requires Python 3.11+ for tomllib.
"""

import io
import os
import shutil
import subprocess
import sys
import threading
import wave
from pathlib import Path

# Ensure platform_support.py is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import numpy as np
from platform_support import get_platform
import sounddevice as sd
from pynput import keyboard

# Minimum recording duration in seconds (avoid accidental triggers)
MIN_RECORDING_SECONDS = 0.3

# Load config from ~/.config/linux-voice/config.toml if it exists
CONFIG = {}
CONFIG_PATH = Path.home() / ".config" / "linux-voice" / "config.toml"
if CONFIG_PATH.exists():
    try:
        import tomllib
        CONFIG = tomllib.loads(CONFIG_PATH.read_text())
    except Exception as e:
        print(f"\033[91mConfig error in {CONFIG_PATH}: {e}. Using defaults.\033[0m")
        CONFIG = {}

# Configuration with defaults
SAMPLE_RATE = CONFIG.get("audio", {}).get("sample_rate", 16000)  # Whisper native rate
CHANNELS = 1
LANGUAGE = CONFIG.get("transcription", {}).get("language", "en")
PROMPT = CONFIG.get("transcription", {}).get("prompt", "")
REPLACEMENTS = CONFIG.get("replacements", {})

# Backend configuration (openai or groq)
BACKEND = CONFIG.get("transcription", {}).get("backend", "openai")
WHISPER_MODEL = CONFIG.get("transcription", {}).get("model", None)
if WHISPER_MODEL is None:
    WHISPER_MODEL = "whisper-large-v3-turbo" if BACKEND == "groq" else "whisper-1"
MODE = os.environ.get(
    "LINUX_VOICE_MODE",
    CONFIG.get("hotkey", {}).get("mode", "hold"),
)

# Platform-aware hotkey defaults
# macOS: Cmd+Shift+Space (avoids Spotlight and input source switching conflicts)
# Linux: Ctrl+Space
_IS_MACOS = sys.platform == "darwin"
_DEFAULT_RECORD_MODS = ["cmd", "shift"] if _IS_MACOS else ["ctrl"]
_DEFAULT_SUBMIT_MODS = ["cmd", "shift", "ctrl"] if _IS_MACOS else ["ctrl", "shift"]
_DEFAULT_EDIT_MODS = ["cmd", "alt"] if _IS_MACOS else ["ctrl", "alt"]


def _parse_modifiers(modifier_names: list[str]) -> set:
    """Parse modifier name list into pynput key set."""
    mods = set()
    for mod in modifier_names:
        if mod == "ctrl":
            mods.update({keyboard.Key.ctrl_l, keyboard.Key.ctrl_r})
        elif mod == "alt":
            mods.update({keyboard.Key.alt_l, keyboard.Key.alt_r})
        elif mod == "shift":
            mods.update({keyboard.Key.shift_l, keyboard.Key.shift_r})
        elif mod in ("super", "cmd"):
            mods.update({keyboard.Key.cmd_l, keyboard.Key.cmd_r})
    return mods


# Hotkey configuration
_hotkey_cfg = CONFIG.get("hotkey", {})
_key_name = _hotkey_cfg.get("key", "space")
HOTKEY_KEY = getattr(keyboard.Key, _key_name, keyboard.KeyCode.from_char(_key_name))
_modifier_names = _hotkey_cfg.get("modifiers", _DEFAULT_RECORD_MODS)
HOTKEY_MODIFIERS = _parse_modifiers(_modifier_names)
if not HOTKEY_MODIFIERS:
    HOTKEY_MODIFIERS = _parse_modifiers(_DEFAULT_RECORD_MODS)

# For "all modifiers" check, we need one key per modifier type
# Normalize "cmd" to "super" for consistent checking
_required_modifier_types = set(
    "super" if m == "cmd" else m for m in (_modifier_names or _DEFAULT_RECORD_MODS)
)

# Submit hotkey - same as main hotkey but presses Enter after
_submit_cfg = CONFIG.get("hotkey_submit", {})
SUBMIT_DELAY = _submit_cfg.get("delay", 150)  # ms delay before Enter key
SUBMIT_KEY = getattr(
    keyboard.Key,
    _submit_cfg.get("key", "space"),
    keyboard.KeyCode.from_char(_submit_cfg.get("key", "space")),
)
_submit_modifier_names = _submit_cfg.get("modifiers", _DEFAULT_SUBMIT_MODS)
SUBMIT_MODIFIERS = _parse_modifiers(_submit_modifier_names)
if not SUBMIT_MODIFIERS:
    SUBMIT_MODIFIERS = _parse_modifiers(_DEFAULT_SUBMIT_MODS)
_submit_modifier_types = set(
    "super" if m == "cmd" else m for m in (_submit_modifier_names or _DEFAULT_SUBMIT_MODS)
)

# Edit hotkey - corrects last transcription via LLM
_edit_cfg = CONFIG.get("hotkey_edit", {})
EDIT_KEY = getattr(
    keyboard.Key,
    _edit_cfg.get("key", "space"),
    keyboard.KeyCode.from_char(_edit_cfg.get("key", "space")),
)
_edit_modifier_names = _edit_cfg.get("modifiers", _DEFAULT_EDIT_MODS)
EDIT_MODIFIERS = _parse_modifiers(_edit_modifier_names)
if not EDIT_MODIFIERS:
    EDIT_MODIFIERS = _parse_modifiers(_DEFAULT_EDIT_MODS)
_edit_modifier_types = set(
    "super" if m == "cmd" else m for m in (_edit_modifier_names or _DEFAULT_EDIT_MODS)
)

# LLM model for corrections (per backend)
LLM_MODEL = CONFIG.get("transcription", {}).get("llm_model", None)
if LLM_MODEL is None:
    LLM_MODEL = "llama-3.3-70b-versatile" if BACKEND == "groq" else "gpt-4o-mini"


def apply_replacements(text: str) -> str:
    """Apply user-configured regex replacements."""
    import re
    for pattern, replacement in REPLACEMENTS.items():
        text = re.sub(pattern, replacement, text)
    return text


def check_connectivity(timeout: float = 2.0) -> bool:
    """Quick check for internet connectivity to API endpoint."""
    import socket
    host = "api.groq.com" if BACKEND == "groq" else "api.openai.com"
    try:
        socket.create_connection((host, 443), timeout=timeout)
        return True
    except OSError:
        return False


def check_environment(platform):
    """Check for required dependencies and environment."""
    errors = []

    # Check for API key: environment variable first, then config.toml fallback
    if BACKEND == "groq":
        api_key = os.environ.get("GROQ_API_KEY") or CONFIG.get("transcription", {}).get("api_key", "")
        if not api_key:
            errors.append("GROQ_API_KEY not set. Set env var or add api_key to [transcription] in config.toml")
        else:
            os.environ["GROQ_API_KEY"] = api_key
    else:
        api_key = os.environ.get("OPENAI_API_KEY") or CONFIG.get("transcription", {}).get("api_key", "")
        if not api_key:
            errors.append("OPENAI_API_KEY not set. Set env var or add api_key to [transcription] in config.toml")
        else:
            os.environ["OPENAI_API_KEY"] = api_key

    # Platform-specific checks (xdotool on Linux, Accessibility on macOS, etc.)
    errors.extend(platform.check_environment())

    if errors:
        for err in errors:
            print(f"\033[91mError: {err}\033[0m")
        sys.exit(1)


def _has_all_modifiers(pressed: set, required_types: set) -> bool:
    """Check if all required modifier types are pressed."""
    for mod_type in required_types:
        if mod_type == "ctrl":
            if keyboard.Key.ctrl_l not in pressed and keyboard.Key.ctrl_r not in pressed:
                return False
        elif mod_type == "alt":
            if keyboard.Key.alt_l not in pressed and keyboard.Key.alt_r not in pressed:
                return False
        elif mod_type == "shift":
            if keyboard.Key.shift_l not in pressed and keyboard.Key.shift_r not in pressed:
                return False
        elif mod_type == "super":
            if keyboard.Key.cmd_l not in pressed and keyboard.Key.cmd_r not in pressed:
                return False
    return True


class VoiceRecorder:
    def __init__(self, platform):
        if BACKEND == "groq":
            from groq import Groq
            self.client = Groq()
        else:
            from openai import OpenAI
            self.client = OpenAI()
        self.platform = platform
        self.recording = False
        self.audio_data = []
        self.stream = None
        self.pressed_modifiers = set()
        self.hotkey_pressed = False
        self.has_ffmpeg = shutil.which("ffmpeg") is not None
        self.submit_mode = False  # Whether to press Enter after typing
        self.edit_mode = False  # Whether to correct last transcription
        self.last_typed_text = ""  # Store for edit mode corrections
        self.active_app = None  # App/window to type into

    def _convert_to_mp3(self, audio: np.ndarray) -> io.BytesIO:
        """Convert numpy audio to MP3 using ffmpeg."""
        command = [
            "ffmpeg",
            "-f", "s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-i", "pipe:0",
            "-codec:a", "libmp3lame",
            "-b:a", "32k",
            "-f", "mp3",
            "pipe:1",
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        mp3_data, _ = process.communicate(input=audio.tobytes())
        mp3_buffer = io.BytesIO(mp3_data)
        mp3_buffer.name = "audio.mp3"
        return mp3_buffer

    def _correct_with_llm(self, original: str, instruction: str) -> str:
        """Use LLM to correct text based on instruction."""
        system_prompt = """You are an intelligent text editing assistant for a voice dictation tool used by a programmer in a Unix terminal environment. Your goal is to apply user correction instructions to the provided text.

Context about the user:
- Working in Unix/Linux terminal, often typing shell commands and code
- Programming languages: Python, shell scripts, and related tools
- Voice recognition often mishears technical terms (e.g., "cash" for "cache", "bite" for "byte", "get" for "git", "pie" for "py", "boss" for "bash")

Follow these rules:
1. GLOBAL APPLICATION: Apply the instruction to the ENTIRE text. Do not stop after the first few instances. Scan from start to finish.
2. BROAD INTENT: Interpret instructions generously. If the user asks to change "numbers", apply it to digits, written words, and mixed formats. If they ask to "remove filler words", remove all types.
3. CONSISTENCY: Ensure the corrected text is stylistically consistent. If a change is applied to one part, apply it to matching patterns elsewhere.
4. INTEGRITY: Do not change the meaning of the content, only the form/style as requested.
5. TECHNICAL AWARENESS: Consider programming/Unix context when interpreting corrections.
6. STRICT OUTPUT: Output ONLY the corrected text. No preamble, no explanation, no quotes."""

        # Include domain context from Whisper prompt if available
        context_note = ""
        if PROMPT:
            context_note = f"\n\nDomain context: {PROMPT}"

        user_prompt = f"""Original text: {original}

Instruction: {instruction}{context_note}"""

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"\033[91mLLM error: {e}\033[0m")
            return original  # Return original on error

    def start_recording(self, submit=False, edit=False):
        if self.recording:
            return
        if edit and not self.last_typed_text:
            print("\033[91mNo previous text to edit\033[0m", flush=True)
            return
        self.audio_data = []
        self.submit_mode = submit
        self.edit_mode = edit
        # Capture active window/app for later focus restoration
        try:
            self.active_app = self.platform.get_active_app()
        except Exception:
            self.active_app = None
        if edit:
            indicator = "● Recording edit instruction..."
        elif submit:
            indicator = "● Recording (submit)..."
        else:
            indicator = "● Recording..."
        print(f"\033[92m{indicator}\033[0m", flush=True)

        def callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}", flush=True)
            if self.recording:
                self.audio_data.append(indata.copy())

        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=np.int16,
                device=None,
                callback=callback,
            )
            self.stream.start()
            self.recording = True
        except Exception as e:
            self.recording = False
            self.hotkey_pressed = False
            print(f"\033[91mAudio error: {e}\033[0m", flush=True)

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if not self.audio_data:
            print("No audio recorded")
            return

        # Combine audio chunks
        audio = np.concatenate(self.audio_data, axis=0)

        # Check minimum duration
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_RECORDING_SECONDS:
            print(f"(recording too short: {duration:.1f}s)")
            return

        print("\033[93m◌ Processing...\033[0m", flush=True)

        # Transcribe in background to not block hotkey listener
        threading.Thread(
            target=self._transcribe_and_type, args=(audio, self.submit_mode, self.edit_mode),
            daemon=True,
        ).start()

    def _transcribe_and_type(self, audio: np.ndarray, submit: bool = False, edit: bool = False):
        import time
        t0 = time.time()
        try:
            # Check internet connectivity
            if not check_connectivity():
                print("\033[91mNo internet connection\033[0m")
                # Save audio to temp file for potential recovery (don't overwrite existing)
                recovery_path = Path("/tmp/linux-voice-recovery.wav")
                if recovery_path.exists():
                    # Don't overwrite - just remind user to recover first
                    try:
                        self.platform.type_text("(no internet - say 'recover' first)")
                    except Exception:
                        pass
                else:
                    with wave.open(str(recovery_path), "wb") as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(2)
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes(audio.tobytes())
                    print(f"Audio saved to {recovery_path}")
                    try:
                        self.platform.type_text("(no internet - say 'recover' to retry)")
                    except Exception:
                        pass
                return

            # Convert to MP3 if ffmpeg available, otherwise WAV
            if self.has_ffmpeg:
                file_buffer = self._convert_to_mp3(audio)
            else:
                file_buffer = io.BytesIO()
                with wave.open(file_buffer, "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(audio.tobytes())
                file_buffer.seek(0)
                file_buffer.name = "audio.wav"

            # Transcribe
            transcript = self.client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=file_buffer,
                language=LANGUAGE,
                prompt=PROMPT,
                response_format="text",
                temperature=0.0,
            )
            t1 = time.time()

            text = transcript.strip()
            if not text:
                print("(no speech detected)")
                return

            # Check for voice commands
            if text.lower() in ("recover", "recover.", "recovery", "recovery."):
                print("Voice command: recover")
                self.recover_audio()
                return

            # Release any stuck modifiers before typing
            self.platform.release_modifiers()

            # Restore focus to the original window (may have changed during API call)
            self.platform.restore_focus(self.active_app)

            if edit:
                # Edit mode: use transcription as instruction to correct previous text
                instruction = text
                print(f"\033[93m◌ Correcting: {instruction}\033[0m", flush=True)

                corrected = self._correct_with_llm(self.last_typed_text, instruction)
                print(f"\033[94m→ {corrected}\033[0m ({time.time()-t0:.1f}s)", flush=True)

                # Clear the current line
                self.platform.clear_line()

                # Type corrected text
                self.platform.type_text(corrected)
                self.last_typed_text = corrected
            else:
                # Normal mode: apply replacements and type
                text = apply_replacements(text)
                print(f"\033[94m→ {text}\033[0m ({t1-t0:.1f}s)", flush=True)

                self.platform.type_text(text)
                self.last_typed_text = text

                # Press Enter if submit mode
                if submit:
                    import time
                    time.sleep(SUBMIT_DELAY / 1000)
                    self.platform.press_key("Return")
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m")

    def recover_audio(self):
        """Recover and transcribe audio from a failed attempt."""
        recovery_path = Path("/tmp/linux-voice-recovery.wav")
        if not recovery_path.exists():
            print("No recovery file found")
            try:
                self.platform.type_text("(no recovery file)")
            except Exception:
                pass
            return

        # Load audio from recovery file
        with wave.open(str(recovery_path), "rb") as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

        print(f"Recovering audio from {recovery_path}...")

        # Clear the "no internet" message first
        self.platform.clear_line()

        # Transcribe in background
        threading.Thread(
            target=self._transcribe_and_type, args=(audio, False, False),
            daemon=True,
        ).start()

        # Remove recovery file
        recovery_path.unlink()

    def on_press(self, key):
        # Track modifier keys (include all hotkey modifiers)
        all_modifiers = HOTKEY_MODIFIERS | SUBMIT_MODIFIERS | EDIT_MODIFIERS
        if key in all_modifiers:
            self.pressed_modifiers.add(key)

        # Check if edit hotkey is pressed (Ctrl+Alt+Space or configured)
        if key == EDIT_KEY and _has_all_modifiers(
            self.pressed_modifiers, _edit_modifier_types
        ):
            if MODE == "toggle":
                if self.recording:
                    self.stop_recording()
                else:
                    self.start_recording(edit=True)
            else:  # hold mode
                if not self.hotkey_pressed:
                    self.hotkey_pressed = True
                    self.start_recording(edit=True)
            return

        # Check if submit hotkey is pressed (Ctrl+Shift+Space or configured)
        if key == SUBMIT_KEY and _has_all_modifiers(
            self.pressed_modifiers, _submit_modifier_types
        ):
            if MODE == "toggle":
                if self.recording:
                    self.stop_recording()
                else:
                    self.start_recording(submit=True)
            else:  # hold mode
                if not self.hotkey_pressed:
                    self.hotkey_pressed = True
                    self.start_recording(submit=True)
            return

        # Check if hotkey combo is pressed (all required modifiers)
        if key == HOTKEY_KEY and _has_all_modifiers(
            self.pressed_modifiers, _required_modifier_types
        ):
            if MODE == "toggle":
                if self.recording:
                    self.stop_recording()
                else:
                    self.start_recording()
            else:  # hold mode
                if not self.hotkey_pressed:
                    self.hotkey_pressed = True
                    self.start_recording()

    def on_release(self, key):
        # Track modifier release
        all_modifiers = HOTKEY_MODIFIERS | SUBMIT_MODIFIERS | EDIT_MODIFIERS
        if key in all_modifiers:
            self.pressed_modifiers.discard(key)

        # In hold mode, stop when space is released (works for all hotkeys)
        if MODE == "hold" and key in (HOTKEY_KEY, SUBMIT_KEY, EDIT_KEY) and self.hotkey_pressed:
            self.hotkey_pressed = False
            self.stop_recording()

    def _setup_wake_listener(self):
        """On macOS, exit on wake from sleep so launchd restarts us.

        This is needed because macOS invalidates Accessibility trust
        tokens after sleep, causing CGEvents to silently fail.
        A fresh process gets a new valid token.
        """
        if sys.platform != "darwin":
            return
        try:
            from AppKit import NSWorkspace, NSWorkspaceDidWakeNotification

            def on_wake(_notification):
                print("System wake detected, waiting for Accessibility restore...", flush=True)
                time.sleep(10)  # wait for macOS to restore Accessibility trust
                print("Restarting...", flush=True)
                os._exit(0)  # launchd KeepAlive will restart us

            center = NSWorkspace.sharedWorkspace().notificationCenter()
            center.addObserverForName_object_queue_usingBlock_(
                NSWorkspaceDidWakeNotification, None, None, on_wake,
            )

            # Pump CFRunLoop on a background thread to receive notifications
            def run_loop():
                from CoreFoundation import CFRunLoopRun
                CFRunLoopRun()  # blocks, no CPU spin

            threading.Thread(target=run_loop, daemon=True).start()
            print("Wake listener active", flush=True)
        except Exception as e:
            print(f"Warning: could not set up wake listener: {e}", flush=True)

    def run(self):
        # Format hotkey names for display
        hotkey_str = "+".join(m.capitalize() for m in _required_modifier_types) + "+Space"
        submit_str = "+".join(m.capitalize() for m in _submit_modifier_types) + "+Space"
        edit_str = "+".join(m.capitalize() for m in _edit_modifier_types) + "+Space"

        print(f"linux-voice started (mode: {MODE}, backend: {BACKEND})")
        print(f"  {hotkey_str}: {'toggle' if MODE == 'toggle' else 'hold to'} record")
        print(f"  {submit_str}: record and submit (press Enter)")
        print(f"  {edit_str}: record correction instruction")
        print("Press Ctrl+C to exit\n")

        self._setup_wake_listener()

        try:
            with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release,
            ) as listener:
                listener.join()
        except Exception as e:
            print(f"\033[91mKeyboard listener error: {e}\033[0m")
            if sys.platform == "darwin":
                print("Make sure Accessibility permissions are granted in System Settings.")
            else:
                print("Make sure you're running on X11 with proper permissions.")
            sys.exit(1)


def recover():
    """Transcribe the recovery audio file from a failed attempt."""
    recovery_path = Path("/tmp/linux-voice-recovery.wav")
    if not recovery_path.exists():
        print("No recovery file found")
        return

    platform = get_platform()
    check_environment(platform)

    # Load audio from recovery file
    with wave.open(str(recovery_path), "rb") as wf:
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    print(f"Recovering audio from {recovery_path}...")
    recorder = VoiceRecorder(platform)
    recorder._transcribe_and_type(audio, submit=False, edit=False)

    # Remove recovery file after successful transcription
    recovery_path.unlink()
    print("Recovery file removed")


def install_agent():
    """Generate and install macOS LaunchAgent."""
    if sys.platform != "darwin":
        print("LaunchAgent installation is only for macOS.")
        return

    plist_dir = Path.home() / "Library" / "LaunchAgents"
    plist_path = plist_dir / "com.linux-voice.agent.plist"
    log_dir = Path.home() / "Library" / "Logs"

    python_path = sys.executable
    script_path = Path(__file__).resolve()

    # Read API key from config or environment
    api_key_name = "GROQ_API_KEY" if BACKEND == "groq" else "OPENAI_API_KEY"
    api_key = (
        CONFIG.get("transcription", {}).get("api_key", "")
        or os.environ.get(api_key_name, "")
    )

    env_block = f"""        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>"""
    if api_key:
        env_block = f"""        <key>{api_key_name}</key>
        <string>{api_key}</string>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>"""

    # Run through /bin/zsh so the process inherits Accessibility permissions
    # (macOS grants permissions to /bin/zsh, not the Python binary)
    plist_content = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.linux-voice.agent</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/zsh</string>
        <string>-c</string>
        <string>exec {python_path} -u {script_path}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{script_path.parent}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir}/linux-voice.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/linux-voice.err</string>
    <key>EnvironmentVariables</key>
    <dict>
{env_block}
    </dict>
</dict>
</plist>
"""
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist_content)
    print(f"Wrote {plist_path}")
    print()
    print("To start now and on every login:")
    print(f"  launchctl bootstrap gui/$(id -u) {plist_path}")
    print()
    print("To restart after changes:")
    print(f"  launchctl kickstart -k gui/$(id -u)/com.linux-voice.agent")
    print()
    print("To stop and remove:")
    print(f"  launchctl bootout gui/$(id -u)/com.linux-voice.agent")


def uninstall_agent():
    """Remove macOS LaunchAgent."""
    if sys.platform != "darwin":
        print("LaunchAgent uninstallation is only for macOS.")
        return

    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.linux-voice.agent.plist"

    # Try to unload first
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}/com.linux-voice.agent"],
        check=False,
        capture_output=True,
    )

    if plist_path.exists():
        plist_path.unlink()
        print(f"Removed {plist_path}")
    else:
        print("LaunchAgent not installed.")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--install-agent":
        install_agent()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--uninstall-agent":
        uninstall_agent()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--recover":
        recover()
        return

    platform = get_platform()
    check_environment(platform)
    recorder = VoiceRecorder(platform)
    try:
        recorder.run()
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
