#!/usr/bin/env python3
"""
linux-voice: Voice-to-text dictation tool for Linux (X11)

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

import numpy as np
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
API_TIMEOUT = CONFIG.get("transcription", {}).get("timeout", 5.0)
MODE = os.environ.get(
    "LINUX_VOICE_MODE",
    CONFIG.get("hotkey", {}).get("mode", "hold"),
)

# Hotkey configuration
_hotkey_cfg = CONFIG.get("hotkey", {})
_key_name = _hotkey_cfg.get("key", "space")
HOTKEY_KEY = getattr(keyboard.Key, _key_name, keyboard.KeyCode.from_char(_key_name))
_modifier_names = _hotkey_cfg.get("modifiers", ["ctrl"])
HOTKEY_MODIFIERS = set()
for mod in _modifier_names:
    if mod == "ctrl":
        HOTKEY_MODIFIERS.update({keyboard.Key.ctrl_l, keyboard.Key.ctrl_r})
    elif mod == "alt":
        HOTKEY_MODIFIERS.update({keyboard.Key.alt_l, keyboard.Key.alt_r})
    elif mod == "shift":
        HOTKEY_MODIFIERS.update({keyboard.Key.shift_l, keyboard.Key.shift_r})
    elif mod == "super":
        HOTKEY_MODIFIERS.update({keyboard.Key.cmd_l, keyboard.Key.cmd_r})
if not HOTKEY_MODIFIERS:
    HOTKEY_MODIFIERS = {keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}

# For "all modifiers" check, we need one key per modifier type
_required_modifier_types = set(_modifier_names) if _modifier_names else {"ctrl"}

# Submit hotkey (Ctrl+Shift+Space by default) - same as main hotkey but presses Enter after
_submit_cfg = CONFIG.get("hotkey_submit", {})
SUBMIT_DELAY = _submit_cfg.get("delay", 150)  # ms delay before Enter key
SUBMIT_KEY = getattr(
    keyboard.Key,
    _submit_cfg.get("key", "space"),
    keyboard.KeyCode.from_char(_submit_cfg.get("key", "space")),
)
_submit_modifier_names = _submit_cfg.get("modifiers", ["ctrl", "shift"])
SUBMIT_MODIFIERS = set()
for mod in _submit_modifier_names:
    if mod == "ctrl":
        SUBMIT_MODIFIERS.update({keyboard.Key.ctrl_l, keyboard.Key.ctrl_r})
    elif mod == "alt":
        SUBMIT_MODIFIERS.update({keyboard.Key.alt_l, keyboard.Key.alt_r})
    elif mod == "shift":
        SUBMIT_MODIFIERS.update({keyboard.Key.shift_l, keyboard.Key.shift_r})
    elif mod == "super":
        SUBMIT_MODIFIERS.update({keyboard.Key.cmd_l, keyboard.Key.cmd_r})
if not SUBMIT_MODIFIERS:
    SUBMIT_MODIFIERS = {keyboard.Key.ctrl_l, keyboard.Key.ctrl_r, keyboard.Key.shift_l, keyboard.Key.shift_r}
_submit_modifier_types = set(_submit_modifier_names) if _submit_modifier_names else {"ctrl", "shift"}

# Edit hotkey (Ctrl+Alt+Space by default) - corrects last transcription via LLM
_edit_cfg = CONFIG.get("hotkey_edit", {})
EDIT_KEY = getattr(
    keyboard.Key,
    _edit_cfg.get("key", "space"),
    keyboard.KeyCode.from_char(_edit_cfg.get("key", "space")),
)
_edit_modifier_names = _edit_cfg.get("modifiers", ["ctrl", "alt"])
EDIT_MODIFIERS = set()
for mod in _edit_modifier_names:
    if mod == "ctrl":
        EDIT_MODIFIERS.update({keyboard.Key.ctrl_l, keyboard.Key.ctrl_r})
    elif mod == "alt":
        EDIT_MODIFIERS.update({keyboard.Key.alt_l, keyboard.Key.alt_r})
    elif mod == "shift":
        EDIT_MODIFIERS.update({keyboard.Key.shift_l, keyboard.Key.shift_r})
    elif mod == "super":
        EDIT_MODIFIERS.update({keyboard.Key.cmd_l, keyboard.Key.cmd_r})
if not EDIT_MODIFIERS:
    EDIT_MODIFIERS = {keyboard.Key.ctrl_l, keyboard.Key.ctrl_r, keyboard.Key.alt_l, keyboard.Key.alt_r}
_edit_modifier_types = set(_edit_modifier_names) if _edit_modifier_names else {"ctrl", "alt"}

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


def check_environment():
    """Check for required dependencies and environment."""
    errors = []

    # Check for API key based on backend
    if BACKEND == "groq":
        if not os.environ.get("GROQ_API_KEY"):
            errors.append("GROQ_API_KEY environment variable not set")
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            errors.append("OPENAI_API_KEY environment variable not set")

    # Check for xdotool
    if not shutil.which("xdotool"):
        errors.append("xdotool not found. Install with: sudo pacman -S xdotool")

    # Check for ffmpeg (optional but recommended for MP3 compression)
    if not shutil.which("ffmpeg"):
        print("\033[93mWarning: ffmpeg not found. Using uncompressed WAV (slower uploads).\033[0m")
        print("Install with: sudo pacman -S ffmpeg\n")

    # Warn about Wayland
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session_type == "wayland":
        print(
            "\033[93mWarning: Running on Wayland. xdotool may not work correctly.\n"
            "Consider using X11 or ydotool for Wayland.\033[0m\n"
        )

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


# Silence detection threshold (RMS value for 16-bit audio)
SILENCE_THRESHOLD = CONFIG.get("audio", {}).get("silence_threshold", 150)


class VoiceRecorder:
    def __init__(self):
        self.recording = False
        self.audio_data = []
        self.stream = None
        self.pressed_modifiers = set()
        self.hotkey_pressed = False
        self.has_ffmpeg = shutil.which("ffmpeg") is not None
        self.submit_mode = False  # Whether to press Enter after typing
        self.edit_mode = False  # Whether to correct last transcription
        self.last_typed_text = ""  # Store for edit mode corrections
        self.active_window_id = None  # Window to type into

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
            # Create fresh client (avoids thread-safety issues)
            if BACKEND == "groq":
                from groq import Groq
                client = Groq()
            else:
                from openai import OpenAI
                client = OpenAI()
            response = client.chat.completions.create(
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
        # Capture active window for later focus restoration
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True,
                text=True,
                check=True,
            )
            self.active_window_id = result.stdout.strip()
        except Exception:
            self.active_window_id = None
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
                device="default",
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
            target=self._transcribe_and_type, args=(audio, self.submit_mode, self.edit_mode)
        ).start()

    def _transcribe_and_type(self, audio: np.ndarray, submit: bool = False, edit: bool = False):
        try:
            # Check for silence before sending
            rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
            if rms < SILENCE_THRESHOLD:
                print("(silence detected, skipping)")
                return

            # Check internet connectivity
            if not check_connectivity():
                print("\033[91mNo internet connection\033[0m")
                # Save audio to temp file for potential recovery (don't overwrite existing)
                recovery_path = Path("/tmp/linux-voice-recovery.wav")
                if recovery_path.exists():
                    # Don't overwrite - just remind user to recover first
                    subprocess.run(
                        ["xdotool", "type", "--delay", "0", "--", "(no internet - say 'recover' first)"],
                        check=False,
                    )
                else:
                    with wave.open(str(recovery_path), "wb") as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(2)
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes(audio.tobytes())
                    print(f"Audio saved to {recovery_path}")
                    subprocess.run(
                        ["xdotool", "type", "--delay", "0", "--", "(no internet - say 'recover' to retry)"],
                        check=False,
                    )
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

            # Create a fresh client for this request (avoids thread-safety issues with shared client)
            if BACKEND == "groq":
                from groq import Groq
                client = Groq()
            else:
                from openai import OpenAI
                client = OpenAI()

            # Transcribe with timeout
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(
                    client.audio.transcriptions.create,
                    model=WHISPER_MODEL,
                    file=file_buffer,
                    language=LANGUAGE,
                    prompt=PROMPT,
                    response_format="text",
                    temperature=0.0,
                )
                transcript = future.result(timeout=API_TIMEOUT)
            except FuturesTimeoutError:
                executor.shutdown(wait=False)
                print(f"\033[91mAPI timeout after {API_TIMEOUT}s\033[0m")
                recovery_path = Path("/tmp/linux-voice-recovery.wav")
                with wave.open(str(recovery_path), "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(audio.tobytes())
                print(f"Audio saved to {recovery_path}")
                subprocess.run(
                    ["xdotool", "type", "--delay", "0", "--", "(timeout - say 'recover' to retry)"],
                    check=False,
                )
                return
            finally:
                executor.shutdown(wait=False)

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
            subprocess.run(
                ["xdotool", "keyup", "ctrl", "shift", "alt", "super"],
                check=False,
            )

            # Restore focus to the original window (may have changed during API call)
            if self.active_window_id:
                subprocess.run(
                    ["xdotool", "windowactivate", "--sync", self.active_window_id],
                    check=False,
                )

            if edit:
                # Edit mode: use transcription as instruction to correct previous text
                instruction = text
                print(f"\033[93m◌ Correcting: {instruction}\033[0m", flush=True)

                corrected = self._correct_with_llm(self.last_typed_text, instruction)
                print(f"\033[94m→ {corrected}\033[0m")

                # Clear the line with Ctrl+U (works in terminals and many input fields)
                subprocess.run(
                    ["xdotool", "key", "ctrl+u"],
                    check=False,
                )

                # Type corrected text
                subprocess.run(
                    ["xdotool", "type", "--delay", "0", "--", corrected],
                    check=True,
                )
                self.last_typed_text = corrected
            else:
                # Normal mode: apply replacements and type
                text = apply_replacements(text)
                print(f"\033[94m→ {text}\033[0m")

                subprocess.run(
                    ["xdotool", "type", "--delay", "0", "--", text],
                    check=True,
                )
                self.last_typed_text = text

                # Press Enter if submit mode
                if submit:
                    import time
                    time.sleep(SUBMIT_DELAY / 1000)
                    subprocess.run(["xdotool", "key", "Return"], check=True)

        except FileNotFoundError:
            print("\033[91mError: xdotool not found. Install with: sudo pacman -S xdotool\033[0m")
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m")

    def recover_audio(self):
        """Recover and transcribe audio from a failed attempt."""
        recovery_path = Path("/tmp/linux-voice-recovery.wav")
        if not recovery_path.exists():
            print("No recovery file found")
            subprocess.run(
                ["xdotool", "type", "--delay", "0", "--", "(no recovery file)"],
                check=False,
            )
            return

        # Load audio from recovery file
        with wave.open(str(recovery_path), "rb") as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

        print(f"Recovering audio from {recovery_path}...")

        # Clear the "no internet" message first
        subprocess.run(["xdotool", "key", "ctrl+u"], check=False)

        # Transcribe in background
        threading.Thread(
            target=self._transcribe_and_type, args=(audio, False, False)
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

        try:
            with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release,
            ) as listener:
                listener.join()
        except Exception as e:
            print(f"\033[91mKeyboard listener error: {e}\033[0m")
            print("Make sure you're running on X11 with proper permissions.")
            sys.exit(1)


def recover():
    """Transcribe the recovery audio file from a failed attempt."""
    recovery_path = Path("/tmp/linux-voice-recovery.wav")
    if not recovery_path.exists():
        print("No recovery file found")
        return

    check_environment()

    # Load audio from recovery file
    with wave.open(str(recovery_path), "rb") as wf:
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    print(f"Recovering audio from {recovery_path}...")
    recorder = VoiceRecorder()
    recorder._transcribe_and_type(audio, submit=False, edit=False)

    # Remove recovery file after successful transcription
    recovery_path.unlink()
    print("Recovery file removed")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--recover":
        recover()
        return

    check_environment()
    recorder = VoiceRecorder()
    try:
        recorder.run()
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
