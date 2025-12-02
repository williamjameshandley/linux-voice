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
from openai import OpenAI
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


def check_environment():
    """Check for required dependencies and environment."""
    errors = []

    # Check for OPENAI_API_KEY
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
        self.client = OpenAI()
        self.recording = False
        self.audio_data = []
        self.stream = None
        self.pressed_modifiers = set()
        self.hotkey_pressed = False
        self.has_ffmpeg = shutil.which("ffmpeg") is not None
        self.last_transcript = ""  # Context for next transcription
        self.submit_mode = False  # Whether to press Enter after typing

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

    def start_recording(self, submit=False):
        if self.recording:
            return
        self.audio_data = []
        self.submit_mode = submit
        indicator = "● Recording..." if not submit else "● Recording (submit)..."
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
            target=self._transcribe_and_type, args=(audio, self.submit_mode)
        ).start()

    def _transcribe_and_type(self, audio: np.ndarray, submit: bool = False):
        try:
            # Check for silence before sending
            rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
            if rms < SILENCE_THRESHOLD:
                print("(silence detected, skipping)")
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

            # Build prompt with base prompt and recent context
            context = self.last_transcript[-200:] if self.last_transcript else ""
            full_prompt = f"{PROMPT} {context}".strip()

            # Transcribe with OpenAI Whisper
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=file_buffer,
                language=LANGUAGE,
                prompt=full_prompt,
                response_format="text",
                temperature=0.0,
            )

            text = transcript.strip()
            if not text:
                print("(no speech detected)")
                return

            # Update context for next transcription
            self.last_transcript += " " + text
            if len(self.last_transcript) > 1000:
                self.last_transcript = self.last_transcript[-500:]

            print(f"\033[94m→ {text}\033[0m")

            # Release any stuck modifiers before typing
            subprocess.run(
                ["xdotool", "keyup", "ctrl", "shift", "alt", "super"],
                check=False,
            )
            subprocess.run(
                ["xdotool", "type", "--delay", "0", "--", text],
                check=True,
            )

            # Press Enter if submit mode
            if submit:
                import time
                time.sleep(SUBMIT_DELAY / 1000)
                subprocess.run(["xdotool", "key", "Return"], check=True)

        except FileNotFoundError:
            print("\033[91mError: xdotool not found. Install with: sudo pacman -S xdotool\033[0m")
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m")

    def on_press(self, key):
        # Track modifier keys (include both hotkey and submit modifiers)
        all_modifiers = HOTKEY_MODIFIERS | SUBMIT_MODIFIERS
        if key in all_modifiers:
            self.pressed_modifiers.add(key)

        # Check if submit hotkey is pressed (Alt+Space or configured)
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
        all_modifiers = HOTKEY_MODIFIERS | SUBMIT_MODIFIERS
        if key in all_modifiers:
            self.pressed_modifiers.discard(key)

        # In hold mode, stop when space is released (works for both hotkeys)
        if MODE == "hold" and key in (HOTKEY_KEY, SUBMIT_KEY) and self.hotkey_pressed:
            self.hotkey_pressed = False
            self.stop_recording()

    def run(self):
        # Format hotkey names for display
        hotkey_str = "+".join(m.capitalize() for m in _required_modifier_types) + "+Space"
        submit_str = "+".join(m.capitalize() for m in _submit_modifier_types) + "+Space"

        print(f"linux-voice started (mode: {MODE})")
        print(f"  {hotkey_str}: {'toggle' if MODE == 'toggle' else 'hold to'} record")
        print(f"  {submit_str}: record and submit (press Enter)")
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


def main():
    check_environment()
    recorder = VoiceRecorder()
    try:
        recorder.run()
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
