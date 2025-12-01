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
SAMPLE_RATE = CONFIG.get("audio", {}).get("sample_rate", 48000)
CHANNELS = 1
LANGUAGE = CONFIG.get("transcription", {}).get("language", "en")
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


def check_environment():
    """Check for required dependencies and environment."""
    errors = []

    # Check for OPENAI_API_KEY
    if not os.environ.get("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY environment variable not set")

    # Check for xdotool
    if not shutil.which("xdotool"):
        errors.append("xdotool not found. Install with: sudo pacman -S xdotool")

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


class VoiceRecorder:
    def __init__(self):
        self.client = OpenAI()
        self.recording = False
        self.audio_data = []
        self.stream = None
        self.pressed_modifiers = set()
        self.hotkey_pressed = False

    def start_recording(self):
        if self.recording:
            return
        self.audio_data = []
        print("\033[92m● Recording...\033[0m", flush=True)

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
        threading.Thread(target=self._transcribe_and_type, args=(audio,)).start()

    def _transcribe_and_type(self, audio: np.ndarray):
        try:
            # Save to WAV in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio.tobytes())
            wav_buffer.seek(0)
            wav_buffer.name = "audio.wav"

            # Transcribe with OpenAI Whisper
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=wav_buffer,
                language=LANGUAGE,
                response_format="text",
            )

            text = transcript.strip()
            if not text:
                print("(no speech detected)")
                return

            print(f"\033[94m→ {text}\033[0m")

            # Release any stuck modifiers before typing
            subprocess.run(
                ["xdotool", "keyup", "ctrl", "shift", "alt", "super"],
                check=False,
            )
            subprocess.run(
                ["xdotool", "type", "--", text],
                check=True,
            )

        except FileNotFoundError:
            print("\033[91mError: xdotool not found. Install with: sudo pacman -S xdotool\033[0m")
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m")

    def on_press(self, key):
        # Track modifier keys
        if key in HOTKEY_MODIFIERS:
            self.pressed_modifiers.add(key)

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
        if key in HOTKEY_MODIFIERS:
            self.pressed_modifiers.discard(key)

        # In hold mode, stop when space is released
        if MODE == "hold" and key == HOTKEY_KEY and self.hotkey_pressed:
            self.hotkey_pressed = False
            self.stop_recording()

    def run(self):
        print(f"linux-voice started (mode: {MODE})")
        print(f"Press Ctrl+Space to {'toggle' if MODE == 'toggle' else 'hold and'} record")
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
