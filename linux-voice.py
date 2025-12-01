#!/usr/bin/env python3
"""
linux-voice: Voice-to-text dictation tool for Linux (X11)

Hold Ctrl+Space to record, release to transcribe and type.
Toggle mode: Press Ctrl+Space to start, press again to stop.
"""

import io
import os
import subprocess
import threading
import wave

import numpy as np
import sounddevice as sd
from openai import OpenAI
from pynput import keyboard

# Configuration
SAMPLE_RATE = 48000  # Most devices support 48kHz
CHANNELS = 1
HOTKEY_KEY = keyboard.Key.space
HOTKEY_MODIFIERS = {keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}
MODE = os.environ.get("LINUX_VOICE_MODE", "hold")  # "hold" or "toggle"


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
        self.recording = True
        self.audio_data = []
        print("\033[92m● Recording...\033[0m", flush=True)

        def callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            if self.recording:
                self.audio_data.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.int16,
            device="default",
            callback=callback,
        )
        self.stream.start()

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("\033[93m◌ Processing...\033[0m", flush=True)

        if not self.audio_data:
            print("No audio recorded")
            return

        # Combine audio chunks
        audio = np.concatenate(self.audio_data, axis=0)

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
                language="en",
                response_format="text",
            )

            text = transcript.strip()
            if not text:
                print("(no speech detected)")
                return

            print(f"\033[94m→ {text}\033[0m")

            # Release any stuck modifiers before typing
            subprocess.run(["xdotool", "keyup", "ctrl", "shift", "alt", "super"], check=False)
            subprocess.run(
                ["xdotool", "type", "--", text],
                check=True,
            )

        except Exception as e:
            print(f"\033[91mError: {e}\033[0m")

    def on_press(self, key):
        # Track modifier keys
        if key in HOTKEY_MODIFIERS:
            self.pressed_modifiers.add(key)

        # Check if hotkey combo is pressed
        if key == HOTKEY_KEY and (
            self.pressed_modifiers & HOTKEY_MODIFIERS
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

        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        ) as listener:
            listener.join()


def main():
    recorder = VoiceRecorder()
    try:
        recorder.run()
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
