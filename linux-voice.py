#!/usr/bin/env python3
"""
voice-to-text: Voice-to-text dictation tool for macOS and Linux

Hold Ctrl+Space to record, release to transcribe and type.
Toggle mode: Press Ctrl+Space to start, press again to stop.

Features:
- Multiple hotkeys: record, submit (Enter), edit, clipboard
- Voice commands: scratch that, recover, open <app>
- Context-aware prompts based on focused application
- Configurable voice snippets
- Audio feedback
- Transcription history

Requires Python 3.11+ for tomllib.
On macOS: Requires Accessibility permissions for Terminal/IDE.
On Linux: Requires X11 and xdotool.
"""

import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

# Platform detection
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

import numpy as np
import sounddevice as sd
from pynput import keyboard

# Minimum recording duration in seconds (avoid accidental triggers)
MIN_RECORDING_SECONDS = 0.3

# History file location
HISTORY_PATH = Path.home() / ".config" / "linux-voice" / "history.jsonl"

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

# Audio feedback configuration
AUDIO_FEEDBACK = CONFIG.get("audio", {}).get("feedback", True)

# Voice snippets (configurable text expansions)
SNIPPETS = CONFIG.get("snippets", {
    "my email": "user@example.com",
    "my phone": "+1-555-555-5555",
})



# Backend configuration (openai or groq)
BACKEND = CONFIG.get("transcription", {}).get("backend", "openai")
WHISPER_MODEL = CONFIG.get("transcription", {}).get("model", None)
if WHISPER_MODEL is None:
    WHISPER_MODEL = "whisper-large-v3-turbo" if BACKEND == "groq" else "whisper-1"
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

# Submit hotkey (Alt+Shift+Space by default) - same as main hotkey but presses Enter after
_submit_cfg = CONFIG.get("hotkey_submit", {})
SUBMIT_DELAY = _submit_cfg.get("delay", 150)  # ms delay before Enter key
SUBMIT_KEY = getattr(
    keyboard.Key,
    _submit_cfg.get("key", "space"),
    keyboard.KeyCode.from_char(_submit_cfg.get("key", "space")),
)
_submit_modifier_names = _submit_cfg.get("modifiers", ["alt", "shift"])
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
    SUBMIT_MODIFIERS = {keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.shift_l, keyboard.Key.shift_r}
_submit_modifier_types = set(_submit_modifier_names) if _submit_modifier_names else {"alt", "shift"}

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

# Clipboard hotkey (Ctrl+Shift+Space by default) - copies to clipboard instead of typing
_clipboard_cfg = CONFIG.get("hotkey_clipboard", {})
_clipboard_key_name = _clipboard_cfg.get("key", "space")
CLIPBOARD_KEY = getattr(
    keyboard.Key,
    _clipboard_key_name,
    keyboard.KeyCode.from_char(_clipboard_key_name),
)
_clipboard_modifier_names = _clipboard_cfg.get("modifiers", ["ctrl", "shift"])
CLIPBOARD_MODIFIERS = set()
for mod in _clipboard_modifier_names:
    if mod == "ctrl":
        CLIPBOARD_MODIFIERS.update({keyboard.Key.ctrl_l, keyboard.Key.ctrl_r})
    elif mod == "alt":
        CLIPBOARD_MODIFIERS.update({keyboard.Key.alt_l, keyboard.Key.alt_r})
    elif mod == "shift":
        CLIPBOARD_MODIFIERS.update({keyboard.Key.shift_l, keyboard.Key.shift_r})
    elif mod == "super":
        CLIPBOARD_MODIFIERS.update({keyboard.Key.cmd_l, keyboard.Key.cmd_r})
if not CLIPBOARD_MODIFIERS:
    CLIPBOARD_MODIFIERS = {keyboard.Key.ctrl_l, keyboard.Key.ctrl_r, keyboard.Key.shift_l, keyboard.Key.shift_r}
_clipboard_modifier_types = set(_clipboard_modifier_names) if _clipboard_modifier_names else {"ctrl", "shift"}

# LLM model for corrections (per backend)
LLM_MODEL = CONFIG.get("transcription", {}).get("llm_model", None)
if LLM_MODEL is None:
    LLM_MODEL = "llama-3.3-70b-versatile" if BACKEND == "groq" else "gpt-4o-mini"


# ============================================================================
# Audio Feedback
# ============================================================================

def play_sound(sound_type: str) -> None:
    """Play audio feedback sound."""
    if not AUDIO_FEEDBACK:
        return

    if IS_MACOS:
        # Use macOS system sounds
        sounds = {
            "start": "/System/Library/Sounds/Pop.aiff",
            "stop": "/System/Library/Sounds/Blow.aiff",
            "success": "/System/Library/Sounds/Glass.aiff",
            "error": "/System/Library/Sounds/Basso.aiff",
        }
        sound_file = sounds.get(sound_type)
        if sound_file and Path(sound_file).exists():
            subprocess.Popen(
                ["afplay", sound_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    elif IS_LINUX:
        # Use paplay or aplay if available
        if shutil.which("paplay"):
            sounds = {
                "start": "/usr/share/sounds/freedesktop/stereo/message.oga",
                "stop": "/usr/share/sounds/freedesktop/stereo/complete.oga",
                "success": "/usr/share/sounds/freedesktop/stereo/complete.oga",
                "error": "/usr/share/sounds/freedesktop/stereo/dialog-error.oga",
            }
            sound_file = sounds.get(sound_type)
            if sound_file and Path(sound_file).exists():
                subprocess.Popen(
                    ["paplay", sound_file],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )


# ============================================================================
# Clipboard Operations
# ============================================================================

def copy_to_clipboard(text: str) -> None:
    """Copy text to system clipboard."""
    if IS_MACOS:
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
    elif IS_LINUX:
        # Try xclip first, then xsel
        if shutil.which("xclip"):
            subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=True)
        elif shutil.which("xsel"):
            subprocess.run(["xsel", "--clipboard", "--input"], input=text.encode(), check=True)


# ============================================================================
# History Logging
# ============================================================================

def log_transcription(text: str, app: str | None = None, mode: str = "normal") -> None:
    """Log transcription to history file."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "text": text,
        "app": app,
        "mode": mode,
    }
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ============================================================================
# App Launcher
# ============================================================================

def open_application(app_name: str) -> bool:
    """Open an application by name."""
    if IS_MACOS:
        # Try to open the app
        result = subprocess.run(
            ["open", "-a", app_name],
            capture_output=True,
        )
        return result.returncode == 0
    elif IS_LINUX:
        # Try common launchers
        for launcher in [app_name.lower(), f"{app_name.lower()}.desktop"]:
            if shutil.which(launcher):
                subprocess.Popen([launcher], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
        # Try gtk-launch
        if shutil.which("gtk-launch"):
            result = subprocess.run(
                ["gtk-launch", app_name.lower()],
                capture_output=True,
            )
            return result.returncode == 0
    return False


def apply_snippets(text: str) -> str:
    """Apply voice snippets (text expansions)."""
    text_lower = text.lower()
    for trigger, expansion in SNIPPETS.items():
        if trigger.lower() in text_lower:
            # Case-insensitive replacement
            pattern = re.escape(trigger)
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    return text


def apply_replacements(text: str) -> str:
    """Apply all text transformations: snippets, user replacements."""
    # Apply voice snippets
    text = apply_snippets(text)
    # Apply user-configured regex replacements
    for pattern, replacement in REPLACEMENTS.items():
        try:
            text = re.sub(pattern, replacement, text)
        except re.error as e:
            print(f"\033[91mBad regex pattern '{pattern}': {e}\033[0m")
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

    if IS_MACOS:
        # Check for osascript (should always be present on macOS)
        if not shutil.which("osascript"):
            errors.append("osascript not found (should be built into macOS)")

        # Check for ffmpeg (optional but recommended for MP3 compression)
        if not shutil.which("ffmpeg"):
            print("\033[93mWarning: ffmpeg not found. Using uncompressed WAV (slower uploads).\033[0m")
            print("Install with: brew install ffmpeg\n")

        # Remind about Accessibility permissions
        print("\033[93mNote: Ensure Terminal/IDE has Accessibility permissions.\033[0m")
        print("System Preferences > Security & Privacy > Privacy > Accessibility\n")

    elif IS_LINUX:
        # Check for xdotool
        if not shutil.which("xdotool"):
            errors.append("xdotool not found. Install with: sudo apt install xdotool")

        # Check for ffmpeg (optional but recommended for MP3 compression)
        if not shutil.which("ffmpeg"):
            print("\033[93mWarning: ffmpeg not found. Using uncompressed WAV (slower uploads).\033[0m")
            print("Install with: sudo apt install ffmpeg\n")

        # Warn about Wayland
        session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if session_type == "wayland":
            print(
                "\033[93mWarning: Running on Wayland. xdotool may not work correctly.\n"
                "Consider using X11 or ydotool for Wayland.\033[0m\n"
            )
    else:
        errors.append(f"Unsupported platform: {platform.system()}")

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


# ============================================================================
# Platform-specific input simulation functions
# ============================================================================

def _escape_applescript_string(text: str) -> str:
    """Escape special characters for AppleScript strings."""
    # Escape backslashes first, then quotes
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    return text


def type_text(text: str) -> None:
    """Type text into the active window using platform-specific method."""
    if IS_MACOS:
        # Use AppleScript to type text character by character for reliability
        # Split into chunks to handle special characters better
        escaped = _escape_applescript_string(text)
        script = f'tell application "System Events" to keystroke "{escaped}"'
        subprocess.run(["osascript", "-e", script], check=True)
    else:
        # Linux: use xdotool
        subprocess.run(["xdotool", "type", "--delay", "0", "--", text], check=True)


def get_active_window() -> str | None:
    """Get the active window/application identifier."""
    if IS_MACOS:
        try:
            result = subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to get name of first application process whose frontmost is true'],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return None
    else:
        # Linux: use xdotool
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return None


def activate_window(window_id: str) -> None:
    """Activate/focus a window by its identifier."""
    if IS_MACOS:
        # window_id is the process name on macOS - use System Events to activate
        escaped = _escape_applescript_string(window_id)
        script = f'''
            tell application "System Events"
                set frontmost of process "{escaped}" to true
            end tell
        '''
        subprocess.run(["osascript", "-e", script], check=False,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        # Linux: use xdotool
        subprocess.run(
            ["xdotool", "windowactivate", "--sync", window_id],
            check=False,
        )


def press_key(key: str) -> None:
    """Press a specific key (e.g., 'Return', 'Escape')."""
    if IS_MACOS:
        # Map common key names to AppleScript key codes
        key_map = {
            "Return": "return",
            "Enter": "return",
            "Escape": "escape",
            "Tab": "tab",
            "Delete": "delete",
            "Backspace": "delete",
        }
        key_name = key_map.get(key, key.lower())
        script = f'tell application "System Events" to keystroke {key_name}'
        subprocess.run(["osascript", "-e", script], check=True)
    else:
        # Linux: use xdotool
        subprocess.run(["xdotool", "key", key], check=True)


def clear_line() -> None:
    """Clear the current line (Ctrl+U equivalent)."""
    if IS_MACOS:
        # Use Command+A to select all text in the current field, then delete
        # Actually Ctrl+U works in terminals on macOS too
        script = 'tell application "System Events" to keystroke "u" using control down'
        subprocess.run(["osascript", "-e", script], check=False)
    else:
        # Linux: use xdotool
        subprocess.run(["xdotool", "key", "ctrl+u"], check=False)


def release_modifiers() -> None:
    """Release any stuck modifier keys."""
    if IS_MACOS:
        # On macOS, we don't need to explicitly release modifiers
        # AppleScript handles key state independently
        pass
    else:
        # Linux: use xdotool
        subprocess.run(
            ["xdotool", "keyup", "ctrl", "shift", "alt", "super"],
            check=False,
        )


class VoiceRecorder:
    def __init__(self):
        if BACKEND == "groq":
            from groq import Groq
            self.client = Groq()
        else:
            from openai import OpenAI
            self.client = OpenAI()
        self.recording = False
        self.audio_data = []
        self.stream = None
        self.pressed_modifiers = set()
        self.hotkey_pressed = False
        self.has_ffmpeg = shutil.which("ffmpeg") is not None
        self.submit_mode = False  # Whether to press Enter after typing
        self.edit_mode = False  # Whether to correct last transcription
        self.clipboard_mode = False  # Whether to copy to clipboard instead of typing
        self.last_typed_text = ""  # Store for edit mode corrections
        self.last_typed_length = 0  # Character count for scratch that
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

    def _add_to_gtd_inbox(self, task_text: str) -> None:
        """Add a task to the GTD inbox with notification."""
        gtd_root = os.path.expanduser("~/gtd")
        gtd_bin = os.path.expanduser("~/bin/gtd")
        env = os.environ.copy()
        env["GTD_ROOT"] = gtd_root

        try:
            result = subprocess.run(
                [gtd_bin, "inbox", "add", task_text],
                env=env,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                # Success notification
                escaped_text = task_text.replace('"', '\\"')
                subprocess.run(
                    ["osascript", "-e",
                     f'display notification "{escaped_text}" with title "GTD Inbox" sound name "Glass"'],
                    check=False,
                )
                play_sound("success")
                print(f"\033[92m✓ Added to GTD inbox: {task_text}\033[0m")
            else:
                # Error notification
                subprocess.run(
                    ["osascript", "-e",
                     'display notification "Failed to add task" with title "GTD Error" sound name "Basso"'],
                    check=False,
                )
                play_sound("error")
                print(f"\033[91m✗ GTD error: {result.stderr}\033[0m")
        except FileNotFoundError:
            subprocess.run(
                ["osascript", "-e",
                 'display notification "gtd command not found" with title "GTD Error" sound name "Basso"'],
                check=False,
            )
            play_sound("error")
            print("\033[91m✗ gtd command not found\033[0m")
        except Exception as e:
            play_sound("error")
            print(f"\033[91m✗ GTD error: {e}\033[0m")

    def _scratch_that(self):
        """Delete the last typed text by sending backspaces."""
        if self.last_typed_length == 0:
            print("(nothing to scratch)")
            return

        if IS_MACOS:
            # Send backspace key for each character
            for _ in range(self.last_typed_length):
                script = 'tell application "System Events" to key code 51'  # 51 = delete/backspace
                subprocess.run(["osascript", "-e", script], check=False,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # Linux: use xdotool to send backspaces
            subprocess.run(
                ["xdotool", "key", "--repeat", str(self.last_typed_length), "BackSpace"],
                check=False,
            )

        print(f"(scratched {self.last_typed_length} characters)")
        self.last_typed_text = ""
        self.last_typed_length = 0

    def start_recording(self, submit=False, edit=False, clipboard=False):
        if self.recording:
            return
        if edit and not self.last_typed_text:
            print("\033[91mNo previous text to edit\033[0m", flush=True)
            play_sound("error")
            return
        self.audio_data = []
        self.submit_mode = submit
        self.edit_mode = edit
        self.clipboard_mode = clipboard
        # Capture active window for later focus restoration
        self.active_window_id = get_active_window()
        if edit:
            indicator = "● Recording edit instruction..."
        elif clipboard:
            indicator = "● Recording (clipboard)..."
        elif submit:
            indicator = "● Recording (submit)..."
        else:
            indicator = "● Recording..."
        print(f"\033[92m{indicator}\033[0m", flush=True)
        play_sound("start")

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
                device=None,  # Use system default input device
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
        play_sound("stop")

        # Transcribe in background to not block hotkey listener
        threading.Thread(
            target=self._transcribe_and_type,
            args=(audio, self.submit_mode, self.edit_mode, self.clipboard_mode)
        ).start()

    def _transcribe_and_type(self, audio: np.ndarray, submit: bool = False, edit: bool = False, clipboard: bool = False):
        try:
            # Check for silence before sending
            rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
            if rms < SILENCE_THRESHOLD:
                print("(silence detected, skipping)")
                return

            # Check internet connectivity
            if not check_connectivity():
                print("\033[91mNo internet connection\033[0m")
                play_sound("error")
                # Save audio to temp file for potential recovery (don't overwrite existing)
                recovery_path = Path("/tmp/linux-voice-recovery.wav")
                if recovery_path.exists():
                    # Don't overwrite - just remind user to recover first
                    try:
                        type_text("(no internet - say 'recover' first)")
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
                        type_text("(no internet - say 'recover' to retry)")
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

            text = transcript.strip()
            if not text:
                print("(no speech detected)")
                return

            text_lower = text.lower().rstrip(".")

            # Check for voice commands
            if text_lower in ("recover", "recovery"):
                print("Voice command: recover")
                self.recover_audio()
                return

            if text_lower in ("scratch that", "scratch this", "undo", "undo that", "delete that"):
                print("Voice command: scratch that")
                self._scratch_that()
                play_sound("success")
                return

            # Check for app launcher command: "open <app>"
            if text_lower.startswith("open "):
                app_name = text[5:].strip().rstrip(".")
                print(f"Voice command: open {app_name}")
                if open_application(app_name):
                    play_sound("success")
                    print(f"\033[92m✓ Opened {app_name}\033[0m")
                else:
                    play_sound("error")
                    print(f"\033[91m✗ Could not open {app_name}\033[0m")
                return

            # Check for GTD entry command: "GTD entry <task>" (handles various transcriptions)
            gtd_match = re.match(r"^g[\s.]*t[\s.]*d[\s.]*entr(?:y|ies?)[:\s,]+(.+)", text_lower)
            if gtd_match:
                task_text = gtd_match.group(1).strip().rstrip(".")
                print(f"Voice command: GTD entry '{task_text}'")
                self._add_to_gtd_inbox(task_text)
                return

            # Release any stuck modifiers before typing
            release_modifiers()

            # Restore focus to the original window (may have changed during API call)
            if self.active_window_id:
                activate_window(self.active_window_id)

            if edit:
                # Edit mode: use transcription as instruction to correct previous text
                instruction = text
                print(f"\033[93m◌ Correcting: {instruction}\033[0m", flush=True)

                corrected = self._correct_with_llm(self.last_typed_text, instruction)
                print(f"\033[94m→ {corrected}\033[0m")

                # Clear the line with Ctrl+U (works in terminals and many input fields)
                clear_line()

                # Type corrected text
                type_text(corrected)
                self.last_typed_text = corrected
                self.last_typed_length = len(corrected)
                log_transcription(corrected, self.active_window_id, "edit")
            elif clipboard:
                # Clipboard mode: copy to clipboard instead of typing
                text = apply_replacements(text)
                print(f"\033[94m→ [clipboard] {text}\033[0m")
                copy_to_clipboard(text)
                self.last_typed_text = text
                self.last_typed_length = 0  # Not typed, so nothing to scratch
                log_transcription(text, self.active_window_id, "clipboard")
            else:
                # Normal mode: apply replacements and type
                text = apply_replacements(text)
                print(f"\033[94m→ {text}\033[0m")

                type_text(text)
                self.last_typed_text = text
                self.last_typed_length = len(text)
                log_transcription(text, self.active_window_id, "normal")

                # Press Enter if submit mode
                if submit:
                    time.sleep(SUBMIT_DELAY / 1000)
                    press_key("Return")
                    self.last_typed_length += 1  # Account for newline

            play_sound("success")

        except FileNotFoundError:
            if IS_MACOS:
                print("\033[91mError: osascript not found (should be built into macOS)\033[0m")
            else:
                print("\033[91mError: xdotool not found. Install with: sudo apt install xdotool\033[0m")
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m")

    def recover_audio(self):
        """Recover and transcribe audio from a failed attempt."""
        recovery_path = Path("/tmp/linux-voice-recovery.wav")
        if not recovery_path.exists():
            print("No recovery file found")
            try:
                type_text("(no recovery file)")
            except Exception:
                pass
            return

        # Load audio from recovery file
        with wave.open(str(recovery_path), "rb") as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

        print(f"Recovering audio from {recovery_path}...")

        # Clear the "no internet" message first
        clear_line()

        # Transcribe in background
        threading.Thread(
            target=self._transcribe_and_type, args=(audio, False, False, False)
        ).start()

        # Remove recovery file
        recovery_path.unlink()

    def on_press(self, key):
        # Track modifier keys (include all hotkey modifiers)
        all_modifiers = HOTKEY_MODIFIERS | SUBMIT_MODIFIERS | EDIT_MODIFIERS | CLIPBOARD_MODIFIERS
        if key in all_modifiers:
            self.pressed_modifiers.add(key)

        # Check if clipboard hotkey is pressed (Ctrl+Shift+Alt+Space or configured)
        # Check this FIRST since it has the most modifiers
        if key == CLIPBOARD_KEY and _has_all_modifiers(
            self.pressed_modifiers, _clipboard_modifier_types
        ):
            if MODE == "toggle":
                if self.recording:
                    self.stop_recording()
                else:
                    self.start_recording(clipboard=True)
            else:  # hold mode
                if not self.hotkey_pressed:
                    self.hotkey_pressed = True
                    self.start_recording(clipboard=True)
            return

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
        all_modifiers = HOTKEY_MODIFIERS | SUBMIT_MODIFIERS | EDIT_MODIFIERS | CLIPBOARD_MODIFIERS
        if key in all_modifiers:
            self.pressed_modifiers.discard(key)

        # In hold mode, stop when space is released (works for all hotkeys)
        if MODE == "hold" and key in (HOTKEY_KEY, SUBMIT_KEY, EDIT_KEY, CLIPBOARD_KEY) and self.hotkey_pressed:
            self.hotkey_pressed = False
            self.stop_recording()

    def run(self):
        # Format hotkey names for display
        hotkey_str = "+".join(m.capitalize() for m in _required_modifier_types) + "+Space"
        submit_str = "+".join(m.capitalize() for m in _submit_modifier_types) + "+Space"
        edit_str = "+".join(m.capitalize() for m in _edit_modifier_types) + "+Space"
        clipboard_str = "+".join(m.capitalize() for m in sorted(_clipboard_modifier_types)) + "+Space"

        platform_name = "macOS" if IS_MACOS else "Linux"
        print(f"voice-to-text started ({platform_name}, mode: {MODE}, backend: {BACKEND})")
        print(f"  {hotkey_str}: {'toggle' if MODE == 'toggle' else 'hold to'} record")
        print(f"  {submit_str}: record and submit (press Enter)")
        print(f"  {edit_str}: record correction instruction")
        print(f"  {clipboard_str}: record to clipboard")
        print("\nVoice commands: 'scratch that', 'open <app>', 'GTD entry <task>', 'recover'")
        print("Press Ctrl+C to exit\n")

        try:
            with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release,
            ) as listener:
                listener.join()
        except Exception as e:
            print(f"\033[91mKeyboard listener error: {e}\033[0m")
            if IS_MACOS:
                print("Make sure Terminal/IDE has Accessibility permissions.")
                print("System Preferences > Security & Privacy > Privacy > Accessibility")
            else:
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
    recorder._transcribe_and_type(audio, submit=False, edit=False, clipboard=False)

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
