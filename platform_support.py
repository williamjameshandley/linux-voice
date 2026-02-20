"""
Platform abstraction for linux-voice.

Provides a common interface for platform-specific operations:
- Text injection into focused windows
- Window focus capture and restoration
- Modifier key release
- Environment validation
"""

import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod


class PlatformInterface(ABC):
    """Abstract interface for platform-specific operations."""

    @abstractmethod
    def get_active_app(self):
        """Capture the currently focused application/window.

        Returns an opaque handle that can be passed to restore_focus().
        """

    @abstractmethod
    def restore_focus(self, app_handle):
        """Restore focus to a previously captured application/window."""

    @abstractmethod
    def type_text(self, text: str):
        """Type text into the currently focused window."""

    @abstractmethod
    def press_key(self, key_name: str):
        """Press and release a key (e.g., 'Return')."""

    @abstractmethod
    def release_modifiers(self):
        """Release all modifier keys to prevent stuck state."""

    @abstractmethod
    def clear_line(self):
        """Clear the current input line."""

    @abstractmethod
    def check_environment(self) -> list[str]:
        """Validate platform-specific requirements.

        Returns a list of error messages. Empty list means all checks passed.
        Also prints warnings for non-fatal issues.
        """


class LinuxX11(PlatformInterface):
    """Linux/X11 implementation using xdotool."""

    def get_active_app(self):
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

    def restore_focus(self, app_handle):
        if app_handle:
            subprocess.run(
                ["xdotool", "windowactivate", "--sync", app_handle],
                check=False,
            )

    def type_text(self, text: str):
        subprocess.run(
            ["xdotool", "type", "--delay", "0", "--", text],
            check=True,
        )

    def press_key(self, key_name: str):
        subprocess.run(["xdotool", "key", key_name], check=True)

    def release_modifiers(self):
        subprocess.run(
            ["xdotool", "keyup", "ctrl", "shift", "alt", "super"],
            check=False,
        )

    def clear_line(self):
        subprocess.run(["xdotool", "key", "ctrl+u"], check=False)

    def check_environment(self) -> list[str]:
        import os

        errors = []

        if not shutil.which("xdotool"):
            errors.append("xdotool not found. Install with: sudo pacman -S xdotool")

        if not shutil.which("ffmpeg"):
            print(
                "\033[93mWarning: ffmpeg not found. Using uncompressed WAV (slower uploads).\033[0m"
            )
            print("Install with: sudo pacman -S ffmpeg\n")

        session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if session_type == "wayland":
            print(
                "\033[93mWarning: Running on Wayland. xdotool may not work correctly.\n"
                "Consider using X11 or ydotool for Wayland.\033[0m\n"
            )

        return errors


class MacOS(PlatformInterface):
    """macOS implementation using pyobjc and pynput."""

    def __init__(self):
        import objc  # noqa: F401 — ensures pyobjc-core is available
        from pynput.keyboard import Controller, Key

        self._keyboard = Controller()
        self._Key = Key

    def get_active_app(self):
        import objc
        from AppKit import NSWorkspace

        with objc.autorelease_pool():
            return NSWorkspace.sharedWorkspace().frontmostApplication()

    def restore_focus(self, app_handle):
        if app_handle is None:
            return
        try:
            import objc
            from AppKit import NSApplicationActivateIgnoringOtherApps

            with objc.autorelease_pool():
                app_handle.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
            # Brief sleep to let window manager process the activation
            time.sleep(0.05)
        except Exception:
            pass

    def _post_key_event(self, keycode, flags=0):
        """Post a keyboard event via CGEvents (goes to frontmost app)."""
        from Quartz import (
            CGEventCreateKeyboardEvent,
            CGEventPost,
            CGEventSetFlags,
            kCGHIDEventTap,
        )

        # Key down
        event = CGEventCreateKeyboardEvent(None, keycode, True)
        if flags:
            CGEventSetFlags(event, flags)
        CGEventPost(kCGHIDEventTap, event)
        # Key up
        event = CGEventCreateKeyboardEvent(None, keycode, False)
        if flags:
            CGEventSetFlags(event, flags)
        CGEventPost(kCGHIDEventTap, event)

    def type_text(self, text: str):
        """Inject text via clipboard paste (Cmd+V).

        Uses CGEvents to post the paste keystroke, which correctly targets
        the frontmost application regardless of which process sends it.
        """
        import objc
        from AppKit import NSPasteboard, NSPasteboardTypeString
        from Quartz import kCGEventFlagMaskCommand

        with objc.autorelease_pool():
            pb = NSPasteboard.generalPasteboard()

            # Save current clipboard
            old_types = pb.types()
            old_contents = {}
            if old_types:
                for t in old_types:
                    data = pb.dataForType_(t)
                    if data:
                        old_contents[t] = data

            # Set clipboard to our text
            pb.clearContents()
            pb.setString_forType_(text, NSPasteboardTypeString)

            # Paste with Cmd+V using CGEvents (keycode 9 = 'v')
            time.sleep(0.05)
            self._post_key_event(9, flags=kCGEventFlagMaskCommand)

            # Wait for paste event to propagate before restoring clipboard
            time.sleep(0.25)

            # Restore original clipboard (must declare all types first)
            if old_contents:
                pb.declareTypes_owner_(list(old_contents.keys()), None)
                for ptype, data in old_contents.items():
                    pb.setData_forType_(data, ptype)

    def press_key(self, key_name: str):
        """Press a key using CGEvents (targets frontmost app)."""
        # macOS keycodes
        key_map = {
            "Return": 36,
            "Enter": 36,
            "Tab": 48,
            "Escape": 53,
        }
        keycode = key_map.get(key_name)
        if keycode is not None:
            self._post_key_event(keycode)

    def release_modifiers(self):
        for mod in (
            self._Key.cmd,
            self._Key.cmd_r,
            self._Key.ctrl,
            self._Key.ctrl_r,
            self._Key.alt,
            self._Key.alt_r,
            self._Key.shift,
            self._Key.shift_r,
        ):
            try:
                self._keyboard.release(mod)
            except Exception:
                pass

    def clear_line(self):
        """Clear line using Cmd+Delete via CGEvents.

        This is the macOS GUI standard for deleting to beginning of line.
        Works in most apps. Terminal apps also support Ctrl+U.
        """
        from Quartz import kCGEventFlagMaskCommand

        # keycode 51 = Delete, with Cmd flag
        self._post_key_event(51, flags=kCGEventFlagMaskCommand)

    def check_environment(self) -> list[str]:
        errors = []

        # Check Accessibility permissions
        try:
            from ApplicationServices import AXIsProcessTrusted

            if not AXIsProcessTrusted():
                import os

                terminal = os.environ.get("TERM_PROGRAM", "your terminal app")
                errors.append(
                    f"Accessibility permissions not granted. "
                    f"Please add '{terminal}' in System Settings > "
                    f"Privacy & Security > Accessibility"
                )
        except ImportError:
            errors.append(
                "pyobjc-framework-ApplicationServices not installed. "
                "Install with: pip install pyobjc-framework-Cocoa pyobjc-framework-Quartz"
            )

        if not shutil.which("ffmpeg"):
            print(
                "\033[93mWarning: ffmpeg not found. Using uncompressed WAV (slower uploads).\033[0m"
            )
            print("Install with: brew install ffmpeg\n")

        return errors


def get_platform() -> PlatformInterface:
    """Return the appropriate platform implementation."""
    if sys.platform == "darwin":
        return MacOS()
    else:
        return LinuxX11()
