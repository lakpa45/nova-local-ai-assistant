"""Deterministic user-command parsing and a fail-closed action registry.

Model replies and personal memories are never inputs to this executor.
"""
from dataclasses import dataclass
import threading
import time
from command_parser import phrase
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Define the Action structure
@dataclass(frozen=True)
class Action:
    action: str
    target: str = ""

# Global volume control
def set_system_volume(level):
    """Sets the system master volume (0 to 100)."""
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))

        # pycaw uses a logarithmic scale (-65.25 to 0.0)
        # Convert 0-100 to the correct scale
        import numpy as np
        volume_level = np.log10(max(level / 100, 0.0001)) / 20.0
        volume.SetMasterVolumeLevelScalar(level / 100, None)
        return f"Volume set to {level} percent, sir."
    except Exception as e:
        return f"I encountered an error adjusting the volume: {e}"

def toggle_system_mute(mute=False):
    """Toggles the system mute status."""
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMute(mute, None)
        status = "muted" if mute else "unmuted"
        return f"System {status}, sir."
    except Exception as e:
        return f"Error toggling mute: {e}"

def media_control(command):
    """Controls system media (play, pause, next, prev)."""
    commands = {
        "play": "playpause",
        "pause": "playpause",
        "next": "nexttrack",
        "previous": "prevtrack",
        "stop": "stop"
    }
    key = commands.get(command.lower())
    if key:
        pyautogui.press(key)
        return f"Executing {command} command, sir."
    return "I'm afraid I don't recognize that media command, sir."

# ... rest of the file ...


COMMAND_ACTIONS = {'search': 'search_web', 'close': 'close_app', 'quit': 'close_app'}
CONFIRM_PHRASES = {'yes confirm', 'confirm', 'continue'}
CANCEL_PHRASES = {'no cancel', 'cancel', 'stop'}


@dataclass(frozen=True)
class Action:
    action: str
    target: str = ""


def parse_command(message):
    message = message.strip()
    parts = message.split(maxsplit=1)
    if not parts:
        return None
    verb = parts[0].lower()
    target = parts[1].strip() if len(parts) == 2 else ""
    if verb in {"open", "launch", "start"}:
        kind = "open_app"
        if target.lower() == "camera":
            kind = "open_camera"
        elif target.lower() in {"downloads", "documents", "pictures", "music", "videos", "desktop"}:
            kind = "open_folder"
        elif target.lower() in {"browser", "youtube"}:
            kind = "open_website"
        return Action(kind, target)
    if verb in COMMAND_ACTIONS:
        return Action(COMMAND_ACTIONS[verb], target)
    lower = message.lower()
    for prefix, kind in (("remember that ", "save_memory"), ("forget that ", "forget_memory"),
                         ("remind me in ", "create_reminder")):
        if lower.startswith(prefix):
            return Action(kind, message)
    for phrases, kind in (
        ({"show my memories", "what do you remember about me", "what do you remember about me?"}, "show_memories"),
        ({"show reminders", "my reminders"}, "show_reminders"),
        ({"time", "what time is it", "what time is it?"}, "show_time"),
    ):
        if lower in phrases:
            return Action(kind, message)
    # Explicit action names must go through the same registry, never the model.
    if "_" in verb or verb in {"delete", "remove", "shutdown", "restart", "install", "uninstall", "run", "execute"}:
        return Action(verb, target)
    return None


class ActionExecutor:
    def __init__(self, policy, handlers, clock=time.monotonic, prompts=None):
        self.policy = policy
        self.handlers = dict(handlers)
        self._pending = None
        self.deadline = None
        self.expired = False
        self.clock = clock
        self.prompts = prompts or {}
        self.lock = threading.RLock()

    @property
    def pending(self):
        with self.lock:
            if self._pending is not None and self.clock() >= self.deadline:
                self.clear_pending()
                self.expired = True
            return dict(self._pending) if self._pending is not None else None

    def clear_pending(self):
        self._pending = None
        self.deadline = None

    def confirmation_prompt(self, action):
        if action.action in self.prompts:
            return self.prompts[action.action](action.target)
        if action.action == 'close_app':
            return f'Closing {action.target} may result in the loss of unsaved data. Shall I proceed, sir? Say yes confirm or no cancel.'
        return f'Shall I proceed with {action.action} for {action.target}, sir? Say yes confirm or no cancel.'

    def execute(self, action, confirmed=False):
        with self.lock:
            if self.pending is not None:
                return 'Please confirm or cancel the pending action first. Say yes confirm or no cancel.'
            kind = action.action
            if kind in self.policy.get("blocked", []) or kind not in self.handlers:
                return f"Action rejected: {kind}."
            sensitive = (kind in self.policy.get("require_confirmation", [])
                         or kind in self.policy.get("confirmation_required", [])
                         or kind in {'close_app', 'forget_topic', 'forget_memory',
                                     'save_topic', 'save_command_example', 'retrain_model'})
            if not sensitive and kind not in self.policy.get("auto_approve", []):
                return f"Action rejected: {kind}."
            if sensitive and not confirmed:
                if kind == 'close_app' and not action.target:
                    return 'Please tell me which application to close.'
                self.expired = False
                self._pending = {'action': kind, 'target': action.target}
                self.deadline = self.clock() + 30
                return self.confirmation_prompt(action)

            # OS Control Handlers
            if kind == 'set_volume':
                return set_system_volume(float(action.target))
            if kind == 'toggle_mute':
                return toggle_system_mute(action.target.lower() == 'on')
            if kind == 'media_control':
                return media_control(action.target)

            return self.handlers[kind](action.target)

    def execute_routine(self, actions):
        """Executes a sequence of Actions sequentially."""
        results = []
        for action in actions:
            res = self.execute(action)
            results.append(res)
            # Brief pause between actions for a more natural feel
            time.sleep(0.5)
        return "\n".join(results)
