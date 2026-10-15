import threading
import subprocess
import sqlite3
import re
import json
import os
import queue
import webbrowser
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import quote_plus
from nova_actions import Action, ActionExecutor
from nova_processes import close_application
from knowledge_manager import KnowledgeManager
from command_processor import CommandProcessor
from voice_manager import VoiceManager

import customtkinter as ctk
import config
import optional_llm
import pyttsx3
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from vosk import Model, KaldiRecognizer, SetLogLevel


BASE_DIR = Path(__file__).resolve().parent
VOSK_MODEL_PATH = str(BASE_DIR / "models/vosk-model-small-en-us-0.15")

SAMPLE_RATE = 16000
RECORD_SECONDS = 4
AUDIO_FILE = "nova_voice.wav"

SetLogLevel(-1)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class NovaApp(ctk.CTk):
    def __init__(self, db_path=None, start_listener=True):
        super().__init__()

        self.closing = False
        self.busy = False
        self.ui_queue = queue.Queue()
        self.speech_lock = threading.Lock()
        self.voice_manager = VoiceManager()
        self.after(25, self.drain_ui_queue)
        self.whisper_model = None
        self.vosk_model = None

        self.db_lock = threading.Lock()
        self.wake_stop_event = threading.Event()
        self.wake_thread = None
        self.monitor_thread = None
        self.monitor_running = False
        self.wake_stop_event = threading.Event()
        self.wake_thread = None

        self.setup_database(db_path)

        # Each first command word is connected to a handler.
        self.command_handlers = {
            "open": self.handle_open_command,
            "launch": self.handle_open_command,
            "start": self.handle_open_command,
            "search": self.handle_search_command,
            "close": lambda target: self.handle_command(f"close {target}"),
            "quit": lambda target: self.handle_command(f"quit {target}")
        }
        with (BASE_DIR / "permission.json").open(encoding="utf-8") as policy_file:
            policy = json.load(policy_file)
        with (BASE_DIR / "permissions.json").open(encoding="utf-8") as policy_file:
            additional_policy = json.load(policy_file)
        for category, names in additional_policy.items():
            policy.setdefault(category, []).extend(names)
        self.actions = ActionExecutor(policy, {
            "close_app": close_application,
            "open_app": self.handle_open_command,
            "open_folder": self.open_folder,
            "open_website": self.handle_open_command,
            "open_camera": lambda target: self.launch_camera(),
            "search_web": self.handle_search_command,
            **{kind: self.handle_builtin_command for kind in (
                "save_memory", "forget_memory", "show_memories",
                "create_reminder", "show_reminders", "show_time"
            )},
        })
        self.command_processor = CommandProcessor(self, self.knowledge)

        self.title("NOVA – Personal AI Assistant")
        self.geometry("850x600")
        self.minsize(650, 450)
        self.protocol("WM_DELETE_WINDOW", self.hide_chat)

        self.create_chat_window()
        self.create_floating_widget()
        self.check_reminders()

        # Start system monitoring thread
        self.monitor_running = True
        self.monitor_thread = threading.Thread(target=self.update_system_stats, daemon=True)
        self.monitor_thread.start()

        if start_listener:
            self.start_wake_listener()

        self.withdraw()

    # --------------------------------------------------
    # DATABASE
    # --------------------------------------------------

    def setup_database(self, db_path=None):
        self.database_path = db_path if db_path is not None else BASE_DIR / "nova.db"
        self.db = sqlite3.connect(
            self.database_path,
            check_same_thread=False
        )
        self.cursor = self.db.cursor()

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                reminder_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                reminder_time TEXT NOT NULL,
                completed INTEGER DEFAULT 0
            )
        """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL UNIQUE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.db.commit()
        self.knowledge = KnowledgeManager(self.db, self.db_lock)

    # --------------------------------------------------
    # PERSONAL MEMORY
    # --------------------------------------------------

    def save_memory(self, fact):
        with self.db_lock:
            self.cursor.execute(
                "INSERT OR IGNORE INTO memories (fact) VALUES (?)",
                (fact,)
            )
            self.db.commit()

    def get_memories(self):
        with self.db_lock:
            self.cursor.execute("""
                SELECT fact
                FROM memories
                ORDER BY memory_id DESC
                LIMIT 20
            """)

            return [row[0] for row in self.cursor.fetchall()]

    def forget_memory(self, fact):
        with self.db_lock:
            self.cursor.execute(
                "DELETE FROM memories WHERE LOWER(fact) = LOWER(?)",
                (fact,)
            )

            deleted = self.cursor.rowcount
            self.db.commit()

        return deleted > 0

    def get_memory_context(self):
        memories = self.get_memories()

        if not memories:
            return "No saved personal memories."

        return "\n".join(
            f"- {memory}" for memory in memories
        )

    # --------------------------------------------------
    # REMINDERS
    # --------------------------------------------------

    def save_reminder(self, task, reminder_time):
        with self.db_lock:
            self.cursor.execute(
                """
                INSERT INTO reminders (task, reminder_time)
                VALUES (?, ?)
                """,
                (task, reminder_time.isoformat())
            )
            self.db.commit()

    def get_active_reminders(self):
        with self.db_lock:
            self.cursor.execute("""
                SELECT task, reminder_time
                FROM reminders
                WHERE completed = 0
                ORDER BY reminder_time
            """)

            return self.cursor.fetchall()

    def check_reminders(self):
        now = datetime.now().isoformat()

        with self.db_lock:
            self.cursor.execute("""
                SELECT reminder_id, task
                FROM reminders
                WHERE completed = 0
                AND reminder_time <= ?
            """, (now,))

            reminders = self.cursor.fetchall()

            for reminder_id, _ in reminders:
                self.cursor.execute(
                    """
                    UPDATE reminders
                    SET completed = 1
                    WHERE reminder_id = ?
                    """,
                    (reminder_id,)
                )

            self.db.commit()

        for _, task in reminders:
            message = f"Reminder: {task}"
            self.add_message(f"NOVA: {message}\n\n")

            self.voice_manager.start(
                target=self.speak_and_restart,
                args=(message,),
            )

        self.post_ui(15000, self.check_reminders)

    # --------------------------------------------------
    # CHAT WINDOW
    # --------------------------------------------------

    def create_chat_window(self):
        ctk.CTkLabel(
            self,
            text="NOVA",
            font=ctk.CTkFont(
                size=32,
                weight="bold"
            )
        ).pack(pady=(20, 0))

        self.status_label = ctk.CTkLabel(
            self,
            text="Background listener • Ready",
            text_color="#68d391"
        )
        self.status_label.pack(pady=(0, 15))

        self.chat_box = ctk.CTkTextbox(
            self,
            wrap="word",
            font=("Arial", 15)
        )
        self.chat_box.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=10
        )

        self.chat_box.insert(
            "end",
            "JARVIS: Online and ready, sir. How may I assist you today?\n\n"
        )
        self.chat_box.configure(state="disabled")

        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.pack(
            fill="x",
            padx=25,
            pady=(5, 25)
        )

        self.message_entry = ctk.CTkEntry(
            bottom_frame,
            placeholder_text="Type a message for NOVA...",
            font=("Arial", 15)
        )
        self.message_entry.pack(
            side="left",
            fill="x",
            expand=True,
            padx=10,
            pady=10
        )
        self.message_entry.bind(
            "<Return>",
            self.send_message
        )

        self.listen_button = ctk.CTkButton(
            bottom_frame,
            text="🎙 Listen",
            width=105,
            command=self.start_listening,
            fg_color="#805ad5",
            hover_color="#6b46c1"
        )
        self.listen_button.pack(
            side="right",
            padx=(0, 8),
            pady=10
        )

        self.send_button = ctk.CTkButton(
            bottom_frame,
            text="Send",
            width=90,
            command=self.send_message
        )
        self.send_button.pack(
            side="right",
            padx=(0, 10),
            pady=10
        )

    # --------------------------------------------------
    # FLOATING WIDGET
    # --------------------------------------------------

    def create_floating_widget(self):
        self.widget = ctk.CTkToplevel(self)
        self.widget.geometry("270x265+1000+650")
        self.widget.resizable(False, False)
        self.widget.attributes("-topmost", True)
        self.widget.overrideredirect(True)

        card = ctk.CTkFrame(
            self.widget,
            corner_radius=20
        )
        card.pack(fill="both", expand=True)

        title = ctk.CTkLabel(
            card,
            text="✦  JARVIS",
            font=ctk.CTkFont(
                size=22,
                weight="bold"
            )
        )
        title.pack(pady=(16, 2))

        # System Health Dashboard (HUD)
        self.hud_frame = ctk.CTkFrame(card, fg_color="transparent")
        self.hud_frame.pack(pady=5)

        self.cpu_label = ctk.CTkLabel(
            self.hud_frame,
            text="CPU: 0%",
            font=("Consolas", 12),
            text_color="#63b3ed"
        )
        self.cpu_label.pack(side="left", padx=10)

        self.ram_label = ctk.CTkLabel(
            self.hud_frame,
            text="RAM: 0%",
            font=("Consolas", 12),
            text_color="#63b3ed"
        )
        self.ram_label.pack(side="left", padx=10)

        # Voice Visualizer Canvas
        self.viz_canvas = ctk.CTkCanvas(
            card,
            width=200,
            height=40,
            bg="#2d3748",
            highlightthickness=0,
            corner_radius=10
        )
        self.viz_canvas.pack(pady=10)

        self.widget_status = ctk.CTkLabel(
            card,
            text="Awaiting orders",
            text_color="#68d391"
        )
        self.widget_status.pack(pady=(0, 12))

        button_frame = ctk.CTkFrame(
            card,
            fg_color="transparent"
        )
        button_frame.pack(pady=2)

        self.widget_listen_button = ctk.CTkButton(
            button_frame,
            text="🎙 Listen",
            width=100,
            command=self.start_listening,
            fg_color="#805ad5",
            hover_color="#6b46c1"
        )
        self.widget_listen_button.pack(
            side="left",
            padx=4
        )

        ctk.CTkButton(
            button_frame,
            text="Camera",
            width=90,
            command=self.open_camera,
            fg_color="#2f855a",
            hover_color="#276749"
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            card,
            text="Open Chat",
            width=205,
            command=self.show_chat
        ).pack(pady=(8, 8))

        ctk.CTkButton(
            card,
            text="×",
            width=28,
            height=28,
            fg_color="transparent",
            hover_color="#9b2c2c",
            command=self.close_app
        ).place(
            relx=0.91,
            rely=0.08,
            anchor="center"
        )

        for item in (
            card,
            title,
            self.widget_status
        ):
            item.bind(
                "<Button-1>",
                self.start_drag
            )
            item.bind(
                "<B1-Motion>",
                self.drag_widget
            )

    def start_drag(self, event):
        self.drag_x = (
            event.x_root - self.widget.winfo_x()
        )
        self.drag_y = (
            event.y_root - self.widget.winfo_y()
        )

    def drag_widget(self, event):
        x = event.x_root - self.drag_x
        y = event.y_root - self.drag_y

        self.widget.geometry(f"+{x}+{y}")

    def show_chat(self):
        self.deiconify()
        self.lift()
        self.message_entry.focus()

    def hide_chat(self):
        self.withdraw()

    def close_app(self):
        self.closing = True
        self.wake_stop_event.set()
        self.voice_manager.stop()
        sd.stop()
        self.withdraw()
        self.widget.withdraw()
        self.finish_shutdown()

    def update_system_stats(self):
        import psutil
        import random
        import math
        while self.monitor_running:
            try:
                cpu = psutil.cpu_percent()
                ram = psutil.virtual_memory().percent

                self.post_ui(0, lambda c=cpu, r=ram: self.update_hud(c, r))
                self.post_ui(0, self.animate_visualizer)

                # Proactive Suggestions
                if cpu > 90:
                    self.post_ui(0, lambda: self.deliver_response("Pardon the interruption, sir, but I believe your system is under significant stress. CPU usage is at 90%."))
                elif ram > 90:
                    self.post_ui(0, lambda: self.deliver_response("Sir, your system memory is nearly exhausted. I suggest closing some resource-heavy applications."))
            except Exception as e:
                print(f"Monitor error: {e}")

            time.sleep(0.1)

    def update_hud(self, cpu, ram):
        self.cpu_label.configure(text=f"CPU: {cpu}%")
        self.ram_label.configure(text=f"RAM: {ram}%")

    def animate_visualizer(self):
        self.viz_canvas.delete("all")
        if not self.busy:
            # Idle state: subtle slow wave
            amplitude = 2
            frequency = 0.1
        else:
            # Active state: energetic pulse
            amplitude = random.randint(5, 15)
            frequency = 0.3

        # Draw a simple sine-wave visualizer
        points = []
        for x in range(0, 201, 5):
            y = 20 + amplitude * math.sin(x * frequency + time.time() * 10)
            points.append((x, y))

        for i in range(len(points) - 1):
            self.viz_canvas.create_line(
                points[i][0], points[i][1],
                points[i+1][0], points[i+1][1],
                fill="#63b3ed", width=2, smooth=True
            )

    # --------------------------------------------------
    # INTERFACE HELPERS
    # --------------------------------------------------

    def post_ui(self, delay, callback):
        """Workers only enqueue work; only the main thread touches Tk."""
        if not self.closing:
            self.ui_queue.put((delay, callback))

    def drain_ui_queue(self):
        if self.closing:
            return
        for _ in range(100):
            try:
                delay, callback = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if delay:
                self.after(delay, callback)
            else:
                callback()
            if self.closing:
                return
        self.after(25, self.drain_ui_queue)

    def add_message(self, message):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", message)
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def set_status(
        self,
        text,
        color="#68d391"
    ):
        self.status_label.configure(
            text=text,
            text_color=color
        )
        self.widget_status.configure(
            text=text,
            text_color=color
        )

    def set_buttons_state(self, state):
        self.listen_button.configure(state=state)
        self.widget_listen_button.configure(
            state=state
        )
        self.send_button.configure(state=state)

    # --------------------------------------------------
    # APP AND FOLDER OPENING
    # --------------------------------------------------

    def open_camera(self):
        return self.actions.execute(Action("open_camera", "camera"))

    def launch_camera(self):
        os.startfile("microsoft.windows.camera:")
        return "Of course. Accessing the camera systems now, sir."

    def open_folder(self, folder_name):
        folders = {
            "downloads": Path.home() / "Downloads",
            "documents": Path.home() / "Documents",
            "pictures": Path.home() / "Pictures",
            "music": Path.home() / "Music",
            "videos": Path.home() / "Videos",
            "desktop": Path.home() / "Desktop"
        }

        folder_path = folders.get(
            folder_name.lower()
        )

        if folder_path and folder_path.exists():
            os.startfile(str(folder_path))
            return f"Right away, sir. Opening the {folder_name} folder."

        return None

    def get_installed_apps(self):
        try:
            powershell_command = (
                "Get-StartApps | "
                "Select-Object Name, AppID | "
                "ConvertTo-Json -Compress"
            )

            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    powershell_command
                ],
                capture_output=True,
                text=True,
                timeout=15
            )

            if not result.stdout.strip():
                return []

            apps = json.loads(result.stdout)

            if isinstance(apps, dict):
                return [apps]

            return apps

        except Exception as error:
            print(f"Installed apps error: {error}")
            return []

    def open_installed_app(self, app_name):
        apps = self.get_installed_apps()

        search_name = app_name.lower().strip()
        selected_app = None

        # Exact match first.
        for app in apps:
            name = app.get("Name", "").lower()

            if name == search_name:
                selected_app = app
                break

        # Partial match second.
        if not selected_app:
            for app in apps:
                name = app.get("Name", "").lower()

                if search_name in name:
                    selected_app = app
                    break

        if not selected_app:
            return None

        app_id = selected_app.get("AppID")
        app_display_name = selected_app.get(
            "Name",
            app_name
        )

        try:
            if app_id.lower().endswith(
                (".lnk", ".exe")
            ):
                os.startfile(app_id)
            else:
                subprocess.Popen([
                    "explorer.exe",
                    f"shell:AppsFolder\\{app_id}"
                ])

            return f"Certainly. Launching {app_display_name} for you, sir."

        except Exception as error:
            print(f"App opening error: {error}")
            return None

    # --------------------------------------------------
    # COMMAND ROUTER
    # --------------------------------------------------

    def handle_open_command(self, target):
        if not target:
            return "Please tell me what you want to open."

        target_name = target.lower().strip()

        special_actions = {
            "camera": (
                self.launch_camera,
                "Opening Camera."
            ),
            "calculator": (
                lambda: subprocess.Popen(
                    ["calc.exe"]
                ),
                "Opening Calculator."
            ),
            "browser": (
                lambda: webbrowser.open(
                    "https://www.google.com"
                ),
                "Opening your browser."
            ),
            "youtube": (
                lambda: webbrowser.open(
                    "https://www.youtube.com"
                ),
                "Opening YouTube."
            )
        }

        selected_action = special_actions.get(
            target_name
        )

        if selected_action:
            action_function, response = selected_action
            action_function()
            return f"As you wish, sir. {response}"

        folder_answer = self.open_folder(
            target_name
        )

        if folder_answer:
            return folder_answer

        app_answer = self.open_installed_app(
            target
        )

        if app_answer:
            return app_answer

        return f"I'm afraid I couldn't locate {target} in the system, sir."

    def handle_search_command(self, target):
        if not target:
            return "Please tell me what you want to search for."

        target_lower = target.lower()

        if target_lower.startswith("youtube for "):
            search_text = target[len("youtube for "):].strip()

            webbrowser.open(
                "https://www.youtube.com/results"
                f"?search_query={quote_plus(search_text)}"
            )

            return (
                f"Searching YouTube for {search_text}."
            )

        if target_lower.startswith("google for "):
            search_text = target[len("google for "):].strip()
        else:
            search_text = target

        webbrowser.open(
            "https://www.google.com/search"
            f"?q={quote_plus(search_text)}"
        )

        return f"Searching Google for {search_text} now, sir."

    def handle_command(self, message):
        return self.command_processor.handle(message)

    def handle_builtin_command(self, message):
        message = message.strip()
        command = message.lower().strip()

        # Save personal memory.
        if command.startswith("remember that "):
            fact = message[len("remember that "):].strip()
            self.save_memory(fact)

            return (
                f"Okay. I will remember that {fact}."
            )

        # Delete personal memory.
        if command.startswith("forget that "):
            fact = message[len("forget that "):].strip()

            if self.forget_memory(fact):
                return f"Okay. I forgot that {fact}."

            return "I could not find that memory."

        # Display personal memories.
        if command in (
            "what do you remember about me",
            "what do you remember about me?",
            "show my memories"
        ):
            memories = self.get_memories()

            if not memories:
                return (
                    "I do not have any saved memories yet."
                )

            return (
                "I remember: "
                + ". ".join(memories)
            )

        # Save reminder.
        reminder_match = re.match(
            (
                r"^remind me in (\d+) "
                r"(minute|minutes|hour|hours) "
                r"to (.+)"
            ),
            message.strip(),
            re.IGNORECASE
        )

        if reminder_match:
            number = int(reminder_match.group(1))
            unit = reminder_match.group(2).lower()
            task = reminder_match.group(3).strip()

            if "hour" in unit:
                reminder_time = (
                    datetime.now()
                    + timedelta(hours=number)
                )
            else:
                reminder_time = (
                    datetime.now()
                    + timedelta(minutes=number)
                )

            self.save_reminder(
                task,
                reminder_time
            )

            return (
                f"Okay. I will remind you in "
                f"{number} {unit} to {task}."
            )

        # Display reminders.
        if command in (
            "show reminders",
            "my reminders"
        ):
            reminders = self.get_active_reminders()

            if not reminders:
                return "You have no active reminders."

            items = []

            for task, reminder_time in reminders:
                formatted_time = (
                    datetime.fromisoformat(
                        reminder_time
                    ).strftime("%I:%M %p")
                )

                items.append(
                    f"{task} at {formatted_time}"
                )

            return (
                "Your reminders are: "
                + ". ".join(items)
            )

        # Time command.
        if command in (
            "time",
            "what time is it",
            "what time is it?"
        ):
            current_time = datetime.now().strftime(
                "%I:%M %p"
            )

            return (
                f"The current time is {current_time}."
            )

        # General command router.
        parts = message.strip().split(maxsplit=1)

        action = (
            parts[0].lower()
            if parts
            else ""
        )

        target = (
            parts[1].strip()
            if len(parts) > 1
            else ""
        )

        handler = self.command_handlers.get(action)

        if handler:
            return handler(target)

        return None

    # --------------------------------------------------
    # WAKE-WORD LISTENER
    # --------------------------------------------------

    def start_wake_listener(self):
        if self.closing or self.busy:
            return
        if (
            self.wake_thread
            and self.wake_thread.is_alive()
        ):
            if self.wake_stop_event.is_set():
                self.post_ui(50, self.start_wake_listener)
            return

        self.wake_stop_event.clear()

        self.wake_thread = threading.Thread(
            target=self.wake_word_listener,
            daemon=True
        )
        self.wake_thread.start()

    def stop_wake_listener(self):
        self.wake_stop_event.set()

    def wake_word_listener(self):
        try:
            if self.vosk_model is None:
                self.vosk_model = Model(
                    VOSK_MODEL_PATH
                )

            recognizer = KaldiRecognizer(
                self.vosk_model,
                SAMPLE_RATE
            )

            audio_queue = queue.Queue()

            def audio_callback(
                indata,
                frames,
                time,
                status
            ):
                if not self.wake_stop_event.is_set():
                    audio_queue.put(bytes(indata))

            with self.voice_manager.microphone, sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=audio_callback
            ):
                while not self.wake_stop_event.is_set():
                    try:
                        audio_data = audio_queue.get(
                            timeout=0.2
                        )
                    except queue.Empty:
                        continue

                    if recognizer.AcceptWaveform(
                        audio_data
                    ):
                        result = json.loads(
                            recognizer.Result()
                        )

                        heard_text = result.get(
                            "text",
                            ""
                        ).lower()

                        if "nova" in heard_text.split():
                            self.wake_stop_event.set()

                            self.post_ui(
                                0,
                                self.on_wake_word_detected
                            )
                            return

        except Exception as error:
            print(f"Wake-word error: {error}")
            self.post_ui(0, lambda error=error: self.set_status(
                f"Wake listener unavailable: {error}", "#fc8181"
            ))

    def on_wake_word_detected(self):
        if self.closing or self.busy:
            return
        self.busy = True
        self.set_status(
            "JARVIS detected. At your service...",
            "#f6e05e"
        )
        self.set_buttons_state("disabled")

        # Spoken acknowledgment for that Jarvis feel
        self.voice_manager.start(
            target=self.speak_and_restart,
            args=("Yes, sir?", True),
        )

        self.post_ui(
            400,
            self.start_recording_thread
        )

    # --------------------------------------------------
    # VOICE INPUT
    # --------------------------------------------------

    def start_listening(self):
        if self.closing or self.busy:
            return
        self.busy = True
        self.stop_wake_listener()

        self.set_status(
            "Speak now...",
            "#f6e05e"
        )
        self.set_buttons_state("disabled")

        self.post_ui(
            400,
            self.start_recording_thread
        )

    def start_recording_thread(self):
        self.voice_manager.start(
            target=self.record_and_transcribe,
        )

    def record_and_transcribe(self):
        try:
            if self.wake_thread:
                self.wake_thread.join()
            if self.closing:
                return
            if self.whisper_model is None:
                self.post_ui(
                    0,
                    lambda: self.set_status(
                        "Loading voice model...",
                        "#f6e05e"
                    )
                )

                self.whisper_model = WhisperModel(
                    "base.en",
                    device="cpu",
                    compute_type="int8"
                )

            with self.voice_manager.microphone:
                if self.closing:
                    return
                audio = sd.rec(
                    int(RECORD_SECONDS * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="int16"
                )
                sd.wait()
            if self.closing:
                return

            write(
                AUDIO_FILE,
                SAMPLE_RATE,
                audio
            )

            self.post_ui(
                0,
                lambda: self.set_status(
                    "Understanding...",
                    "#f6e05e"
                )
            )

            segments, _ = (
                self.whisper_model.transcribe(
                    AUDIO_FILE,
                    language="en"
                )
            )

            spoken_text = "".join(
                segment.text
                for segment in segments
            ).strip()

            if spoken_text:
                self.post_ui(
                    0,
                    lambda: self.handle_voice_text(
                        spoken_text
                    )
                )
            else:
                self.post_ui(
                    0,
                    lambda: self.voice_error(
                        "I could not hear anything."
                    )
                )

        except Exception as error:
            self.post_ui(
                0,
                lambda error=error: self.voice_error(
                    f"Voice error: {error}"
                )
            )

    def handle_voice_text(self, text):
        self.show_chat()
        self.add_message(f"You 🎙: {text}\n")

        self.set_status(
            "NOVA is thinking...",
            "#f6e05e"
        )

        self.voice_manager.start(
            target=self.get_nova_response,
            args=(text,),
        )

    def voice_error(self, message):
        self.busy = False
        self.add_message(
            f"NOVA: {message}\n\n"
        )
        self.set_status("Say JARVIS")
        self.set_buttons_state("normal")
        self.start_wake_listener()

    # --------------------------------------------------
    # VOICE OUTPUT
    # --------------------------------------------------

    def speak(self, text):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()

        except Exception as error:
            print(f"Speech error: {error}")

    def speak_and_restart(self, text, release_busy=False):
        try:
            with self.speech_lock:
                self.stop_wake_listener()
                if self.wake_thread:
                    self.wake_thread.join()
                if not self.closing:
                    with self.voice_manager.microphone:
                        self.speak(text)
        finally:
            self.post_ui(0, lambda: self.finish_response(release_busy))

    def finish_response(self, release_busy=True):
        if release_busy:
            self.busy = False
            self.set_buttons_state("normal")
        if not self.busy:
            self.set_status("Say JARVIS")
            self.start_wake_listener()

    # --------------------------------------------------
    # TEXT AND AI RESPONSES
    # --------------------------------------------------

    def send_message(self, event=None):
        if self.closing or self.busy:
            return
        user_message = (
            self.message_entry.get().strip()
        )

        if not user_message:
            return

        self.busy = True
        self.stop_wake_listener()
        self.add_message(
            f"You: {user_message}\n"
        )
        self.message_entry.delete(0, "end")

        self.set_status(
            "NOVA is thinking...",
            "#f6e05e"
        )
        self.set_buttons_state("disabled")

        self.voice_manager.start(
            target=self.get_nova_response,
            args=(user_message,),
        )

    def deliver_response(self, answer):
        self.post_ui(
            0,
            lambda: self.add_message(
                f"NOVA: {answer}\n\n"
            )
        )

        self.post_ui(
            0,
            lambda: self.set_status(
                "NOVA is speaking...",
                "#63b3ed"
            )
        )

        self.voice_manager.start(
            target=self.speak_and_restart,
            args=(self.spoken_summary(answer), True),
        )

    @staticmethod
    def spoken_summary(answer):
        words = answer.split()
        if len(words) <= 45:
            return answer
        suffix = ' Say yes confirm or no cancel.' if 'Say yes confirm or no cancel.' in answer else ''
        return ' '.join(words[:35]) + '. The full text is in the chat.' + suffix

    def get_nova_response(self, user_message):
        response_sent = False

        try:
            command_answer = self.handle_command(
                user_message
            )

            if command_answer:
                self.deliver_response(
                    command_answer
                )
                response_sent = True
                return

            answer = optional_llm.answer(user_message, self.get_memory_context())
            self.deliver_response(answer)
            response_sent = True

        except Exception as error:
            self.post_ui(
                0,
                lambda error=error: self.add_message(
                    f"NOVA: Error: {error}\n\n"
                )
            )

        finally:
            if not response_sent:
                self.post_ui(0, self.finish_response)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NOVA personal command assistant")
    parser.add_argument('--smoke-test', action='store_true',
                        help='Start with an in-memory database and close after two seconds.')
    options = parser.parse_args()
    app = NovaApp(db_path=':memory:' if options.smoke_test else None)
    if options.smoke_test:
        app.widget.withdraw()
        app.after(2000, app.close_app)
    app.mainloop()
    if options.smoke_test:
        print('NOVA_STARTUP_OK')
