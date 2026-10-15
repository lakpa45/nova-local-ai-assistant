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

import customtkinter as ctk
import ollama
import pyttsx3
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from vosk import Model, KaldiRecognizer, SetLogLevel


MODEL_NAME = "qwen3:4b"
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000
RECORD_SECONDS = 6
AUDIO_FILE = "nova_voice.wav"

SetLogLevel(-1)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class NovaApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.whisper_model = None
        self.db_lock = threading.Lock()
        self.wake_stop_event = threading.Event()
        self.wake_thread = None

        self.setup_database()

        self.title("NOVA – Personal AI Assistant")
        self.geometry("850x600")
        self.minsize(650, 450)
        self.protocol("WM_DELETE_WINDOW", self.hide_chat)

        self.create_chat_window()
        self.create_floating_widget()
        self.check_reminders()
        self.start_wake_listener()

        self.withdraw()

    def setup_database(self):
        self.db = sqlite3.connect("nova.db", check_same_thread=False)
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
                    created_at TEXT DEFAULT CURRENT_TIMESTAMPT
                    )
                """)

        self.db.commit()

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

        return "\n".join(f"- {memory}" for memory in memories)
    
    def save_reminder(self, task, reminder_time):
        with self.db_lock:
            self.cursor.execute(
                "INSERT INTO reminders (task, reminder_time) VALUES (?, ?)",
                (task, reminder_time.isoformat())
            )
            self.db.commit()

    def check_reminders(self):
        now = datetime.now().isoformat()

        with self.db_lock:
            self.cursor.execute("""
                SELECT reminder_id, task
                FROM reminders
                WHERE completed = 0 AND reminder_time <= ?
            """, (now,))

            reminders = self.cursor.fetchall()

            for reminder_id, _ in reminders:
                self.cursor.execute(
                    "UPDATE reminders SET completed = 1 WHERE reminder_id = ?",
                    (reminder_id,)
                )

            self.db.commit()

        for _, task in reminders:
            message = f"Reminder: {task}"
            self.add_message(f"NOVA: {message}\n\n")
            threading.Thread(
                target=self.speak_and_restart,
                args=(message,),
                daemon=True
            ).start()

        self.after(15000, self.check_reminders)

    def create_chat_window(self):
        ctk.CTkLabel(
            self,
            text="NOVA",
            font=ctk.CTkFont(size=32, weight="bold")
        ).pack(pady=(20, 0))

        self.status_label = ctk.CTkLabel(
            self,
            text="Background listener • Ready",
            text_color="#68d391"
        )
        self.status_label.pack(pady=(0, 15))

        self.chat_box = ctk.CTkTextbox(self, wrap="word", font=("Arial", 15))
        self.chat_box.pack(fill="both", expand=True, padx=25, pady=10)
        self.chat_box.insert(
            "end",
            "NOVA: I am running in the background. Say NOVA to wake me.\n\n"
        )
        self.chat_box.configure(state="disabled")

        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.pack(fill="x", padx=25, pady=(5, 25))

        self.message_entry = ctk.CTkEntry(
            bottom_frame,
            placeholder_text="Type a message for NOVA...",
            font=("Arial", 15)
        )
        self.message_entry.pack(side="left", fill="x", expand=True, padx=10, pady=10)
        self.message_entry.bind("<Return>", self.send_message)

        self.listen_button = ctk.CTkButton(
            bottom_frame,
            text="🎙 Listen",
            width=105,
            command=self.start_listening,
            fg_color="#805ad5",
            hover_color="#6b46c1"
        )
        self.listen_button.pack(side="right", padx=(0, 8), pady=10)

        self.send_button = ctk.CTkButton(
            bottom_frame,
            text="Send",
            width=90,
            command=self.send_message
        )
        self.send_button.pack(side="right", padx=(0, 10), pady=10)

    def create_floating_widget(self):
        self.widget = ctk.CTkToplevel(self)
        self.widget.geometry("270x185+1000+650")
        self.widget.resizable(False, False)
        self.widget.attributes("-topmost", True)
        self.widget.overrideredirect(True)

        card = ctk.CTkFrame(self.widget, corner_radius=20)
        card.pack(fill="both", expand=True)

        title = ctk.CTkLabel(
            card,
            text="✦  NOVA",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        title.pack(pady=(16, 2))

        self.widget_status = ctk.CTkLabel(
            card,
            text="Say NOVA",
            text_color="#68d391"
        )
        self.widget_status.pack(pady=(0, 12))

        buttons = ctk.CTkFrame(card, fg_color="transparent")
        buttons.pack(pady=2)

        self.widget_listen_button = ctk.CTkButton(
            buttons,
            text="🎙 Listen",
            width=100,
            command=self.start_listening,
            fg_color="#805ad5",
            hover_color="#6b46c1"
        )
        self.widget_listen_button.pack(side="left", padx=4)

        ctk.CTkButton(
            buttons,
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
        ).place(relx=0.91, rely=0.08, anchor="center")

        for item in (card, title, self.widget_status):
            item.bind("<Button-1>", self.start_drag)
            item.bind("<B1-Motion>", self.drag_widget)

    def start_drag(self, event):
        self.drag_x = event.x_root - self.widget.winfo_x()
        self.drag_y = event.y_root - self.widget.winfo_y()

    def drag_widget(self, event):
        self.widget.geometry(
            f"+{event.x_root - self.drag_x}+{event.y_root - self.drag_y}"
        )

    def show_chat(self):
        self.deiconify()
        self.lift()
        self.message_entry.focus()

    def hide_chat(self):
        self.withdraw()

    def close_app(self):
        self.wake_stop_event.set()
        self.db.close()
        self.destroy()

    def set_status(self, text, color="#68d391"):
        self.status_label.configure(text=text, text_color=color)
        self.widget_status.configure(text=text, text_color=color)

    def set_buttons_state(self, state):
        self.listen_button.configure(state=state)
        self.widget_listen_button.configure(state=state)
        self.send_button.configure(state=state)

    def add_message(self, message):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", message)
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def open_camera(self):
        subprocess.Popen("start microsoft.windows.camera:", shell=True)

    def open_folder(self, folder_name):
        folders = {
            "downloads": Path.home() / "Downloads",
            "documents": Path.home() / "Documents",
            "pictures": Path.home() / "Pictures",
            "music": Path.home() / "Music",
            "videos": Path.home() / "Videos",
            "desktop": Path.home() / "Desktop"
        }

        folder_path = folders.get(folder_name.lower())

        if folder_path and folder_path.exists():
            os.startfile(str(folder_path))
            return f"Opening {folder_name}."

        return None

    def open_installed_app(self, app_name):
        try:
            command = (
                "Get-StartApps | Select-Object Name, AppID | "
                "ConvertTo-Json -Compress"
            )

            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                timeout=15
            )

            apps = json.loads(result.stdout)

            if isinstance(apps, dict):
                apps = [apps]

            search_name = app_name.lower().strip()
            selected_app = None

            for app in apps:
                if app["Name"].lower() == search_name:
                    selected_app = app
                    break

            if not selected_app:
                for app in apps:
                    if search_name in app["Name"].lower():
                        selected_app = app
                        break

            if not selected_app:
                return None

            app_id = selected_app["AppID"]
            app_display_name = selected_app["Name"]

            if app_id.lower().endswith((".lnk", ".exe")):
                os.startfile(app_id)
            else:
                subprocess.Popen([
                    "explorer.exe",
                    f"shell:AppsFolder\\{app_id}"
                ])

            return f"Opening {app_display_name}."

        except Exception as error:
            print(f"App launcher error: {error}")
            return None

    def start_wake_listener(self):
        if self.wake_thread and self.wake_thread.is_alive():
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
            model = Model(VOSK_MODEL_PATH)
            recognizer = KaldiRecognizer(model, SAMPLE_RATE)
            audio_queue = queue.Queue()

            def audio_callback(indata, frames, time, status):
                if not self.wake_stop_event.is_set():
                    audio_queue.put(bytes(indata))

            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=audio_callback
            ):
                while not self.wake_stop_event.is_set():
                    try:
                        audio_data = audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if recognizer.AcceptWaveform(audio_data):
                        result = json.loads(recognizer.Result())
                        heard_text = result.get("text", "").lower()

                        if "nova" in heard_text:
                            self.wake_stop_event.set()
                            self.after(0, self.on_wake_word_detected)
                            return

        except Exception as error:
            print(f"Wake-word error: {error}")

    def on_wake_word_detected(self):
        self.set_status("NOVA detected. Speak now...", "#f6e05e")
        self.set_buttons_state("disabled")
        self.after(400, self.start_recording_thread)

    def start_listening(self):
        self.stop_wake_listener()
        self.set_status("Speak now...", "#f6e05e")
        self.set_buttons_state("disabled")
        self.after(400, self.start_recording_thread)

    def start_recording_thread(self):
        threading.Thread(
            target=self.record_and_transcribe,
            daemon=True
        ).start()

    def record_and_transcribe(self):
        try:
            if self.whisper_model is None:
                self.after(
                    0,
                    lambda: self.set_status("Loading voice model...", "#f6e05e")
                )

                self.whisper_model = WhisperModel(
                    "base.en",
                    device="cpu",
                    compute_type="int8"
                )

            audio = sd.rec(
                int(RECORD_SECONDS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16"
            )
            sd.wait()

            write(AUDIO_FILE, SAMPLE_RATE, audio)

            self.after(
                0,
                lambda: self.set_status("Understanding...", "#f6e05e")
            )

            segments, _ = self.whisper_model.transcribe(
                AUDIO_FILE,
                language="en"
            )

            spoken_text = "".join(
                segment.text for segment in segments
            ).strip()

            if spoken_text:
                self.after(0, lambda: self.handle_voice_text(spoken_text))
            else:
                self.after(
                    0,
                    lambda: self.voice_error(
                        "I could not hear anything. Please try again."
                    )
                )

        except Exception as error:
            self.after(
                0,
                lambda: self.voice_error(f"Voice error: {error}")
            )

    def handle_voice_text(self, text):
        self.show_chat()
        self.add_message(f"You 🎙: {text}\n")
        self.set_status("NOVA is thinking...", "#f6e05e")

        threading.Thread(
            target=self.get_nova_response,
            args=(text,),
            daemon=True
        ).start()

    def voice_error(self, message):
        self.add_message(f"NOVA: {message}\n\n")
        self.set_status("Say NOVA")
        self.set_buttons_state("normal")
        self.start_wake_listener()

    def speak(self, text):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
        except Exception as error:
            print(f"Speech error: {error}")

    def speak_and_restart(self, text):
        self.stop_wake_listener()
        self.speak(text)
        self.after(0, self.start_wake_listener)

    def handle_command(self, message):
        command = message.lower().strip()

        if command.startswith("remember that "):
            fact = message[len("remember that "):].strip()
            self.save_memory(fact)
            return f"Okay. I will remember that {fact}."

        if command in ("what do you remember about me?", "show my memories"):
            memories = self.get_memories()

            if not memories:
                return "I do not have any saved memories yet."

            return "I remember: " + ". ".join(memories)

        if command.startswith("forget that "):
            fact = message[len("forget that "):].strip()

            if self.forget_memory(fact):
                return f"Okay. I forgot that {fact}."

            return "I could not find that memory."

        reminder_match = re.match(
            r"^remind me in (\d+) (minute|minutes|hour|hours) to (.+)",
            message.strip(),
            re.IGNORECASE
        )

        if reminder_match:
            number = int(reminder_match.group(1))
            unit = reminder_match.group(2).lower()
            task = reminder_match.group(3).strip()

            if "hour" in unit:
                reminder_time = datetime.now() + timedelta(hours=number)
            else:
                reminder_time = datetime.now() + timedelta(minutes=number)

            self.save_reminder(task, reminder_time)

            return f"Okay. I will remind you in {number} {unit} to {task}."

        if command in ("show reminders", "my reminders"):
            with self.db_lock:
                self.cursor.execute("""
                    SELECT task, reminder_time
                    FROM reminders
                    WHERE completed = 0
                    ORDER BY reminder_time
                """)
                reminders = self.cursor.fetchall()

            if not reminders:
                return "You have no active reminders."

            items = []

            for task, reminder_time in reminders:
                formatted_time = datetime.fromisoformat(
                    reminder_time
                ).strftime("%I:%M %p")

                items.append(f"{task} at {formatted_time}")

            return "Your reminders are: " + ". ".join(items)

        if "open camera" in command:
            self.open_camera()
            return "Opening the Camera."

        if "open calculator" in command:
            subprocess.Popen("start calc", shell=True)
            return "Opening Calculator."

        if "open browser" in command:
            webbrowser.open("https://www.google.com")
            return "Opening your browser."

        if "open youtube" in command:
            webbrowser.open("https://www.youtube.com")
            return "Opening YouTube."

        if command.startswith("search google for "):
            search_text = message[len("search google for "):].strip()
            webbrowser.open(
                f"https://www.google.com/search?q={quote_plus(search_text)}"
            )
            return f"Searching Google for {search_text}."

        if command.startswith("search youtube for "):
            search_text = message[len("search youtube for "):].strip()
            webbrowser.open(
                f"https://www.youtube.com/results?search_query={quote_plus(search_text)}"
            )
            return f"Searching YouTube for {search_text}."

        if "what time is it" in command or command == "time":
            current_time = datetime.now().strftime("%I:%M %p")
            return f"The current time is {current_time}."

        if command.startswith("open "):
            target = message[5:].strip()

            folder_answer = self.open_folder(target)
            if folder_answer:
                return folder_answer

            app_answer = self.open_installed_app(target)
            if app_answer:
                return app_answer

            return f"I could not find {target} in your installed apps."

        return None

    def deliver_response(self, answer):
        self.after(0, lambda: self.add_message(f"NOVA: {answer}\n\n"))

        threading.Thread(
            target=self.speak_and_restart,
            args=(answer,),
            daemon=True
        ).start()

    def send_message(self, event=None):
        user_message = self.message_entry.get().strip()

        if not user_message:
            return

        self.add_message(f"You: {user_message}\n")
        self.message_entry.delete(0, "end")
        self.set_status("NOVA is thinking...", "#f6e05e")
        self.set_buttons_state("disabled")

        threading.Thread(
            target=self.get_nova_response,
            args=(user_message,),
            daemon=True
        ).start()

    def get_nova_response(self, user_message):
        response_sent = False

        try:
            command_answer = self.handle_command(user_message)

            if command_answer:
                self.deliver_response(command_answer)
                response_sent = True
                return

            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are NOVA, a personal AI assistant. "
                            "Be friendly, clear, and helpful. "
                            "Keep normal answers under 45 words. "
                            "Use simple English and short explanations. "
                            "Do not give long lists unless the user asks."
                        )
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                think=False,
                keep_alive="30m",
                options={
                    "temperature": 0.4,
                    "num_predict": 120
                }
            )

            answer = response["message"]["content"]
            self.deliver_response(answer)
            response_sent = True

        except Exception as error:
            self.after(
                0,
                lambda: self.add_message(f"NOVA: Error: {error}\n\n")
            )

        finally:
            self.after(0, lambda: self.set_status("Say NOVA"))
            self.after(0, lambda: self.set_buttons_state("normal"))

            if not response_sent:
                self.after(0, self.start_wake_listener)


if __name__ == "__main__":
    app = NovaApp()
    app.mainloop() 