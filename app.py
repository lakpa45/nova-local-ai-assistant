import threading
import subprocess
import sqlite3
import re
import webbrowser
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import customtkinter as ctk
import ollama
import pyttsx3
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel


MODEL_NAME = "qwen3:4b"
SAMPLE_RATE = 16000
RECORD_SECONDS = 6
AUDIO_FILE = "nova_voice.wav"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class NovaApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.whisper_model = None
        self.db_lock = threading.Lock()

        self.setup_database()

        self.title("NOVA – Personal AI Assistant")
        self.geometry("850x600")
        self.minsize(650, 450)
        self.protocol("WM_DELETE_WINDOW", self.hide_chat)

        self.create_chat_window()
        self.create_floating_widget()
        self.check_reminders()

        # Start with only the floating NOVA widget visible.
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

        self.db.commit()

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

            due_reminders = self.cursor.fetchall()

            for reminder_id, _ in due_reminders:
                self.cursor.execute(
                    "UPDATE reminders SET completed = 1 WHERE reminder_id = ?",
                    (reminder_id,)
                )

            self.db.commit()

        for _, task in due_reminders:
            message = f"Reminder: {task}"
            self.add_message(f"NOVA: {message}\n\n")
            self.set_status("Reminder!", "#f6e05e")

            threading.Thread(
                target=self.speak,
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
            text="Local AI Assistant • Ready",
            text_color="#68d391"
        )
        self.status_label.pack(pady=(0, 15))

        self.chat_box = ctk.CTkTextbox(self, wrap="word", font=("Arial", 15))
        self.chat_box.pack(fill="both", expand=True, padx=25, pady=10)
        self.chat_box.insert(
            "end",
            "NOVA: Hello! Click Listen and speak your question.\n\n"
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
            text="Ready to help",
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
        self.db.close()
        self.destroy()

    def open_camera(self):
        try:
            subprocess.Popen("start microsoft.windows.camera:", shell=True)
        except Exception as error:
            print(f"Camera error: {error}")

    def add_message(self, message):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", message)
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def set_status(self, text, color="#68d391"):
        self.status_label.configure(text=text, text_color=color)
        self.widget_status.configure(text=text, text_color=color)

    def set_buttons_state(self, state):
        self.listen_button.configure(state=state)
        self.widget_listen_button.configure(state=state)
        self.send_button.configure(state=state)

    def speak(self, text):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
        except Exception as error:
            print(f"Speech error: {error}")

    def start_listening(self):
        self.set_status("Listening...", "#f6e05e")
        self.set_buttons_state("disabled")

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

            self.after(
                0,
                lambda: self.set_status(
                    f"Speak now ({RECORD_SECONDS} seconds)...",
                    "#f6e05e"
                )
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
        self.set_status("Ready to help")
        self.set_buttons_state("normal")

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

    def handle_command(self, message):
        command = message.lower().strip()

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

            return (
                f"Okay. I will remind you in {number} {unit} "
                f"to {task}."
            )

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

            reminder_list = []

            for task, reminder_time in reminders:
                formatted_time = datetime.fromisoformat(
                    reminder_time
                ).strftime("%I:%M %p")

                reminder_list.append(f"{task} at {formatted_time}")

            return "Your reminders are: " + ". ".join(reminder_list)

        if "open camera" in command:
            self.open_camera()
            return "Opening the Camera."

        if "open calculator" in command:
            subprocess.Popen("start calc", shell=True)
            return "Opening Calculator."

        if "open chrome" in command or "open browser" in command:
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

        return None

    def deliver_response(self, answer):
        self.after(0, lambda: self.add_message(f"NOVA: {answer}\n\n"))

        threading.Thread(
            target=self.speak,
            args=(answer,),
            daemon=True
        ).start()

    def get_nova_response(self, user_message):
        try:
            command_answer = self.handle_command(user_message)

            if command_answer:
                self.deliver_response(command_answer)
                return

            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are NOVA, a helpful personal AI assistant. "
                            "Give clear, friendly, concise answers."
                        )
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            )

            answer = response["message"]["content"]
            self.deliver_response(answer)

        except Exception as error:
            self.after(
                0,
                lambda: self.add_message(f"NOVA: Error: {error}\n\n")
            )

        finally:
            self.after(
                0,
                lambda: self.set_status("Ready to help")
            )
            self.after(
                0,
                lambda: self.set_buttons_state("normal")
            )


if __name__ == "__main__":
    app = NovaApp()
    app.mainloop()