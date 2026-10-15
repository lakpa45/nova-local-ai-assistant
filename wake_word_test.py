import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer

MODEL_PATH = "models/vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000

audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    if status:
        print(status)

    audio_queue.put(bytes(indata))


print("Loading local wake-word model...")
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, SAMPLE_RATE)

print("Listening continuously. Say: NOVA")
print("Press Ctrl + C to stop.")

try:
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):
        while True:
            audio_data = audio_queue.get()

            if recognizer.AcceptWaveform(audio_data):
                result = json.loads(recognizer.Result())
                heard_text = result.get("text", "").lower()

                if heard_text:
                    print(f"You said: {heard_text}")

                if "nova" in heard_text:
                    print("NOVA detected!")

except KeyboardInterrupt:
    print("\nWake-word listener stopped.")