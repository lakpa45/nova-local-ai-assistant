import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
RECORD_SECONDS = 5
AUDIO_FILE = "voice_test.wav"

print("Loading local speech-recognition model...")
model = WhisperModel("base.en", device="cpu", compute_type="int8")

print(f"Speak now. Recording for {RECORD_SECONDS} seconds...")
audio = sd.rec(
    int(RECORD_SECONDS * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16"
)
sd.wait()

write(AUDIO_FILE, SAMPLE_RATE, audio)
print("Recording complete. Converting voice to text...")

segments, info = model.transcribe(AUDIO_FILE, language="en")

text = "".join(segment.text for segment in segments).strip()

print("\nYou said:")
print(text)