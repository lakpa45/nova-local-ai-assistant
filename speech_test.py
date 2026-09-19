import pyttsx3

engine = pyttsx3.init()

engine.setProperty('rate', 175)
engine.setProperty("volume", 1.0)

engine.say("Hello. I am NOVA, your personal voice assistant.")
engine.runAndWait()