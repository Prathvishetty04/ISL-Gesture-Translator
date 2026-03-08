import pyttsx3

engine = pyttsx3.init('sapi5')
engine.say("Hello this is a test")
engine.runAndWait()