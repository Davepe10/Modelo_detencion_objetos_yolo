import pyttsx3
from threading import Lock
import queue

engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine_lock = Lock()
message_queue = queue.Queue()

def speak(text):
    message_queue.put(text)

def process_queue():
    while True:
        text = message_queue.get()
        with engine_lock:
            engine.say(text)
            engine.runAndWait()
        message_queue.task_done()
