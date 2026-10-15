"""Track background work and serialize microphone ownership for the existing GUI."""
import threading


class VoiceManager:
    def __init__(self):
        self.microphone = threading.Lock()
        self.lock = threading.Lock()
        self.threads = set()
        self.stopping = False

    def start(self, target, args=()):
        def run():
            try:
                target(*args)
            finally:
                with self.lock:
                    self.threads.discard(threading.current_thread())
        with self.lock:
            if self.stopping:
                return None
            worker = threading.Thread(target=run, daemon=True)
            self.threads.add(worker)
            worker.start()
            return worker

    def stop(self):
        with self.lock:
            self.stopping = True

    def active(self):
        with self.lock:
            return any(worker.is_alive() for worker in self.threads)
