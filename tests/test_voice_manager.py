import threading
import unittest
from voice_manager import VoiceManager


class VoiceManagerTests(unittest.TestCase):
    def test_tracks_worker_and_rejects_work_after_shutdown(self):
        manager = VoiceManager()
        entered, release = threading.Event(), threading.Event()
        def work():
            entered.set()
            release.wait(timeout=2)
        worker = manager.start(work)
        self.assertTrue(entered.wait(timeout=1))
        self.assertTrue(manager.active())
        manager.stop()
        self.assertIsNone(manager.start(lambda: None))
        release.set()
        worker.join(timeout=2)
        self.assertFalse(manager.active())

    def test_microphone_has_only_one_owner(self):
        manager = VoiceManager()
        acquired = []
        with manager.microphone:
            thread = manager.start(lambda: acquired.append(manager.microphone.acquire(blocking=False)))
            thread.join(timeout=1)
        self.assertEqual(acquired, [False])
