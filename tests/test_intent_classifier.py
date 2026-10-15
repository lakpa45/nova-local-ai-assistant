import tempfile
import unittest
from intent_classifier import IntentClassifier, tokenize


class IntentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = IntentClassifier()

    def test_required_intents(self):
        samples = {
            'Open Spotify': 'open_app', 'Please launch WhatsApp': 'open_app',
            'Start Calculator': 'open_app', 'Close WhatsApp': 'close_app',
            'Search Google for Python tutorials': 'search_web', 'Open YouTube': 'open_website',
            'Remind me to study at 8 PM': 'set_reminder', 'What time is it?': 'get_time',
            'Learn about neural networks': 'learn_topic',
            'What do you know about neural networks?': 'recall_information',
            'Forget neural networks': 'forget_topic', 'Yes, confirm': 'confirm_action',
            'No, cancel': 'reject_action', 'A completely unknown sentence': 'unknown',
        }
        for phrase, expected in samples.items():
            with self.subTest(phrase=phrase):
                result = self.classifier.predict(phrase)
                self.assertEqual(result['intent'], expected, result)
                self.assertTrue(0 <= result['confidence'] <= 1)

    def test_threshold_and_empty_or_unseen_text(self):
        self.assertEqual(IntentClassifier(threshold=1.1).predict('Open Spotify')['intent'], 'unknown')
        for phrase in ('', 'zqxwy blorb'):
            self.assertEqual(self.classifier.predict(phrase), {'intent': 'unknown', 'confidence': 0.0})

    def test_missing_model_never_trains(self):
        with tempfile.TemporaryDirectory() as directory:
            classifier = IntentClassifier(model_dir=directory)
            self.assertEqual(classifier.predict('Open Spotify')['intent'], 'unknown')

    def test_normalization(self):
        self.assertEqual(tokenize('Yes, CONFIRM!'), ['yes', 'confirm'])
