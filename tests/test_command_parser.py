import unittest
from command_parser import extract, phrase, unsafe_command


class ParserTests(unittest.TestCase):
    def test_targets_queries_topics_and_time(self):
        cases = [
            ('Please open Spotify', 'open_app', {'target': 'Spotify'}),
            ('Please launch WhatsApp', 'open_app', {'target': 'WhatsApp'}),
            ('Run Visual Studio Code', 'open_app', {'target': 'Visual Studio Code'}),
            ('Close WhatsApp', 'close_app', {'target': 'WhatsApp'}),
            ('Open YouTube', 'open_website', {'target': 'YouTube'}),
            ('Search Google for Python decorators', 'search_web', {'engine': 'google', 'query': 'Python decorators'}),
            ('Remind me to study at 8 PM', 'set_reminder', {'reminder': 'study', 'time': '8 PM'}),
            ('Learn about neural networks', 'learn_topic', {'topic': 'neural networks'}),
            ('What do you know about neural networks?', 'recall_information', {'topic': 'neural networks'}),
            ('Forget neural networks', 'forget_topic', {'topic': 'neural networks'}),
        ]
        for text, intent, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(extract(text, intent).values, expected)

    def test_invalid_or_misclassified_command_does_not_invent_target(self):
        self.assertIsNone(extract('hello', 'close_app'))
        self.assertIsNone(extract('close', 'close_app'))
        self.assertIsNone(extract('say hello', 'unknown'))

    def test_confirmation_punctuation(self):
        self.assertEqual(phrase('Yes, confirm!'), 'yes confirm')
        self.assertEqual(phrase('No, cancel.'), 'no cancel')

    def test_programs_cannot_be_taught(self):
        for text in ('powershell Remove-Item x', 'Open Spotify; delete x', 'python -c evil', 'import os'):
            self.assertTrue(unsafe_command(text))
        self.assertFalse(unsafe_command('Please launch WhatsApp'))
