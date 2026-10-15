import builtins
from datetime import datetime
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from nova_actions import ActionExecutor
from command_processor import CommandProcessor
from knowledge_manager import KnowledgeManager


class CommandFlowTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.addCleanup(self.db.close)
        self.open = Mock(return_value='opened')
        self.close = Mock(return_value='closed')
        self.search = Mock(return_value='searched')
        self.app = SimpleNamespace(
            actions=ActionExecutor({'auto_approve': ['open_app', 'search_web']}, {
                'open_app': self.open, 'close_app': self.close, 'search_web': self.search}),
            handle_builtin_command=Mock(return_value='time or personal memory'),
            save_reminder=Mock(), database_path=':memory:')
        self.processor = CommandProcessor(self.app, KnowledgeManager(self.db))

    def test_required_commands_dispatch_with_values(self):
        for text, target in [('Open Spotify', 'Spotify'), ('Please launch WhatsApp', 'WhatsApp'),
                             ('Start Calculator', 'Calculator')]:
            self.assertEqual(self.processor.handle(text), 'opened')
            self.open.assert_called_with(target)
        self.assertIn('could lose unsaved work', self.processor.handle('Close WhatsApp'))
        self.close.assert_not_called()
        self.assertEqual(self.processor.handle('Yes, confirm'), 'closed')
        self.close.assert_called_once_with('WhatsApp')
        self.assertEqual(self.processor.handle('Search Google for Python tutorials'), 'searched')
        self.search.assert_called_once_with('google for Python tutorials')
        with patch('command_processor.webbrowser.open') as browser:
            self.processor.handle('Open YouTube')
            browser.assert_called_once_with('https://www.youtube.com')

    def test_absolute_reminder(self):
        self.assertIn('study', self.processor.handle('Remind me to study at 8 PM'))
        task, when = self.app.save_reminder.call_args.args
        self.assertEqual(task, 'study')
        self.assertEqual((when.hour, when.minute), (20, 0))
        self.assertGreater(when, datetime.now())

    def test_topic_learning_and_forgetting_flow(self):
        self.assertIn('dictate', self.processor.handle('Learn about neural networks'))
        self.processor.handle('dictate')
        self.assertIn('Say yes confirm', self.processor.handle('Neural networks learn weights.'))
        self.processor.handle('Yes, confirm')
        self.assertIn('learn weights', self.processor.handle('What do you know about neural networks?'))
        self.assertIn('Remove these', self.processor.handle('Forget neural networks'))
        self.processor.handle('No, cancel')
        self.assertIn('learn weights', self.processor.handle('What do you know about neural networks?'))

    def test_unknown_sentence_offers_teaching_without_execution(self):
        self.assertIn('teach', self.processor.handle('A completely unknown sentence'))
        self.open.assert_not_called()
        self.close.assert_not_called()
        self.search.assert_not_called()

    def test_misclassified_confirmation_cannot_confirm(self):
        self.processor.handle('Close WhatsApp')
        self.processor.classifier = Mock()
        self.processor.classifier.predict.return_value = {'intent': 'confirm_action', 'confidence': 1.0}
        self.assertIn('confirm or cancel', self.processor.handle('sure go ahead'))
        self.close.assert_not_called()

    def test_manual_retraining_is_confirmed_and_never_shell(self):
        with patch('train_model.train', return_value={'examples': 150}) as train, patch.object(self.processor.classifier, 'reload'):
            self.processor.handle('retrain command model')
            train.assert_not_called()
            self.assertIn('150 examples', self.processor.handle('confirm'))
            train.assert_called_once_with(db_path=':memory:')

    def test_raw_shell_and_unregistered_action_rejected(self):
        self.assertIn('not supported', self.processor.handle('powershell Remove-Item data'))
        self.assertIn('rejected', self.processor.handle('delete_file data'))
        self.open.assert_not_called()
        self.close.assert_not_called()

    def test_open_unseen_app_name_still_classifies(self):
        self.assertEqual(self.processor.handle('Open VLC'), 'opened')
        self.open.assert_called_once_with('VLC')

    def test_taught_phrase_can_request_missing_target(self):
        self.processor.classifier = Mock()
        self.processor.classifier.predict.return_value = {'intent': 'close_app', 'confidence': 0.99}
        self.assertIn('Which application', self.processor.handle('finish my music session'))
        self.assertIn('could lose unsaved work', self.processor.handle('Spotify'))
        self.close.assert_not_called()
        self.processor.handle('confirm')
        self.close.assert_called_once_with('Spotify')


class NoOllamaTests(unittest.TestCase):
    def test_real_app_starts_without_ollama_import(self):
        original = builtins.__import__
        def blocked(name, *args, **kwargs):
            if name == 'ollama' or name.startswith('ollama.'):
                raise ImportError('Ollama deliberately unavailable')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=blocked):
            import app
            n = app.NovaApp(db_path=':memory:', start_listener=False)
            try:
                n.widget.withdraw()
                n.update()
                self.assertIn('current time', n.handle_command('What time is it?'))
                with patch.object(n, 'open_installed_app', return_value='Opening Spotify.'):
                    self.assertEqual(n.handle_command('Open Spotify'), 'Opening Spotify.')
                self.assertIn('command assistant', n.handle_command('A completely unknown sentence'))
                self.assertEqual(n.handle_command('cancel'), 'Action cancelled.')
            finally:
                n.close_app()
