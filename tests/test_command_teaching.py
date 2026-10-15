import sqlite3
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock
from action_registry import ActionExecutor
from command_teaching import CommandTeacher
from knowledge_manager import KnowledgeManager
from train_model import load_examples


class TeachingTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.addCleanup(self.db.close)
        self.knowledge = KnowledgeManager(self.db)
        self.executor = ActionExecutor({}, {'open_app': Mock()})
        self.teacher = CommandTeacher(self.executor, self.knowledge)

    def test_confirmed_learning_flow(self):
        self.assertIn('teach', self.teacher.begin('bring up my music'))
        self.assertIn('existing intent', self.teacher.handle('yes'))
        self.assertIn('Choose an existing', self.teacher.handle('run_shell'))
        self.assertIn('bring up my music', self.teacher.handle('open_app'))
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM command_examples').fetchone()[0], 0)
        self.executor.handle('yes confirm')
        self.assertEqual(self.db.execute('SELECT phrase,intent,confirmed FROM command_examples').fetchone(),
                         ('bring up my music', 'open_app', 1))
        self.executor.handlers['open_app'].assert_not_called()

    def test_cancel_does_not_save(self):
        self.teacher.begin('bring up music')
        self.assertEqual(self.teacher.handle('cancel'), 'Action cancelled.')
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM command_examples').fetchone()[0], 0)

    def test_raw_program_rejected(self):
        self.assertIn('cannot learn', self.teacher.begin('powershell Remove-Item data'))
        self.assertIsNone(self.teacher.state)

    def test_two_confirmed_examples_required_for_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.db'
            db = sqlite3.connect(path)
            try:
                manager = KnowledgeManager(db)
                manager.save_example('bring up music', 'open_app', True)
                self.assertNotIn(('bring up music', 'open_app'), load_examples(db_path=path))
                manager.save_example('bring up editor', 'open_app', True)
                examples = load_examples(db_path=path)
                self.assertIn(('bring up music', 'open_app'), examples)
                self.assertIn(('bring up editor', 'open_app'), examples)
            finally:
                db.close()
