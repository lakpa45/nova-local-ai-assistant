from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from nova_actions import ActionExecutor, Action
from knowledge_manager import KnowledgeManager
from topic_learning import TopicLearning, read_text_file


class TopicTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.addCleanup(self.db.close)
        self.knowledge = KnowledgeManager(self.db)
        self.executor = ActionExecutor({}, {})
        self.learning = TopicLearning(self.executor, self.knowledge)

    def test_inline_learning_preview_confirmation_recall_and_forget(self):
        self.learning.begin({'content': 'Python is a programming language', 'mode': 'dictate'})
        preview = self.learning.handle('Python')
        self.assertIn('Python is a programming language', preview)
        self.assertEqual(self.knowledge.topics(), [])
        self.executor.handle('yes confirm')
        self.assertEqual(self.knowledge.search('Python')[0][2], 'Python is a programming language')
        self.assertIn('Python is a programming language', self.learning.request_forget('Python'))
        self.executor.handle('cancel')
        self.assertEqual(self.knowledge.topics(), ['Python'])
        self.learning.request_forget('Python')
        self.executor.handle('confirm')
        self.assertEqual(self.knowledge.topics(), [])

    def test_topic_learning_does_not_train_command_model(self):
        self.learning.begin({'topic': 'networks'})
        self.learning.handle('dictate')
        self.learning.handle('Networks have nodes.')
        self.executor.handle('confirm')
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM command_examples').fetchone()[0], 0)

    def test_file_validation_and_inert_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'example.txt'
            path.write_text('import os; os.system("never execute")', encoding='utf-8')
            content, source = read_text_file(str(path))
            self.assertIn('never execute', content)
            self.assertEqual(source, str(path.resolve()))
            bad = Path(directory) / 'bad.py'
            bad.write_text('raise RuntimeError()')
            with self.assertRaises(ValueError):
                read_text_file(str(bad))
            path.write_bytes(b'x' * (256 * 1024 + 1))
            with self.assertRaises(ValueError):
                read_text_file(str(path))

    def test_online_has_no_silent_download(self):
        self.learning.begin({'topic': 'neural networks'})
        with patch('topic_learning.build_opener') as network:
            self.assertIn('No online sources', self.learning.handle('online'))
            network.assert_not_called()

    def test_new_records_after_preview_are_not_deleted(self):
        self.knowledge.save('Python', 'old content')
        self.learning.request_forget('Python')
        self.knowledge.save('Python', 'new content')
        self.executor.handle('confirm')
        self.assertEqual(self.knowledge.search('Python')[0][2], 'new content')
