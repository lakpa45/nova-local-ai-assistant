from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading
import unittest
from knowledge_manager import KnowledgeManager


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:', check_same_thread=False)
        self.addCleanup(self.db.close)
        self.manager = KnowledgeManager(self.db)

    def test_insert_retrieve_search_and_delete(self):
        self.manager.save('neural networks', 'Neurons learn weights.', 'dictation')
        self.assertEqual(self.manager.search('neural networks')[0][2], 'Neurons learn weights.')
        self.assertEqual(self.manager.search('weights')[0][1], 'neural networks')
        self.assertEqual(self.manager.topics(), ['neural networks'])
        self.assertEqual(self.manager.forget_snapshot(self.manager.exact_topic('neural networks')), 1)
        self.assertEqual(self.manager.search('weights'), [])

    def test_sql_fallback(self):
        manager = KnowledgeManager(self.db, enable_fts=False)
        manager.save('Python', 'A programming language')
        self.assertEqual(manager.search('language')[0][1], 'Python')
        self.assertEqual(manager.search("'; DROP TABLE knowledge; --"), [])

    def test_migration_preserves_existing_data_and_is_repeatable(self):
        self.db.execute('CREATE TABLE memories(fact TEXT)')
        self.db.execute("INSERT INTO memories VALUES('original memory')")
        self.db.execute('CREATE TABLE reminders(task TEXT)')
        self.db.execute("INSERT INTO reminders VALUES('original reminder')")
        for _ in range(2):
            KnowledgeManager(self.db)
        self.assertEqual(self.db.execute('SELECT fact FROM memories').fetchone()[0], 'original memory')
        self.assertEqual(self.db.execute('SELECT task FROM reminders').fetchone()[0], 'original reminder')

    def test_example_requires_confirmation_and_known_intent(self):
        for intent, confirmed in [('open_app', False), ('run_shell', True)]:
            with self.assertRaises(ValueError):
                self.manager.save_example('open music', intent, confirmed)
        self.manager.save_example('open music', 'open_app', True)
        self.assertEqual(self.db.execute('SELECT confirmed FROM command_examples').fetchone()[0], 1)
        self.assertEqual(self.manager.topics(), [])

    def test_concurrent_storage(self):
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda i: self.manager.save(f'topic {i}', 'content'), range(30)))
        self.assertEqual(len(self.manager.topics()), 30)
