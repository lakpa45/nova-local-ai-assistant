"""Thread-safe data storage; learned text is never interpreted as code."""
import re
import sqlite3
import threading
from database import migrate
from command_parser import unsafe_command
from action_registry import TEACHABLE_INTENTS


class KnowledgeManager:
    def __init__(self, connection, lock=None, enable_fts=True):
        self.db = connection
        self.lock = lock if lock is not None else threading.RLock()
        with self.lock:
            self.fts = migrate(connection, enable_fts)

    def save_example(self, phrase, intent, confirmed=False):
        if not confirmed or intent not in TEACHABLE_INTENTS or unsafe_command(phrase):
            raise ValueError('Only confirmed phrases for existing intents can be learned.')
        phrase = phrase.strip()
        if not phrase or len(phrase) > 1000:
            raise ValueError('Command phrase must contain 1 to 1000 characters.')
        with self.lock, self.db:
            # A repeated phrase never silently relabels a previously confirmed example.
            self.db.execute('INSERT OR IGNORE INTO command_examples(phrase,intent,confirmed) VALUES(?,?,1)',
                            (phrase, intent))

    def save(self, topic, content, source='dictation'):
        topic, content = topic.strip(), content.strip()
        if not topic or not content or len(topic) > 200 or len(content.encode('utf-8')) > 256 * 1024:
            raise ValueError('Provide a topic (up to 200 characters) and bounded, nonempty text.')
        with self.lock, self.db:
            cursor = self.db.execute('INSERT INTO knowledge(topic,content,source) VALUES(?,?,?)',
                                     (topic, content, source))
            return cursor.lastrowid

    def search(self, query, limit=5):
        query = query.strip()
        if not query:
            return []
        with self.lock:
            rows = self.db.execute('SELECT knowledge_id,topic,content,source FROM knowledge '
                                   'WHERE topic=? COLLATE NOCASE ORDER BY knowledge_id DESC LIMIT ?',
                                   (query, limit)).fetchall()
            if rows:
                return rows
            terms = re.findall(r'\w+', query)
            if self.fts and terms:
                expression = ' OR '.join('"' + term + '"' for term in terms)
                rows = self.db.execute('''SELECT k.knowledge_id,k.topic,k.content,k.source
                    FROM knowledge_fts f JOIN knowledge k ON k.knowledge_id=f.rowid
                    WHERE knowledge_fts MATCH ? ORDER BY bm25(knowledge_fts) LIMIT ?''',
                    (expression, limit)).fetchall()
                if rows:
                    return rows
            # instr is literal: wildcard/query syntax from the user is not SQL.
            return self.db.execute('''SELECT knowledge_id,topic,content,source FROM knowledge
                WHERE instr(lower(topic),lower(?)) > 0 OR instr(lower(content),lower(?)) > 0
                ORDER BY knowledge_id DESC LIMIT ?''', (query, query, limit)).fetchall()

    def exact_topic(self, topic):
        with self.lock:
            return self.db.execute('SELECT knowledge_id,topic,content,source FROM knowledge '
                                   'WHERE topic=? COLLATE NOCASE ORDER BY knowledge_id', (topic,)).fetchall()

    def forget_snapshot(self, rows):
        """Delete only records, with contents, shown in the confirmation preview."""
        count = 0
        with self.lock, self.db:
            for identifier, topic, content, source in rows:
                cursor = self.db.execute('DELETE FROM knowledge WHERE knowledge_id=? AND topic=? '
                                         'AND content=? AND source IS ?', (identifier, topic, content, source))
                count += cursor.rowcount
        return count

    def topics(self):
        with self.lock:
            return [row[0] for row in self.db.execute('SELECT DISTINCT topic FROM knowledge ORDER BY topic COLLATE NOCASE')]
