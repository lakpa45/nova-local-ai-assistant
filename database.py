"""Additive SQLite migrations. Existing reminders and memories are untouched."""
import sqlite3


def migrate(connection, enable_fts=True):
    with connection:
        connection.execute('''CREATE TABLE IF NOT EXISTS command_examples (
            example_id INTEGER PRIMARY KEY AUTOINCREMENT,
            phrase TEXT NOT NULL UNIQUE, intent TEXT NOT NULL,
            confirmed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
        connection.execute('''CREATE TABLE IF NOT EXISTS knowledge (
            knowledge_id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL, content TEXT NOT NULL, source TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
        connection.execute('CREATE INDEX IF NOT EXISTS knowledge_topic ON knowledge(topic COLLATE NOCASE)')
    if not enable_fts:
        return False
    try:
        with connection:
            connection.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts
                USING fts5(topic, content, content='knowledge', content_rowid='knowledge_id')''')
            connection.execute('''CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
                INSERT INTO knowledge_fts(rowid,topic,content) VALUES(new.knowledge_id,new.topic,new.content); END''')
            connection.execute('''CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge BEGIN
                INSERT INTO knowledge_fts(knowledge_fts,rowid,topic,content)
                VALUES('delete',old.knowledge_id,old.topic,old.content); END''')
            connection.execute('''CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge BEGIN
                INSERT INTO knowledge_fts(knowledge_fts,rowid,topic,content)
                VALUES('delete',old.knowledge_id,old.topic,old.content);
                INSERT INTO knowledge_fts(rowid,topic,content) VALUES(new.knowledge_id,new.topic,new.content); END''')
            connection.execute("INSERT INTO knowledge_fts(knowledge_fts) VALUES('rebuild')")
        return True
    except sqlite3.OperationalError as error:
        if 'no such module' not in str(error).lower():
            raise
        return False
