"""Run explicitly with: python train_model.py. Never trained on app startup."""
import json
import os
from pathlib import Path
import random
import sqlite3
import tempfile
from contextlib import closing
import torch
import config
from intent_classifier import IntentNetwork, features, vectorize


def load_examples(data_path=config.TRAINING_DATA, db_path=config.DATABASE_PATH):
    data = json.loads(Path(data_path).read_text(encoding='utf-8'))
    examples = [(phrase, intent) for intent, phrases in data.items() for phrase in phrases]
    if db_path and Path(db_path).exists():
        with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + '?mode=ro', uri=True)) as db:
            exists = db.execute("SELECT 1 FROM sqlite_master WHERE name='command_examples'").fetchone()
            if exists:
                rows = db.execute('''SELECT phrase, intent FROM command_examples
                    WHERE confirmed=1 AND intent IN (SELECT intent FROM command_examples
                    WHERE confirmed=1 GROUP BY intent HAVING COUNT(*) >= 2)
                    ORDER BY example_id''').fetchall()
                # A database record cannot add a new executable intent.
                examples.extend((phrase, intent) for phrase, intent in rows if intent in data)
    return examples


def train(data_path=config.TRAINING_DATA, model_dir=config.MODEL_DIR,
          db_path=config.DATABASE_PATH, epochs=500):
    random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    examples = load_examples(data_path, db_path)
    vocabulary = {word: i for i, word in enumerate(sorted({
        word for phrase, _ in examples for word in features(phrase)}))}
    labels = sorted({intent for _, intent in examples})
    x = torch.stack([vectorize(phrase, vocabulary) for phrase, _ in examples])
    y = torch.tensor([labels.index(intent) for _, intent in examples])
    model = IntentNetwork(len(vocabulary), len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    accuracy = float((model(x).argmax(dim=1) == y).float().mean())
    destination = Path(model_dir)
    destination.mkdir(parents=True, exist_ok=True)
    # All authoritative metadata is in one atomically replaced checkpoint.
    with tempfile.NamedTemporaryFile(dir=destination, suffix='.tmp', delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        torch.save({'state_dict': model.state_dict(), 'vocabulary': vocabulary,
                    'labels': labels, 'seed': config.RANDOM_SEED}, temporary)
        os.replace(temporary, destination / 'intent_classifier.pth')
    finally:
        temporary.unlink(missing_ok=True)
    for filename, value in (('vocabulary.json', vocabulary), ('labels.json', labels)):
        (destination / filename).write_text(json.dumps(value, indent=2), encoding='utf-8')
    return {'examples': len(examples), 'training_accuracy': accuracy,
            'parameters': sum(p.numel() for p in model.parameters())}


if __name__ == '__main__':
    print(json.dumps(train(), indent=2))
