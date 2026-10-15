"""Small CPU-only intent network. Predictions never contain executable code."""
import json
import re
import torch
from torch import nn
import config


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def features(text):
    words = tokenize(text)
    return words + [a + ' ' + b for a, b in zip(words, words[1:])]


def vectorize(text, vocabulary):
    vector = torch.zeros(len(vocabulary), dtype=torch.float32)
    for word in features(text):
        index = vocabulary.get(word)
        if index is not None:
            vector[index] = 1.0
    return vector


class IntentNetwork(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, 64), nn.ReLU(),
                                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, outputs))

    def forward(self, x):
        return self.layers(x)


class IntentClassifier:
    def __init__(self, model_dir=config.MODEL_DIR, threshold=config.CONFIDENCE_THRESHOLD):
        self.threshold = threshold
        self.model_dir = model_dir
        self.model = None
        self.error = None
        self.reload()

    def reload(self):
        from pathlib import Path
        try:
            checkpoint = torch.load(Path(self.model_dir) / 'intent_classifier.pth',
                                    map_location='cpu', weights_only=True)
            vocabulary, labels = checkpoint['vocabulary'], checkpoint['labels']
            model = IntentNetwork(len(vocabulary), len(labels))
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            self.vocabulary, self.labels, self.model = vocabulary, labels, model
            self.error = None
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            self.error = str(error)
            self.model = None

    def predict(self, text):
        if self.model is None:
            return {'intent': 'unknown', 'confidence': 0.0}
        vector = vectorize(text, self.vocabulary)
        if not torch.any(vector):
            return {'intent': 'unknown', 'confidence': 0.0}
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(vector), dim=-1)
            confidence, index = probabilities.max(dim=-1)
        score = float(confidence)
        return {'intent': self.labels[int(index)] if score >= self.threshold else 'unknown',
                'confidence': score}
