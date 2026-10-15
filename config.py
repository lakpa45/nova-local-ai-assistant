"""Local NOVA settings. Paths are independent of the working directory."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'
DATABASE_PATH = BASE_DIR / 'nova.db'
TRAINING_DATA = BASE_DIR / 'training_data.json'
CONFIDENCE_THRESHOLD = 0.75
RANDOM_SEED = 42
ENABLE_LOCAL_LLM = False
LOCAL_LLM_MODEL = 'qwen3:4b'
MAX_LEARNING_BYTES = 256 * 1024
APPROVED_TEXT_EXTENSIONS = {'.txt', '.md', '.csv'}
# Online learning is opt-in. Supply exact HTTPS URLs, never arbitrary domains.
APPROVED_ONLINE_SOURCES = {}
COMMAND_ONLY_RESPONSE = ('I currently work as a command assistant. You can teach me '
                         'information by saying, NOVA, learn this topic.')
