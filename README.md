# NOVA personal command assistant

Windows desktop assistant with a floating widget, Vosk wake-word detection,
Faster-Whisper transcription, pyttsx3 speech, and a small CPU PyTorch intent
classifier. Ollama is **not required** for startup or commands.

## Install and run (PowerShell, from this folder)

```powershell
# If a virtual environment does not exist yet:
python -m venv venv

.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe train_model.py
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe app.py
```

The existing Vosk model stays in `models/vosk-model-small-en-us-0.15`.
Faster-Whisper uses `base.en`; its first use may download speech weights if they
are not already cached. A working microphone and Windows speech output are
needed for voice; text commands use the same pipeline.

For an isolated startup check (in-memory database, no commands executed):

```powershell
.\venv\Scripts\python.exe app.py --smoke-test
```

## Commands

- `Open Spotify`, `Please launch WhatsApp`, `Run Visual Studio Code`, `Open YouTube`
- `Close WhatsApp`, `Quit Spotify`, `Close Google Chrome`
- `Search Google for Python tutorials`, `Search YouTube for music`
- `Remind me in 5 minutes to stretch`, `Remind me to study at 8 PM`
- `What time is it?`, `Show reminders`, `Weather in Delhi`
- `Remember that I like tea`, `Show my memories`, `Forget that I like tea`
- `Learn this: Python is a programming language` (then supply a topic name)
- `Learn about neural networks` (then choose `dictate`, `file`, or `online`)
- `Learn from file` (then supply a topic name and local text-file path)
- `What do you know about neural networks?`, `List learned topics`, `Forget neural networks`
- `Retrain command model` (requires confirmation)

Sensitive actions and learning previews require `yes confirm`, `confirm`, or
`continue`. Cancel with `no cancel`, `cancel`, or `stop`. Punctuation such as
`Yes, confirm` is accepted. An unrelated command cannot bypass a pending
confirmation. Confirmations expire after 30 seconds; teaching/source-selection
dialogues expire after two minutes. Reminder times without a date use the next
occurrence of that local time.

When NOVA does not recognize a phrase, it offers command teaching. Accept,
choose an existing intent from the displayed list, and explicitly confirm the
phrase/intent pair. This does not execute the phrase or retrain immediately.
At least two confirmed examples for an intent are required before taught data
is included by `train_model.py` or the confirmed `retrain command model` action.
If a learned phrase omits an extractable app name, NOVA asks which app to use.

Topic learning stores information, not executable shortcuts or training text.
Only UTF-8 `.txt`, `.md`, and `.csv` files up to 256 KB are accepted. The full
preview/result is displayed in chat; speech is shortened. Online learning is
disabled until exact HTTPS plain-text URLs are added to
`APPROVED_ONLINE_SOURCES` in `config.py`; the user selects a source before any
fetch. Redirects and arbitrary URLs are rejected.

## Layout and safety

The existing flat project layout is retained rather than moving working code
into a second `nova/` directory:

- `app.py`: existing GUI, floating widget, voice, reminders, and memories.
- `voice_manager.py`: microphone ownership and background-worker lifecycle.
- `intent_classifier.py`, `train_model.py`, `training_data.json`: feed-forward
  network, reproducible explicit training, and seed examples.
- `command_parser.py`, `command_processor.py`: deterministic value extraction
  and the shared text/voice pipeline.
- `nova_actions.py`, `action_registry.py`, `nova_processes.py`: registered
  callables, confirmation state, and protected application closing.
- `database.py`, `knowledge_manager.py`, `command_teaching.py`,
  `topic_learning.py`: additive migrations, retrieval, and learning dialogues.
- `config.py`, `optional_llm.py`: settings and optional general chat.
- `models/intent_classifier.pth`, `models/vocabulary.json`, `models/labels.json`:
  saved classifier artifacts. Weights load with `weights_only=True` on CPU;
  startup never retrains. The checkpoint contains authoritative metadata so a
  live reload cannot mix vocabulary and weights from different training runs.
- `tests/`: automated tests with mocked application/process side effects.

The default confidence threshold is 0.75. Low-confidence predictions become
`unknown`. A prediction is never an executable command. Only explicitly
registered Python functions can run; shell snippets cannot be taught as commands.
Existing `permission.json` settings are preserved and extended by
`permissions.json` and the new registry. Closing, forgetting, storing learned
data, and retraining require confirmation.

SQLite adds `command_examples` and `knowledge` without changing existing
`memories` or `reminders`. FTS5 search and synchronization triggers are created
when available, with normal SQL search as the fallback. Deletion uses the exact
records shown in the preview, so newer records are not silently deleted.

Closing uses exact approved process aliases, `psutil.process_iter()` and
`terminate()`, never a shell or force-kill fallback. Windows system processes,
NOVA's runtime tree/hosts, Python, and assistant dependencies are protected.
On Windows, `terminate()` is not guaranteed to offer a save dialog: save your
work before confirming. An editor hosting NOVA may be refused for this reason.

## Limitations and optional chat

The small classifier is intended for these command intents, not general
language understanding. Training accuracy does not measure arbitrary unseen
speech accuracy. Strict extraction, confidence rejection, and confirmations
remain the safety boundaries. Weather currently opens a browser forecast;
NOVA does not synthesize live weather readings. Approved websites are listed
in `command_processor.WEBSITES`; app closing aliases remain in
`nova_processes.PROCESS_ALIASES`.

For optional general chat, install `ollama`, run a local Ollama service with the
configured model, and set `ENABLE_LOCAL_LLM = True` in `config.py`. This is off
by default. Replies are only displayed/spoken, never sent to the action registry.

Automated tests mock transcription, TTS, web opening, and process termination;
they do not close real applications. Test actual voice recognition and audio
output manually on your hardware after the automated checks pass.
