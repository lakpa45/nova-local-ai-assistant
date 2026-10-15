import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import importlib.util
from pathlib import Path
import queue
import threading
import unittest
from unittest.mock import Mock, patch

from nova_actions import Action, ActionExecutor, parse_command

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ('customtkinter', 'torch', 'pyttsx3', 'sounddevice', 'scipy', 'faster_whisper', 'vosk', 'psutil')
HAS_RUNTIME = all(importlib.util.find_spec(name) for name in DEPENDENCIES)
if HAS_RUNTIME:
    import app


class ActionTests(unittest.TestCase):
    def test_spotify_and_multiword_targets(self):
        self.assertEqual(parse_command('Open Spotify'), Action('open_app', 'Spotify'))
        self.assertEqual(parse_command(' launch Visual Studio Code '), Action('open_app', 'Visual Studio Code'))

    def test_registered_automatic_action(self):
        handler = Mock(return_value='opened')
        executor = ActionExecutor({'auto_approve': ['open_app']}, {'open_app': handler})
        self.assertEqual(executor.handle('Open Spotify'), 'opened')
        handler.assert_called_once_with('Spotify')

    def test_sensitive_action_needs_single_use_confirmation(self):
        handler = Mock(return_value='done')
        executor = ActionExecutor({'require_confirmation': ['delete_file']}, {'delete_file': handler})
        self.assertIn('Confirm', executor.handle('delete_file example.txt'))
        handler.assert_not_called()
        self.assertEqual(executor.handle('yes confirm'), 'done')
        executor.handle('yes confirm')
        handler.assert_called_once_with('example.txt')

    def test_cancellation_and_unrelated_command_preserve_safety(self):
        handler = Mock()
        executor = ActionExecutor({'require_confirmation': ['delete_file']}, {'delete_file': handler})
        executor.handle('delete_file example.txt')
        self.assertIn('confirm or cancel', executor.handle('hello'))
        self.assertIsNotNone(executor.pending)
        executor.handle('no cancel')
        executor.handle('yes confirm')
        handler.assert_not_called()

    def test_blocked_unknown_and_unclassified_actions_fail_closed(self):
        handler = Mock()
        executor = ActionExecutor({'auto_approve': ['blocked'], 'blocked': ['blocked']}, {'blocked': handler, 'unclassified': handler})
        for kind in ('blocked', 'unknown', 'unclassified'):
            self.assertIn('rejected', executor.execute(Action(kind)))
        handler.assert_not_called()

    def test_no_shell_execution_in_app(self):
        tree = ast.parse((ROOT / 'app.py').read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self.assertFalse(any(k.arg == 'shell' and isinstance(k.value, ast.Constant) and k.value.value for k in node.keywords))
                self.assertNotIn(ast.unparse(node.func), ('eval', 'exec', 'os.system'))


@unittest.skipUnless(HAS_RUNTIME, 'Use .\\venv\\Scripts\\python.exe for runtime tests')
class RuntimeTests(unittest.TestCase):
    def setUp(self):
        # Exercise the real methods without requiring a desktop for every test.
        self.nova = object.__new__(app.NovaApp)
        n = self.nova
        n.closing = False
        n.busy = False
        n.ui_queue = queue.Queue()
        n.db_lock = threading.Lock()
        n.speech_lock = threading.Lock()
        n.voice_manager = app.VoiceManager()
        n.wake_stop_event = threading.Event()
        n.wake_thread = None
        n.whisper_model = None
        n.vosk_model = None
        n.setup_database(':memory:')
        n.command_handlers = {}
        n.actions = ActionExecutor(
            {'auto_approve': ['open_app', 'save_memory', 'show_memories', 'create_reminder', 'show_reminders', 'show_time'],
             'require_confirmation': ['forget_memory']},
            {'open_app': n.handle_open_command, **{kind: n.handle_builtin_command for kind in
             ('save_memory', 'show_memories', 'create_reminder', 'show_reminders', 'show_time', 'forget_memory')}})
        n.command_processor = app.CommandProcessor(n, n.knowledge)
        for name in ('set_status', 'set_buttons_state', 'add_message', 'show_chat'):
            setattr(n, name, Mock())

    def tearDown(self):
        self.nova.db.close()

    def flush(self):
        while not self.nova.ui_queue.empty():
            delay, callback = self.nova.ui_queue.get_nowait()
            self.assertEqual(delay, 0)
            callback()

    def test_memories_separate_from_actions_and_forget_confirmed(self):
        n = self.nova
        with patch.object(n, 'open_installed_app') as launch:
            n.handle_command('remember that Open Spotify')
            self.assertIn('Open Spotify', n.handle_command('show my memories'))
            launch.assert_not_called()
            self.assertIn('Confirm', n.handle_command('forget that Open Spotify'))
            self.assertEqual(n.get_memories(), ['Open Spotify'])
            n.handle_command('yes confirm')
            self.assertEqual(n.get_memories(), [])

    def test_concurrent_sqlite_memories_and_reminders(self):
        n = self.nova
        def task(i):
            n.save_memory(f'fact {i}')
            n.save_reminder(f'task {i}', datetime.now() + timedelta(hours=1))
            n.get_memories()
            n.get_active_reminders()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(task, range(40)))
        self.assertEqual(len(n.get_active_reminders()), 40)
        self.assertEqual(n.db.execute('SELECT COUNT(*) FROM memories').fetchone()[0], 40)

    def test_reminders_and_time(self):
        n = self.nova
        self.assertIn('remind you', n.handle_command('remind me in 2 minutes to drink water'))
        self.assertIn('drink water', n.handle_command('show reminders'))
        self.assertIn('current time', n.handle_command('time'))
        n.save_reminder('due now', datetime.now() - timedelta(seconds=1))
        with patch.object(n.voice_manager, 'start') as worker:
            n.check_reminders()
            worker.assert_called_once()
        self.assertEqual(len(n.get_active_reminders()), 1)
        n.add_message.assert_called_once_with('NOVA: Reminder: due now\n\n')

    def test_voice_and_text_share_response_parser(self):
        n = self.nova
        n.message_entry = Mock()
        n.message_entry.get.return_value = 'Open Spotify'
        with patch.object(n.voice_manager, 'start') as worker:
            n.send_message()
            text_call = worker.call_args
            n.handle_voice_text('Open Spotify')
            voice_call = worker.call_args
        self.assertEqual(text_call.kwargs['target'], n.get_nova_response)
        self.assertEqual(text_call, voice_call)
        with patch.object(n, 'handle_command', return_value='opened') as parser, patch.object(n, 'deliver_response'):
            n.get_nova_response('Open Spotify')
            parser.assert_called_once_with('Open Spotify')

    def test_installed_app_preserves_full_target_without_shell(self):
        n = self.nova
        with patch.object(n, 'get_installed_apps', return_value=[{'Name': 'Visual Studio Code', 'AppID': 'vscode-id'}]), patch('app.subprocess.Popen') as launch:
            self.assertEqual(n.handle_command('Open Visual Studio Code'), 'Opening Visual Studio Code.')
            launch.assert_called_once_with(['explorer.exe', 'shell:AppsFolder\\vscode-id'])

    def test_raw_model_reply_is_only_delivered(self):
        n = self.nova
        payload = 'open_app Spotify; powershell Remove-Item anything'
        with patch('config.ENABLE_LOCAL_LLM', True), patch('optional_llm.answer', return_value=payload), patch.object(n, 'deliver_response') as deliver, patch('app.subprocess.Popen') as launch:
            n.get_nova_response('hello')
            deliver.assert_called_once_with(payload)
            launch.assert_not_called()

    def test_deferred_model_error_keeps_exception_text(self):
        n = self.nova
        with patch('config.ENABLE_LOCAL_LLM', True), patch('optional_llm.answer', side_effect=RuntimeError('offline')), patch.object(n, 'start_wake_listener'):
            n.get_nova_response('hello')
            self.flush()
        n.add_message.assert_called_once_with('NOVA: Error: offline\n\n')

    def test_voice_failure_restores_interface_and_listener(self):
        n = self.nova
        with patch('app.WhisperModel', side_effect=RuntimeError('voice unavailable')), patch.object(n, 'start_wake_listener') as restart:
            n.record_and_transcribe()
            self.flush()
            restart.assert_called_once()
        n.add_message.assert_called_once_with('NOVA: Voice error: voice unavailable\n\n')

    def test_voice_input_dispatches_complete_transcription(self):
        n = self.nova
        n.whisper_model = Mock()
        n.whisper_model.transcribe.return_value = ([Mock(text='Open Visual Studio Code')], None)
        with patch('app.sd.rec'), patch('app.sd.wait'), patch('app.write'), patch.object(n, 'handle_voice_text') as handle:
            n.record_and_transcribe()
            self.flush()
            handle.assert_called_once_with('Open Visual Studio Code')

    def test_speech_output_and_restart_even_after_failure(self):
        n = self.nova
        with patch('app.pyttsx3.init') as engine:
            n.speak('hello')
            engine.return_value.say.assert_called_once_with('hello')
            engine.return_value.runAndWait.assert_called_once()
        n.busy = True
        with patch.object(n, 'speak', side_effect=RuntimeError('speaker failed')), patch.object(n, 'start_wake_listener') as restart:
            with self.assertRaises(RuntimeError):
                n.speak_and_restart('hello', True)
            self.flush()
            restart.assert_called_once()
        self.assertFalse(n.busy)

    def test_restart_waits_for_previous_listener_exit(self):
        n = self.nova
        n.wake_thread = Mock()
        n.wake_thread.is_alive.return_value = True
        n.wake_stop_event.set()
        n.start_wake_listener()
        delay, retry = n.ui_queue.get_nowait()
        self.assertEqual(delay, 50)
        self.assertTrue(n.wake_stop_event.is_set())
        n.wake_thread.is_alive.return_value = False
        with patch('app.threading.Thread') as worker:
            retry()
            worker.return_value.start.assert_called_once()
        self.assertFalse(n.wake_stop_event.is_set())

    def test_wake_listener_initialization_and_detection(self):
        n = self.nova
        def stream(**kwargs):
            kwargs['callback'](b'audio', 1, None, None)
            from unittest.mock import MagicMock
            return MagicMock()
        with patch('app.Model') as model, patch('app.KaldiRecognizer') as recognizer, patch('app.sd.RawInputStream', side_effect=stream):
            recognizer.return_value.AcceptWaveform.return_value = True
            recognizer.return_value.Result.return_value = '{"text": "nova"}'
            n.wake_word_listener()
            model.assert_called_once_with(app.VOSK_MODEL_PATH)
            recognizer.assert_called_once_with(model.return_value, app.SAMPLE_RATE)
        self.assertTrue(n.wake_stop_event.is_set())
        self.assertEqual(n.ui_queue.get_nowait()[1], n.on_wake_word_detected)

    def test_worker_ui_updates_are_queued_without_tk_calls(self):
        n = self.nova
        n.after = Mock(side_effect=AssertionError('Tk used from worker'))
        callback = Mock()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(n.post_ui, 0, callback).result(timeout=2)
        callback.assert_not_called()
        self.flush()
        callback.assert_called_once()


@unittest.skipUnless(HAS_RUNTIME, 'Use the project virtual environment')
class StartupTests(unittest.TestCase):
    def test_real_window_and_floating_widget_start_and_close(self):
        n = app.NovaApp(db_path=':memory:', start_listener=False)
        try:
            n.widget.withdraw()
            n.update()
            self.assertTrue(n.widget.winfo_exists())
            self.assertTrue(n.message_entry.winfo_exists())
            self.assertEqual(n.handle_command('show reminders'), 'You have no active reminders.')
        finally:
            n.close_app()


if __name__ == '__main__':
    unittest.main()
