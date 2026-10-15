import ast
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import psutil
import nova_processes as processes
from nova_actions import Action, ActionExecutor, parse_command

ROOT = Path(__file__).resolve().parents[1]


class CloseApplicationTests(unittest.TestCase):
    def setUp(self):
        self.clock = Mock(return_value=100.0)
        self.executor = ActionExecutor(
            json.loads((ROOT / 'permissions.json').read_text()),
            {'close_app': processes.close_application, 'open_app': Mock()},
            clock=self.clock,
        )
        # Mock every process API: these tests cannot terminate real applications.
        self.enumerate = self.enterContext(patch('nova_processes.psutil.process_iter', return_value=[]))
        self.own = self.enterContext(patch('nova_processes.psutil.Process')).return_value
        self.own.parents.return_value = []
        self.own.children.return_value = []
        self.wait = self.enterContext(patch('nova_processes.psutil.wait_procs', return_value=([], [])))

    def process(self, name='WhatsApp.exe', pid=987654):
        p = Mock(pid=pid, info={'name': name})
        p.name.return_value = name
        self.enumerate.return_value = [p]
        return p

    def confirmed(self, command='Close WhatsApp'):
        self.executor.handle(command)
        result = self.executor.handle('yes confirm')
        self.assertIsNone(self.executor.pending)
        return result

    def test_parse_close_and_quit_with_complete_targets(self):
        self.assertEqual(parse_command('Close WhatsApp'), Action('close_app', 'WhatsApp'))
        self.assertEqual(parse_command('Quit Visual Studio Code'), Action('close_app', 'Visual Studio Code'))
        for target in ('Spotify', 'Discord', 'Google Chrome', 'Visual Studio Code'):
            self.assertEqual(parse_command(f'close {target}'), Action('close_app', target))

    def test_pending_action_and_warning_before_any_process_access(self):
        self.assertEqual(self.executor.handle('Close WhatsApp'),
                         'Closing WhatsApp could lose unsaved work. Say yes confirm or no cancel.')
        self.assertEqual(self.executor.pending, {'action': 'close_app', 'target': 'WhatsApp'})
        self.enumerate.assert_not_called()
        self.own.parents.assert_not_called()

    def test_all_explicit_confirmation_phrases(self):
        for phrase in ('yes confirm', 'confirm', 'continue', 'YES CONFIRM'):
            p = self.process()
            self.executor.handle('Close WhatsApp')
            self.assertEqual(self.executor.handle(phrase), 'Closed WhatsApp.')
            self.assertIsNone(self.executor.pending)
            p.terminate.assert_called_once_with()
            p.kill.assert_not_called()

    def test_cancellation_phrases_clear_pending(self):
        for phrase in ('no cancel', 'cancel', 'stop'):
            self.executor.handle('Close WhatsApp')
            self.assertEqual(self.executor.handle(phrase), 'Action cancelled.')
            self.assertIsNone(self.executor.pending)
        self.enumerate.assert_not_called()

    def test_unrelated_commands_and_ambiguous_yes_are_not_executed(self):
        self.executor.handle('Close WhatsApp')
        for command in ('yes', 'no', 'Open Spotify', 'hello', 'Close Discord'):
            self.assertIn('confirm or cancel', self.executor.handle(command))
            self.assertEqual(self.executor.pending['target'], 'WhatsApp')
        self.executor.handlers['open_app'].assert_not_called()
        self.enumerate.assert_not_called()

    def test_timeout_clears_pending_and_consumes_late_confirmation(self):
        self.executor.handle('Close WhatsApp')
        self.clock.return_value = 130.0
        self.assertIsNone(self.executor.pending)
        self.assertEqual(self.executor.handle('yes confirm'),
                         'The confirmation expired. Please give the command again.')
        self.assertIn('no pending action', self.executor.handle('confirm'))
        self.enumerate.assert_not_called()

    def test_unknown_application_is_rejected(self):
        self.assertEqual(self.confirmed('Close Unknown App'), 'Unknown application: Unknown App.')
        self.enumerate.assert_not_called()

    def test_not_running(self):
        self.assertEqual(self.confirmed(), 'WhatsApp is not currently running.')

    def test_protected_processes_and_dependencies_are_rejected(self):
        for name in processes.PROTECTED_PROCESSES | processes.ASSISTANT_PROCESSES | {'python3.14', 'ollama runner'}:
            self.assertIn('protected', self.confirmed(f'Close {name}'))
        self.enumerate.assert_not_called()

    def test_alias_cannot_bypass_protection(self):
        with patch.dict(processes.PROCESS_ALIASES, {'whatsapp': ['svchost']}):
            self.assertIn('protected', self.confirmed())
        self.enumerate.assert_not_called()

    def test_nova_self_pid_is_rejected_even_if_renamed(self):
        p = self.process(pid=os.getpid())
        self.assertIn('protected', self.confirmed())
        p.terminate.assert_not_called()

    def test_assistant_children_and_hosts_are_protected(self):
        p = self.process()
        for family in (self.own.parents, self.own.children):
            family.return_value = [p]
            self.assertIn('protected', self.confirmed())
            family.return_value = []
        p.terminate.assert_not_called()

    def test_access_denied_clears_pending(self):
        p = self.process()
        p.terminate.side_effect = psutil.AccessDenied(p.pid)
        self.assertEqual(self.confirmed(), 'Permission denied while closing WhatsApp.')

    def test_discovery_access_denied_fails_closed(self):
        self.enumerate.side_effect = psutil.AccessDenied(123)
        self.assertEqual(self.confirmed(), 'Permission denied while closing WhatsApp.')

    def test_only_exact_application_processes_are_terminated(self):
        a = self.process('chrome.exe', 10)
        b = self.process('Chrome.EXE', 11)
        other = self.process('chrome_helper.exe', 12)
        discord = self.process('discord.exe', 13)
        self.enumerate.return_value = [a, b, other, discord]
        self.assertEqual(self.confirmed('Close Google Chrome'), 'Closed Google Chrome.')
        for p in (a, b):
            p.terminate.assert_called_once()
        for p in (other, discord):
            p.terminate.assert_not_called()

    def test_exit_race_is_handled(self):
        p = self.process()
        p.terminate.side_effect = psutil.NoSuchProcess(p.pid)
        self.assertEqual(self.confirmed(), 'WhatsApp is not currently running.')

    def test_timeout_does_not_force_kill_or_claim_success(self):
        p = self.process()
        self.wait.return_value = ([], [p])
        self.assertIn('has not finished closing', self.confirmed())
        p.kill.assert_not_called()

    def test_handler_failure_clears_pending(self):
        self.executor.handlers['close_app'] = Mock(side_effect=RuntimeError('failure'))
        self.executor.handle('Close WhatsApp')
        with self.assertRaises(RuntimeError):
            self.executor.handle('confirm')
        self.assertIsNone(self.executor.pending)

    def test_confirmation_is_single_use(self):
        p = self.process()
        self.confirmed()
        self.executor.handle('confirm')
        p.terminate.assert_called_once()

    def test_raw_input_cannot_reach_shell_or_process_api(self):
        self.assertIn('Unknown application', self.confirmed('Close WhatsApp; taskkill /F /IM python.exe'))
        self.enumerate.assert_not_called()
        tree = ast.parse((ROOT / 'nova_processes.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self.assertNotIn(ast.unparse(node.func), ('eval', 'exec', 'os.system', 'subprocess.run', 'subprocess.Popen'))
                self.assertFalse(any(k.arg == 'shell' for k in node.keywords))


if __name__ == '__main__':
    unittest.main()
