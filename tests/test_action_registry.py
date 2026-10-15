import unittest
from unittest.mock import Mock
from action_registry import Action, ActionExecutor, extend_registry


class RegistryTests(unittest.TestCase):
    def test_unregistered_action_is_never_executed(self):
        executor = ActionExecutor({}, {})
        self.assertIn('rejected', executor.execute(Action('powershell', 'anything')))
        with self.assertRaises(ValueError):
            extend_registry(executor, {'run_program': Mock()})

    def test_sensitive_operations_always_require_confirmation(self):
        for name in ('close_app', 'forget_topic', 'save_topic', 'save_command_example', 'retrain_model'):
            handler = Mock(return_value='done')
            executor = extend_registry(ActionExecutor({}, {}), {name: handler})
            executor.execute(Action(name, 'data'))
            handler.assert_not_called()
            self.assertEqual(executor.handle('Yes, confirm'), 'done')
            handler.assert_called_once()

    def test_cancel_punctuation(self):
        handler = Mock()
        executor = extend_registry(ActionExecutor({}, {}), {'forget_topic': handler})
        executor.execute(Action('forget_topic', 'Python'))
        self.assertEqual(executor.handle('No, cancel'), 'Action cancelled.')
        handler.assert_not_called()
