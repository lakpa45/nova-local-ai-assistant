"""Dialogue for saving explicitly confirmed labels, never new action code."""
import time
from action_registry import TEACHABLE_INTENTS, Action, extend_registry
from command_parser import phrase, unsafe_command
from nova_actions import CONFIRM_PHRASES, CANCEL_PHRASES
import config


class CommandTeacher:
    def __init__(self, executor, knowledge, clock=time.monotonic):
        self.executor, self.knowledge, self.clock = executor, knowledge, clock
        self.state = None
        extend_registry(executor, {'save_command_example': self.save}, {
            'save_command_example': lambda data: (
                f'Learn command "{data["phrase"]}" as {data["intent"]}? '
                'Say yes confirm or no cancel.')})

    def begin(self, text):
        if unsafe_command(text) or len(text) > 1000:
            return 'I cannot learn raw programs or shell commands.'
        self.state = {'phrase': text, 'stage': 'offer', 'deadline': self.clock() + 120}
        return (config.COMMAND_ONLY_RESPONSE + '\nI do not understand that command. '
                'Would you like to teach it? Say yes confirm or no cancel.')

    def handle(self, text):
        state = self.state
        if not state:
            return None
        if self.clock() >= state['deadline']:
            self.state = None
            return 'The teaching session expired. Please give the command again.'
        reply = phrase(text)
        if reply in CANCEL_PHRASES:
            self.state = None
            return 'Action cancelled.'
        if state['stage'] == 'offer':
            if reply not in CONFIRM_PHRASES | {'yes'}:
                return 'Say yes confirm to teach this command or no cancel to stop.'
            state['stage'] = 'intent'
            return 'Which existing intent does it represent? Choose: ' + ', '.join(sorted(TEACHABLE_INTENTS))
        intent = text.lower().strip().replace(' ', '_')
        if intent not in TEACHABLE_INTENTS or intent not in self.executor.handlers:
            return 'Choose an existing intent from the list, or say cancel.'
        data = {'phrase': state['phrase'], 'intent': intent}
        self.state = None
        return self.executor.execute(Action('save_command_example', data))

    def save(self, data):
        self.knowledge.save_example(data['phrase'], data['intent'], confirmed=True)
        return ('Command example saved. Add at least two examples for this intent, '
                'then say retrain command model when ready.')
