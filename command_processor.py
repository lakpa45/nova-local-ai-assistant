"""Shared voice/text pipeline: classify, extract, validate, dispatch."""
from datetime import datetime, timedelta
import threading
import time
import webbrowser
from urllib.parse import quote_plus
import config
from action_registry import Action, extend_registry
from command_parser import extract, phrase, unsafe_command
from command_teaching import CommandTeacher
from topic_learning import TopicLearning
from intent_classifier import IntentClassifier
from nova_actions import parse_command, CONFIRM_PHRASES, CANCEL_PHRASES

WEBSITES = {'youtube': 'https://www.youtube.com', 'google': 'https://www.google.com',
            'wikipedia': 'https://www.wikipedia.org', 'browser': 'https://www.google.com'}


class CommandProcessor:
    def __init__(self, app, knowledge, classifier=None):
        self.app, self.knowledge = app, knowledge
        self.executor = app.actions
        self.classifier = classifier if classifier is not None else IntentClassifier()
        self.lock = threading.RLock()
        self.awaiting_target = None
        self.topic_learning = TopicLearning(self.executor, knowledge)
        self.teacher = CommandTeacher(self.executor, knowledge)
        extend_registry(self.executor, {
            'open_website': self.open_website,
            'set_reminder': self.set_reminder,
            'get_time': lambda _: app.handle_builtin_command('time'),
            'get_weather': self.get_weather,
            'remember_information': lambda data: app.handle_builtin_command('remember that ' + data['content']),
            'recall_information': self.recall,
            'list_learned_topics': lambda _: ', '.join(knowledge.topics()) or 'No learned topics yet.',
            'retrain_model': self.retrain,
        })

    def handle(self, message):
        with self.lock:
            return self._handle(message)

    def _handle(self, message):
        with self.lock:
            # 1. Check for Routines first
            normalized = phrase(message)
            try:
                with open('routines.json', 'r') as f:
                    routines = json.load(f)
                for routine_name, actions_data in routines.items():
                    if routine_name.replace('_', ' ') in normalized:
                        # Convert raw data to Action objects
                        action_list = [Action(a['action'], a['target']) for a in actions_data]
                        return self.executor.execute_routine(action_list)
            except Exception as e:
                print(f"Routine error: {e}")

            # Confirmation can never be inferred by the neural network.
            if self.executor.pending is not None or self.executor.expired:
                return self.executor.handle(message)
        normalized = phrase(message)
        if self.topic_learning.state:
            return self.executor.execute(Action('learn_topic', {'reply': message}))
        if self.teacher.state:
            return self.teacher.handle(message)
        if self.awaiting_target:
            intent, deadline = self.awaiting_target
            if normalized in CANCEL_PHRASES:
                self.awaiting_target = None
                return 'Action cancelled.'
            if time.monotonic() >= deadline:
                self.awaiting_target = None
                return 'The command expired. Please give it again.'
            if unsafe_command(message) or not message.strip():
                return 'Give only an application name, or say cancel.'
            self.awaiting_target = None
            return self.executor.execute(Action(intent, message.strip()))
        if normalized in CONFIRM_PHRASES | CANCEL_PHRASES:
            return self.executor.handle(message)
        if normalized == 'retrain command model':
            return self.executor.execute(Action('retrain_model', 'confirmed command examples'))
        if unsafe_command(message):
            return 'Raw programs and shell commands are not supported.'
        prediction = self.classifier.predict(message)
        # Preserve deterministic legacy commands, including personal memory deletion.
        legacy = parse_command(message)
        if legacy and legacy.action in {'save_memory', 'forget_memory', 'show_memories',
                                        'create_reminder', 'show_reminders', 'show_time',
                                        'open_folder', 'open_camera'}:
            return self.executor.execute(legacy)
        if legacy and legacy.action not in self.executor.handlers and '_' in legacy.action:
            return self.executor.execute(legacy)
        intent = prediction['intent']
        if intent in {'unknown', 'confirm_action', 'reject_action'}:
            if config.ENABLE_LOCAL_LLM:
                return None
            return self.teacher.begin(message)
        parsed = extract(message, intent)
        if parsed is None:
            if intent in {'open_app', 'close_app'}:
                self.awaiting_target = (intent, time.monotonic() + 30)
                return f'Which application should I {"open" if intent == "open_app" else "close"}?'
            return (f'I recognized {intent}, but could not safely extract its values. '
                    'Please use a complete command with an application, topic, query, or time.')
        values = parsed.values
        if intent in {'open_app', 'close_app'}:
            return self.executor.execute(Action(intent, values['target']))
        if intent == 'search_web':
            target = values['engine'] + ' for ' + values['query']
            return self.executor.execute(Action(intent, target))
        if intent == 'forget_topic':
            return self.topic_learning.request_forget(values['topic'])
        return self.executor.execute(Action(intent, values))

    def open_website(self, data):
        target = data['target'] if isinstance(data, dict) else data
        url = WEBSITES.get(target.lower())
        if not url:
            return f'I am afraid I cannot locate the website for {target}, sir. Approved systems include: ' + ', '.join(WEBSITES)
        webbrowser.open(url)
        return f'Of course. Accessing {target} for you now, sir.'

    def set_reminder(self, data):
        now = datetime.now()
        try:
            if 'delay' in data:
                if not 1 <= data['delay'] <= 100000:
                    return 'I suggest a reminder delay between 1 and 100,000 units, sir.'
                delta = {'hours' if 'hour' in data['unit'] else 'minutes': data['delay']}
                when = now + timedelta(**delta)
            else:
                text = data['time'].replace(' ', '')
                clock = datetime.strptime(text, '%I:%M%p' if ':' in text else '%I%p')
                when = now.replace(hour=clock.hour, minute=clock.minute, second=0, microsecond=0)
                if when <= now:
                    when += timedelta(days=1)
        except (ValueError, OverflowError):
            return 'I apologize, but the time format was unrecognized. Perhaps try "8 PM" or "8:30 PM", sir?'
        self.app.save_reminder(data['reminder'], when)
        return f'Consider it done, sir. I shall remind you to {data["reminder"]} at {when:%I:%M %p on %d %b}.'

    def get_weather(self, data):
        location = data['location'] or 'my location'
        webbrowser.open('https://www.google.com/search?q=' + quote_plus('weather in ' + location))
        return f'Retrieving the current weather for {location} now, sir.'

    def recall(self, data):
        if data.get('personal'):
            return self.app.handle_builtin_command('show my memories')
        rows = self.knowledge.search(data['topic'])
        if not rows:
            return f'I have no recorded information regarding {data["topic"]} in my database, sir.'
        return '\n\n'.join(f'{topic}: {content}\nSource: {source or "unspecified"}'
                            for _, topic, content, source in rows)

    def retrain(self, _):
        from train_model import train
        result = train(db_path=self.app.database_path)
        self.classifier.reload()
        return f'Command model retrained using {result["examples"]} examples.'
