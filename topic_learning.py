"""Topic-learning dialogue and bounded, non-executable text ingestion."""
from pathlib import Path
import re
import time
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
import config
from action_registry import Action, extend_registry
from command_parser import phrase
from nova_actions import CANCEL_PHRASES


def readable_text(raw):
    if len(raw) > config.MAX_LEARNING_BYTES:
        raise ValueError('The text exceeds the 256 KB limit.')
    text = raw.decode('utf-8-sig')
    if '\x00' in text or not text.strip():
        raise ValueError('The source must contain readable, nonempty UTF-8 text.')
    return text.strip()


def read_text_file(filename):
    raw_path = filename.strip().strip('"')
    if raw_path.startswith(('\\\\', '//')):
        raise ValueError('Use a local text file, not a network path.')
    path = Path(raw_path).expanduser().resolve(strict=True)
    if str(path).startswith(('\\\\', '//')):
        raise ValueError('Use a local text file, not a network path.')
    if not path.is_file() or path.suffix.lower() not in config.APPROVED_TEXT_EXTENSIONS:
        raise ValueError('Only local .txt, .md, and .csv files are approved.')
    if ':' in path.name or path.stat().st_size > config.MAX_LEARNING_BYTES:
        raise ValueError('Invalid file path or file exceeds the 256 KB limit.')
    with path.open('rb') as stream:
        content = readable_text(stream.read(config.MAX_LEARNING_BYTES + 1))
    return content, str(path)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('Redirected sources are not approved.')


def read_approved_source(name):
    url = config.APPROVED_ONLINE_SOURCES.get(name)
    if not url or urlparse(url).scheme != 'https':
        raise ValueError('Choose a configured approved HTTPS source.')
    request = Request(url, headers={'User-Agent': 'NOVA/1.0'})
    with build_opener(NoRedirect()).open(request, timeout=10) as response:
        if response.headers.get_content_type() not in {'text/plain', 'text/markdown', 'text/csv'}:
            raise ValueError('The approved source must serve plain text, Markdown, or CSV.')
        return readable_text(response.read(config.MAX_LEARNING_BYTES + 1)), url


class TopicLearning:
    def __init__(self, executor, knowledge, clock=time.monotonic):
        self.executor, self.knowledge, self.clock = executor, knowledge, clock
        self.state = None
        extend_registry(executor, {'learn_topic': self.begin, 'save_topic': self.save,
                                  'forget_topic': self.forget}, {
            'save_topic': lambda data: f'Store topic "{data["topic"]}" from {data["source"]}:\n'
                                      f'{data["content"]}\nSay yes confirm or no cancel.',
            'forget_topic': lambda data: 'Remove these stored records?\n' + '\n'.join(
                f'{row[1]}: {row[2]}' for row in data['rows']) + '\nSay yes confirm or no cancel.'})

    def begin(self, data):
        if 'reply' in data:
            return self.handle(data['reply'])
        self.state = {**data, 'deadline': self.clock() + 120, 'source': 'dictation'}
        if data.get('path'):
            try:
                content, source = read_text_file(data['path'])
                self.state.update(content=content, source=source)
            except (ValueError, OSError, UnicodeError) as error:
                self.state = None
                return str(error)
        if not self.state.get('topic'):
            self.state['stage'] = 'topic'
            return 'What topic name should I use?'
        return self.choose_method()

    def choose_method(self):
        self.state['stage'] = 'method'
        return 'Would you like to dictate information, provide a file, or use an approved online source? Say dictate, file, or online.'

    def handle(self, text):
        state = self.state
        if not state:
            return 'Say learn topic followed by the topic name.'
        if self.clock() >= state['deadline']:
            self.state = None
            return 'The learning session expired. Please start again.'
        reply = phrase(text)
        if reply in CANCEL_PHRASES:
            self.state = None
            return 'Action cancelled.'
        try:
            stage = state['stage']
            if stage == 'topic':
                if not text.strip() or len(text) > 200:
                    return 'Give a topic name of 1 to 200 characters.'
                state['topic'] = text.strip()
                if state.get('content'):
                    return self.preview()
                if state.get('mode') == 'file':
                    state['stage'] = 'file'
                    return 'Give the path to a local .txt, .md, or .csv file (maximum 256 KB).'
                return self.choose_method()
            if stage == 'method':
                if reply in {'dictate', 'dictation'}:
                    state['stage'] = 'content'
                    return f'What should I store about {state["topic"]}?'
                if reply == 'file':
                    state['stage'] = 'file'
                    return 'Give the path to a local .txt, .md, or .csv file (maximum 256 KB).'
                if reply in {'online', 'approved online source'}:
                    if not config.APPROVED_ONLINE_SOURCES:
                        return 'No online sources are approved yet. Choose dictate or file.'
                    state['stage'] = 'online'
                    return 'Choose the approved source to fetch: ' + ', '.join(config.APPROVED_ONLINE_SOURCES)
                return 'Choose dictate, file, or online, or say cancel.'
            if stage == 'content':
                state['content'] = readable_text(text.encode('utf-8'))
            elif stage == 'file':
                state['content'], state['source'] = read_text_file(text)
            elif stage == 'online':
                state['content'], state['source'] = read_approved_source(text.strip())
            return self.preview()
        except (ValueError, OSError, UnicodeError) as error:
            self.state = None
            return f'Could not learn that source: {error}'

    def preview(self):
        data = {key: self.state[key] for key in ('topic', 'content', 'source')}
        self.state = None
        return self.executor.execute(Action('save_topic', data))

    def save(self, data):
        self.knowledge.save(data['topic'], data['content'], data['source'])
        return f'Saved information about {data["topic"]}.'

    def request_forget(self, topic):
        rows = self.knowledge.exact_topic(topic)
        if not rows:
            return f'I have no stored topic named {topic}.'
        return self.executor.execute(Action('forget_topic', {'rows': rows}))

    def forget(self, data):
        count = self.knowledge.forget_snapshot(data['rows'])
        return f'Removed {count} stored record(s).'
