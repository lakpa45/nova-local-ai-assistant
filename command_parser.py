"""Extract values from a predicted intent without inventing executable actions."""
from dataclasses import dataclass, field
import re


@dataclass
class ParsedCommand:
    intent: str
    values: dict = field(default_factory=dict)


def phrase(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


def clean(text):
    text = re.sub(r'^nova[,\s]+', '', text.strip(), flags=re.I)
    text = re.sub(r'^(?:(?:please|can you|could you|would you|would you mind|i would like you to|be a dear and)\s+)+', '', text, flags=re.I)
    return text.rstrip(' .?!').strip()


def match(pattern, text):
    return re.fullmatch(pattern, text, re.I)


def extract(message, intent):
    text = clean(message)
    values = {}
    if intent in {'open_app', 'close_app', 'open_website'}:
        verbs = {'open_app': r'open|launch|start|run',
                 'close_app': r'close|quit|exit|stop|shut down',
                 'open_website': r'open|visit|go to|show'}[intent]
        found = match(rf'(?:{verbs})\s+(?:the\s+)?(.+)', text)
        if not found:
            return None
        target = re.sub(r'\s+(?:application|app|website)$', '', found[1], flags=re.I).strip()
        if not target:
            return None
        values['target'] = target
    elif intent == 'search_web':
        found = match(r'(?:search(?:\s+(Google|YouTube|the web))?(?:\s+for)?|google|look up|find information on)\s+(.+)', text)
        if not found:
            return None
        values = {'engine': (found[1] or 'google').lower(), 'query': found[2]}
    elif intent == 'set_reminder':
        relative = match(r'remind me in (\d+) (minutes?|hours?) to (.+)', text)
        absolute = match(r'(?:remind me|set a reminder) to (.+) at (\d{1,2}(?::\d{2})?\s*(?:AM|PM))', text)
        if relative:
            values = {'reminder': relative[3], 'delay': int(relative[1]), 'unit': relative[2]}
        elif absolute:
            values = {'reminder': absolute[1], 'time': absolute[2].upper()}
        else:
            return None
    elif intent == 'learn_topic':
        inline = match(r'learn this:\s*(.+)', text)
        from_file = match(r'learn from file(?:\s+(.+))?', text)
        topic = match(r'(?:learn (?:about|topic)|save this information about|i want to teach you about)\s+(.+)', text)
        if inline:
            values = {'content': inline[1], 'mode': 'dictate'}
        elif from_file:
            values = {'mode': 'file', 'path': from_file[1] or ''}
        elif topic:
            values = {'topic': topic[1]}
        elif text.lower() not in {'learn this topic', 'learn this information', 'learn this'}:
            return None
    elif intent == 'recall_information':
        if phrase(text) in {'show my memories', 'what do you remember about me'}:
            values = {'personal': True}
        else:
            found = match(r'(?:what do you know about|tell me about|recall(?: the topic)?|show information about|what have you learned about)\s+(.+)', text)
            if not found:
                return None
            values = {'topic': found[1]}
    elif intent == 'remember_information':
        found = match(r'(?:remember(?: that| this fact)?|save a memory that|save this personal memory|make a note that)\s+(.+)', text)
        if not found:
            return None
        values = {'content': found[1]}
    elif intent == 'forget_topic':
        found = match(r'(?:forget(?: topic| about)?|delete (?:information about|topic)|remove the topic)\s+(.+)', text)
        if not found:
            return None
        values = {'topic': found[1]}
    elif intent == 'get_weather':
        found = re.search(r'\b(?:in|for)\s+(.+)', text, re.I)
        values = {'location': found[1] if found else ''}
    elif intent not in {'get_time', 'list_learned_topics', 'confirm_action', 'reject_action'}:
        return None
    return ParsedCommand(intent, values)


def unsafe_command(text):
    """Reject program text as a taught command (knowledge content stays inert)."""
    return bool(re.search(r'[;`|\n\r]|&&|\$\(|\b(?:powershell|cmd\.exe|taskkill|'
                          r'os\.system|subprocess|eval\s*\(|exec\s*\(|import\s+\w+|'
                          r'python\s+-c|del\s+/|remove-item)\b', text, re.I))
