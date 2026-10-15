"""The only bridge from an intent to an existing Python action."""
from nova_actions import Action, ActionExecutor

SUPPORTED_INTENTS = frozenset({
    'open_app', 'close_app', 'open_website', 'search_web', 'set_reminder',
    'get_time', 'get_weather', 'remember_information', 'recall_information',
    'learn_topic', 'forget_topic', 'list_learned_topics', 'confirm_action',
    'reject_action', 'unknown',
})
TEACHABLE_INTENTS = SUPPORTED_INTENTS - {'confirm_action', 'reject_action', 'unknown'}


def extend_registry(executor, handlers, prompts=None):
    """Add explicit callables only. Neither database rows nor model text register code."""
    allowed = SUPPORTED_INTENTS | {'save_topic', 'save_command_example', 'retrain_model'}
    for name, handler in handlers.items():
        if name not in allowed or not callable(handler):
            raise ValueError(f'Unregistered intent: {name}')
    executor.handlers.update(handlers)
    executor.prompts.update(prompts or {})
    automatic = executor.policy.setdefault('auto_approve', [])
    sensitive = {'close_app', 'forget_topic', 'save_topic', 'save_command_example', 'retrain_model'}
    automatic.extend(name for name in handlers if name not in sensitive and name not in automatic)
    executor.policy.setdefault('confirmation_required', []).extend(sorted(sensitive))
    return executor
