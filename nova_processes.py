"""Allowlisted application closing. No user text becomes a command or path."""
import os
import psutil

PROCESS_ALIASES = {
    'whatsapp': ['whatsapp'],
    'spotify': ['spotify'],
    'discord': ['discord'],
    'chrome': ['chrome'],
    'google chrome': ['chrome'],
    'edge': ['msedge'],
    'microsoft edge': ['msedge'],
    'visual studio code': ['code'],
    'notepad': ['notepad'],
    'calculator': ['calculatorapp', 'calculator'],
}

PROTECTED_PROCESSES = {
    'system', 'registry', 'smss', 'csrss', 'wininit', 'winlogon',
    'services', 'lsass', 'svchost', 'explorer',
}
ASSISTANT_PROCESSES = {
    'nova', 'python', 'pythonw', 'py', 'ollama', 'ollama app',
    'ollama_llama_server', 'llama-server', 'whisper', 'vosk',
    'wake_word_listener', 'wake_word_test', 'ffmpeg', 'ffplay',
}


def process_name(name):
    return name.casefold().removesuffix('.exe')


def is_protected(name):
    return (name in PROTECTED_PROCESSES | ASSISTANT_PROCESSES
            or name.startswith(('python', 'ollama', 'nova', 'vosk', 'whisper', 'wake_word')))


def close_application(target):
    """Called only by the confirmed action handler; never escalate or force-kill."""
    key = ' '.join(target.casefold().split())
    if is_protected(process_name(key)):
        return f'Cannot close protected application: {target}.'
    aliases = PROCESS_ALIASES.get(key)
    if not aliases:
        return f'Unknown application: {target}.'
    names = {process_name(name) for name in aliases}
    if any(is_protected(name) for name in names):
        return f'Cannot close protected application: {target}.'

    try:
        # Protect NOVA's runtime tree and launching hosts, even if renamed.
        own = psutil.Process(os.getpid())
        protected_pids = {os.getpid(), 0, 4}
        protected_pids.update(p.pid for p in own.parents())
        protected_pids.update(p.pid for p in own.children(recursive=True))
        matches = []
        for process in psutil.process_iter(['name']):
            try:
                name = process_name(process.info.get('name') or '')
                if name not in names:
                    continue
                if process.pid in protected_pids or is_protected(name):
                    return f'Cannot close protected application: {target}.'
                matches.append(process)
            except psutil.NoSuchProcess:
                continue
    except psutil.AccessDenied:
        return f'Permission denied while closing {target}.'
    except psutil.Error:
        return f'Could not safely inspect processes for {target}.'

    if not matches:
        return f'{target} is not currently running.'
    terminated = []
    denied = False
    for process in matches:
        try:
            # Check the current executable name again before terminating.
            if process_name(process.name()) not in names:
                continue
            process.terminate()
            terminated.append(process)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            denied = True
        except psutil.Error:
            return f'Could not close all processes for {target}.'
    if denied:
        suffix = ' Some matching processes were already closed.' if terminated else ''
        return f'Permission denied while closing {target}.{suffix}'
    if terminated:
        try:
            _, alive = psutil.wait_procs(terminated, timeout=3)
        except psutil.Error:
            return f'Could not verify whether {target} closed.'
        if alive:
            return f'{target} has not finished closing. No force kill was attempted.'
        return f'Closed {target}.'
    return f'{target} is not currently running.'
