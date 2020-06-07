#!/usr/bin/python
# utf-8
from psutil import cpu_percent, virtual_memory, swap_memory
from os import cpu_count, environ
from datetime import datetime, timedelta
from platform import system, release
from collections.abc import Iterable
from collections import defaultdict
from threading import Timer

history = defaultdict(lambda: datetime.now())
settings = {
    'startup': True,
    'time': True,
    'stopwatch': True,
    'virt': True,
    'swap': True,
    'cpu': True,
    'methodmark': True
}

def initsettings():
    def bool(s):
        return s.upper() in ['TRUE', '1', 'T', 'Y', 'YES', 'YEAH', 'YUP', 'CERTAINLY', 'UH-HUH']
    for k in settings.keys():
        env = environ.get('FP_LOG_{}'.format(k.upper()))
        if env is not None:
            settings[k] = bool(env)

def startup():
    initsettings()
    if settings['startup']:
        log({'total VIRT': '{:.0f} MiB'.format(virtual_memory().total / 2**20),
             'total SWAP': '{:.0f} MiB'.format(swap_memory().total / 2**20),
             'threads': cpu_count(),
             'system': '{} {}'.format(system(), release())
        }, startup)
def logp(message = '', methodmark = None, delay = 0):
    if isinstance(delay, timedelta):
        delay = delay.total_seconds()
    Timer(delay, lambda: log(message, methodmark)).start()
def log(message = '', methodmark = None):
    if isinstance(message, str):
        message = message.split('\n')
        loglines(message, methodmark)
    elif isinstance(message, dict):
        for pair in message.items():
            logline('{}: {}'.format(pair[0], pair[1]), methodmark)
    elif isinstance(message, Iterable):
        for m in message:
            log(m, methodmark)
    else:
        logline(str(message), methodmark)
def logline(message = '', methodmark = None):
    oldprint = history[methodmark]
    newprint = datetime.now()
    results = []
    if settings['time']:
        results += [str(datetime.now())]
    if settings['stopwatch']:
        results += ['[+{}]'.format(newprint - oldprint)]
    if settings['virt']:
        results += ['VIRT: {:>6.0f} MiB'.format(virtual_memory().used / 2**20)]
    if settings['swap']:
        results += ['SWAP: {:>7.0f} MiB'.format(swap_memory().used / 2**20)]
    if settings['cpu']:
        results += ['CPU: {:>5.1f} %'.format(cpu_percent())]
    if settings['methodmark'] and methodmark is not None:
        def resolver(methodmark):
            if hasattr(methodmark, '__name__'):
                if hasattr(methodmark, '__module__'):
                    return '#'.join((methodmark.__module__, methodmark.__name__))
                return methodmark.__name__
            elif isinstance(methodmark, Iterable):
                meth = ()
                for m in methodmark:
                    meth += (resolver(m),)
                return meth
            else:
                return methodmark
        results += [str(resolver(methodmark))]
    output = '; '.join(results)
    if message is not None and len(message) > 0:
        if len(output) > 0:
            output = '{}: {}'.format(output, message)
        else:
            output = message
    print(output)
    history[methodmark] = datetime.now()
def loglines(messages = [], methodmark = None):
    f = True
    for m in filter(lambda s: s is not None and s != '', messages):
        f = False
        logline(m, methodmark)
    if f:
        logline('', methodmark)
startup()
