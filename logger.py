#!/usr/bin/python
# utf-8
from psutil import cpu_percent, virtual_memory, swap_memory
from os import cpu_count
from datetime import datetime
from platform import system, release
from collections.abc import Iterable
from collections import defaultdict

history = defaultdict(lambda: datetime.now())

def startup():
    log({'total VIRT': '{:.0f} MiB'.format(virtual_memory().total / 2**20),
         'total SWAP': '{:.0f} MiB'.format(swap_memory().total / 2**20),
         'threads': cpu_count(),
         'system': '{} {}'.format(system(), release())
    }, startup)
def log(message = "", method = None):
    if isinstance(message, str):
        message = message.split('\n')
        loglines(message, method)
    elif isinstance(message, dict):
        for pair in message.items():
            logline('{}: {}'.format(pair[0], pair[1]), method)
    elif isinstance(message, Iterable):
        for m in message:
            log(m, method)
    else:
        logline(str(message), method)
def logline(message = "", method = None):
    global oldprint
    oldprint = history[method]
    newprint = datetime.now()
    print('{} [+{}] VIRT: {:>6.0f} MiB; SWAP: {:>7.0f} MiB; CPU: {:>5.1f} %{}: {}'.format(datetime.now(), newprint - oldprint, virtual_memory().used / 2**20, swap_memory().used / 2**20, cpu_percent(), '; {}'.format(method.__name__) if method is not None else "", message))
    history[method] = datetime.now()
def loglines(messages = [], method = None):
    f = True
    for m in filter(lambda s: s is not None and s != "", messages):
        f = False
        logline(m, method)
    if f:
        logline("", method)
startup()
