#!/usr/bin/python
# utf-8
from psutil import cpu_percent, virtual_memory, swap_memory
from os import cpu_count
from datetime import datetime
from platform import system, release

oldprint = datetime.now()

def startup():
    log('startup: total VIRT: {:>6.0f} MiB; total SWAP: {:>7.0f} MiB; threads: {}; system: {} {}'.format(virtual_memory().total / 2**20, swap_memory().total / 2**20, cpu_count(), system(), release()))
def log(message = ""):
    global oldprint
    newprint = datetime.now()
    print('{} [+{}] VIRT: {:>6.0f} MiB; SWAP: {:>7.0f} MiB; CPU: {:>5.1f} %: {}'.format(datetime.now(), newprint - oldprint, virtual_memory().used / 2**20, swap_memory().used / 2**20, cpu_percent(), message))
    oldprint = datetime.now()
startup()
