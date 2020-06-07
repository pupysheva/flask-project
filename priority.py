#!/usr/bin/python
# utf-8


def priority(i):
    """ Set the priority of the process to below-normal."""

    import sys
    try:
        sys.getwindowsversion()
    except AttributeError:
        isWindows = False
    else:
        isWindows = True
    try:
        if isWindows:
            # Based on:
            #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
            #   http://code.activestate.com/recipes/496767/
            import win32api,win32process,win32con
    
            pid = win32api.GetCurrentProcessId()
            handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
            win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS if i >= 0 else win32process.ABOVE_NORMAL_PRIORITY_CLASS)
        else:
            import os
            os.nice(i)
            pass
    except PermissionError as e:
        from logger import log
        log('Ignore nice: {}'.format(e), priority)


def lowpriority():
    priority(15)


def hightpriority():
    priority(-15)
