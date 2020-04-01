from my_module import RecommendationAlgoritm
import struct

def lowpriority():
    """ Set the priority of the process to below-normal."""

    import sys
    try:
        sys.getwindowsversion()
    except AttributeError:
        isWindows = False
    else:
        isWindows = True

    if isWindows:
        # Based on:
        #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
        #   http://code.activestate.com/recipes/496767/
        import win32api,win32process,win32con

        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.IDLE_PRIORITY_CLASS)
    else:
        import os

        os.nice(19)

class ProgressInFile:
    def __init__(self, id_thread):
        self._progress = 0
        self.f = open('thread_' + str(id_thread), 'wb')
    
    def set_progress(self, p: float):
        self.f.seek(0)
        self.f.write(struct.pack('f', float(p)))
        self.f.flush()
        self._progress = p
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close();

def train_model(id_thread):
    lowpriority()
    ra = RecommendationAlgoritm()
    with ProgressInFile(id_thread) as f:
        ra.train_model(f)
    return ra
    
