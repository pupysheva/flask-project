#!/usr/bin/python
# utf-8
from reco_engine import RecommendationAlgorithm
from multiprocessing import Queue
from priority import lowpriority
import struct
import tempfile

tmppath = '{}/{}'.format(tempfile.gettempdir(), 'flask-project')


class ProgressInFile:
    def __init__(self, id_thread):
        self._progress = 0
        self.f = open(tmppath + '/thread_' + str(id_thread), 'wb')
    
    def set_progress(self, p: float):
        self.f.seek(0)
        self.f.write(struct.pack('f', float(p)))
        self.f.flush()
        self._progress = p
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()


def train_model(q: Queue, id_thread: int):
    lowpriority()
    ra = RecommendationAlgorithm(from_pkl=False)
    with ProgressInFile(id_thread) as f:
        ra.train_model(f)
    q.put(ra)

