from .my_module import RecommendationAlgoritm
import struct

def get_recommendation(user_id):
    ra = RecommendationAlgoritm()
    return ra.get_recommendation(user_id)

class ProgressInFile:
    def __init__(self, id_thread):
        self._progress = 0
        self.f = open('thread_' + id_thread, 'rb')
    
    def set_progress(self, p: float):
        self.f.seek(0)
        self.f.write(struct.pack(f, float(p)))
        self.f.flush()
        self._progress = p
    
    def __del__(self):
        f.close();

def train_model(id_thread):
    ra = RecommendationAlgoritm()
    with ProgressInFile() as f:
        ra.train_model(f)
