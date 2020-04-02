from my_module import RecommendationAlgoritm
import my_module_interface

from flask import Flask, render_template, current_app
from multiprocessing import Process, Queue
import subprocess


import random
import struct
import tempfile
import threading
import time
import os


app = Flask(__name__, static_url_path='',
            static_folder='./web/static',
            template_folder='./web/templates')
app.debug = True

tmppath = '{}/{}'.format(tempfile.gettempdir(), 'flask-project')
if not os.path.exists(tmppath):
    os.mkdir(tmppath)

rec_alg = RecommendationAlgoritm()


@app.route('/get_recommendation/<int:user_id>', methods=["GET"])
def get_recommendation(user_id):
    global rec_alg
    recommendations = rec_alg.get_recommendation(user_id)
    return render_template('main.html',  tables=[recommendations.to_html(classes='data', index=False)],
                           titles=recommendations.columns.values)

@app.route('/train', methods=["POST"])
def train_model():
    def t():
        print(time.time(), "train_model.t started")
        q = Queue()
        p = Process(target=my_module_interface.train_model, args=(q, thread_id))
        p.start()
        global rec_alg
        rec_alg = q.get()
        print(time.time(), "train_model: Updated.")
    thread_id = random.randint(0, 100000)
    t = threading.Thread(target=t, args=())
    t.start()
    return str(thread_id)

@app.route('/progress/<int:thread_id>')
def progress(thread_id):
    filename = tmppath + '/thread_' + str(thread_id)
    if os.path.exists(filename):
        with open(filename, 'rb+') as f:
            bytes = f.read(4)
            if len(bytes) >= 4:
                data = str(struct.unpack('f', bytes)[0])
            else:
                data = 0
        if data == 1:
            os.remove(filename)
    else:
        data = 0
    return str(data)
    


if __name__ == '__main__':
    from priority import hightpriority
    hightpriority()
    app.run()
