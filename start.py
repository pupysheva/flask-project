#!/usr/bin/python
# utf-8
import sys

if '-h' in sys.argv:
    print(
'''usage: ./start.py [-h|-pkl]
-h:	this help.
-pkl:	read and save via pkl files.
''')
    exit()

from reco_engine import RecommendationAlgorithm
import module_for_retraining

from flask import Flask, render_template
from multiprocessing import Process, Queue


import random
import struct
import tempfile
import threading
import time
import os


app = Flask(__name__, static_url_path='',
            static_folder='./web/static',
            template_folder='./web/templates')
app.debug = False

tmppath = '{}/{}'.format(tempfile.gettempdir(), 'flask-project')
if not os.path.exists(tmppath):
    os.mkdir(tmppath)

rec_alg = None
from_pkl = '-pkl' in sys.argv

@app.route('/get_recommendation/<int:user_id>', methods=["GET"])
def get_recommendation(user_id):
    past = time.time()
    global rec_alg
    recommendations = rec_alg.get_recommendation(user_id)
    return render_template('main.html',  tables=[recommendations.to_html(classes='data', index=False)],
                           titles=recommendations.columns.values,
                           time=(time.time() - past))

@app.route('/train', methods=["POST"])
def train_model():
    def thf():
        print(time.time(), "train_model.t started")
        q = Queue()
        p = Process(target=module_for_retraining.train_model, args=(q, thread_id, from_pkl))
        p.start()
        global rec_alg
        rec_alg = q.get()
        print(time.time(), "train_model: Updated.")
    thread_id = random.randint(0, 100000)
    th = threading.Thread(target=thf, args=())
    th.start()
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

def train():
    train_model()
    threading.Timer(60*2*60, train).start()

def main():
    from priority import hightpriority
    hightpriority()
    global rec_alg
    rec_alg = RecommendationAlgorithm(from_pkl=from_pkl)
    t = threading.Timer(5*60, train)
    t.start()
    # app.run()

# if __name__ == '__main__':
main()
