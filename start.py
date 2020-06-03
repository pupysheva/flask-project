#!/usr/bin/python
# utf-8
import struct
import sys

if '-h' in sys.argv:
    print(
'''usage: python start.py [-h|-pkl|-no-t|-t]
-h	this help.
-pkl	read and save via pkl files.
-no-t	do not run a first train.
-t	do first train and not wait.
''')
    exit()

from reco_engine import RecommendationAlgorithm
import module_for_retraining

from flask import Flask, render_template
from multiprocessing import Process, Queue
from priority import hightpriority


import random
import tempfile
import threading
import time
import os
from datetime import datetime


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
    now = time.time()
    global rec_alg
    recommendations = rec_alg.get_recommendation(user_id)
    return render_template('recom_index.html',  tables=[recommendations.to_html(classes='data', index=False)],
                           titles=recommendations.columns.values,
                           time=(time.time() - now))


@app.route('/rated_by_user/<int:user_id>', methods=["GET"])
def rated_by_user(user_id):
    now = time.time()
    global rec_alg
    rated_by_user = rec_alg.get_films_rated_by_user(user_id)
    return render_template('rated_index.html',  tables=[rated_by_user.to_html(classes='data', index=False)],
                           titles=rated_by_user.columns.values,
                           time=(time.time() - now))


# @app.route('/train', methods=["POST"])
# def train_model():
#     def thf():
#         global rec_alg
#         rec_alg = module_for_retraining.train_model(thread_id, from_pkl)
#     thread_id = random.randint(0, 100000)
#     th = threading.Thread(target=thf, args=())
#     th.start()
#     return str(thread_id)

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


def train_model():
    def thf():
        print(datetime.now(), "train_model.t started")
        q = Queue()
        p = Process(target=module_for_retraining.train_model, args=(q, thread_id, from_pkl))
        p.start()
        global rec_alg
        rec_alg = q.get()
        print(datetime.now(), "train_model: Updated.")
    thread_id = random.randint(0, 100000)
    th = threading.Thread(target=thf, args=())
    th.start()
    return str(thread_id)


def train():
    train_model()
    threading.Timer(30*60, train).start()


def first_train():
    hightpriority()
    global rec_alg
    rec_alg = RecommendationAlgorithm(from_pkl=from_pkl)
    t = threading.Timer(
        60*2*60 if '-no-t' in sys.argv
        else 0 if '-t' in sys.argv
        else 0.1*60, train)
    t.start()


# При создании новых потоков в режиме spawn (Microsoft Windows),
# новые потоки выполняют инициализацию кода с флагом __name__ = '__mp_main__'.
# Надо запускать обучение только в том случае, если данный процесс
# является основным, то есть не является новым мульти-потоком mp (multiprocessing).
if __name__ != '__mp_main__':
    first_train()

# Запускать Flask надо только в том случае, если скрипт был запущен напрямую.
# Если скрипт запущен через Flask, то Flask запускать не нужно, он уже запущен.
if __name__ == '__main__':
    app.run(port=5000)
