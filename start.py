from my_module import RecommendationAlgoritm
import my_module_interface

from flask import Flask, render_template, current_app
from multiprocessing import Process
import subprocess


import random
import struct
import threading
import time
import os


app = Flask(__name__, static_url_path='',
            static_folder='./web/static',
            template_folder='./web/templates')
app.debug = True

rec_alg = RecommendationAlgoritm()


@app.route('/get_recommendation/<int:user_id>', methods=["GET"])
def get_recommendation(user_id):
    global rec_alg
    recommendations = rec_alg.get_recommendation(user_id)
    return render_template('main.html',  tables=[recommendations.to_html(classes='data', index=False)],
                           titles=recommendations.columns.values)


@app.route('/train', methods=["POST"])
def train_model():
    thread_id = random.randint(0, 100000)
    p = Process(target=my_module_interface.train_model, args=(thread_id,))
    p.start()
    return str(thread_id)

@app.route('/progress/<int:thread_id>')
def progress(thread_id):
    filename = 'thread_' + str(thread_id)
    if os.path.exists(filename):
        with open(filename, 'rb+') as f:
            data = str(struct.unpack('f', f.read()))
        os.remove(filename)
    else:
        data = 0
    return str(data)
    


if __name__ == '__main__':
    app.run()
