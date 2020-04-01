from my_module import RecommendationAlgoritm

from flask import Flask, render_template, current_app


import random
import threading
import time


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
    global rec_alg
    return get_start_thread(lambda progress: rec_alg.train_model(progress))


class ExportingThread(threading.Thread):
    def __init__(self):
        self.progress = 0
        super().__init__()

    def set_callback_thread(self, callback_thread):
        self.callback_thread = callback_thread
    
    def set_progress(self, p: float):
        print("%s%%" % (p * 100))
        self.progress = p

    def run(self):
        try:
            self.callback_thread(self)
        finally:
            self.progress = 1


exporting_threads = {}


def get_start_thread(callback_thread):
    global exporting_threads

    thread_id = random.randint(0, 10000)
    exporting_threads[thread_id] = ExportingThread()
    exporting_threads[thread_id].set_callback_thread(callback_thread)
    exporting_threads[thread_id].start()

    return str(thread_id)


@app.route('/progress/<int:thread_id>')
def progress(thread_id):
    global exporting_threads

    return str(exporting_threads[thread_id].progress)


if __name__ == '__main__':
    app.run()
