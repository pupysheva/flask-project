from my_module import RecommendationAlgoritm

from flask import Flask, render_template, current_app


import random
import threading
import time


app = Flask(__name__)

rec_alg = RecommendationAlgoritm()


@app.route('/get_recommendation/<int:user_id>', methods=["GET"])
def get_recommendation(user_id):
    global rec_alg
    recommendations = rec_alg.get_recommendation(user_id)
    return render_template('main.html',  tables=[recommendations.to_html(classes='data', index=False)],
                           titles=recommendations.columns.values)

# if __name__ == '__main__':
#     app.run()
#     print("!!!!!!!!!!!!!!!!!!!!!!!!!!1",  current_app.name)

#@app.route('/get_recommendation/<int:user_id>/load', methods=["GET"])
#def get_recommendation(user_id):


class ExportingThread(threading.Thread):
    def __init__(self):
        self.progress = 0
        super().__init__()

    def run(self):
        # Your exporting stuff goes here ...
        for _ in range(10):
            time.sleep(1)
            self.progress += 10


exporting_threads = {}
app = Flask(__name__, static_url_path='', 
            static_folder='web/static',
            template_folder='web/templates')
app.debug = True

@app.route('/')
def index():
    global exporting_threads

    thread_id = random.randint(0, 10000)
    exporting_threads[thread_id] = ExportingThread()
    exporting_threads[thread_id].start()

    return '%s' % thread_id


@app.route('/progress/<int:thread_id>')
def progress(thread_id):
    global exporting_threads

    return str(exporting_threads[thread_id].progress)


if __name__ == '__main__':
    app.run()
