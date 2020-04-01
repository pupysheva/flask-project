from my_module import RecommendationAlgoritm

from flask import Flask, render_template, current_app


import random
import threading
import time


app = Flask(__name__)

rec_alg = RecommendationAlgoritm()

def train(this):
    class Progress(keras.callbacks.Callback):
        def on_batch_end(self, batch, logs={}):
            print('self:', self)
            print('batch:', batch)
            print('logs:', logs)
    
    lr, reg, factors = (0.02, 0.016, 64)
    epochs = 10 

    this.progress = 25
    svd = SVD(learning_rate=lr, regularization=reg, n_epochs=epochs, n_factors=factors,
              min_rating=0.5, max_rating=5)
    
    this.progress = 50
    svd.fit(X=train_user, X_val=val_user, early_stopping=False, shuffle=False, callbacks=[Progress()])#early_stopping=True

    this.progress = 75
    pred = svd.predict(test_user)
    mae = mean_absolute_error(test_user["rating"], pred)
    rmse = np.sqrt(mean_squared_error(test_user["rating"], pred))
    print("Test MAE:  {:.2f}".format(mae))
    print("Test RMSE: {:.2f}".format(rmse))
    print('{} factors, {} lr, {} reg'.format(factors, lr, reg))
    this.progress = 100

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
    
    def set_callback_thread(self, callback_thread):
        self.callback_thread = callback_thread
    
    def run(self):
        try:
            self.callback_thread(self)
        finally:
            self.progress = 100


exporting_threads = {}
app = Flask(__name__, static_url_path='', 
            static_folder='web/static',
            template_folder='web/templates')
app.debug = True

def get_start_thread(callback_thread):
    global exporting_threads

    thread_id = random.randint(0, 10000)
    exporting_threads[thread_id] = ExportingThread()
    exporting_threads[thread_id].set_callback_thread(callback_thread)
    exporting_threads[thread_id].start()

    return str(thread_id)

@app.route('/train')
def trainUrl():
    return get_start_thread(train)

@app.route('/progress/<int:thread_id>')
def progress(thread_id):
    global exporting_threads

    return str(exporting_threads[thread_id].progress)


if __name__ == '__main__':
    app.run()
