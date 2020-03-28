from collections import namedtuple
import pandas as pd
import my_module

from flask import Flask, render_template, redirect, url_for, request


app = Flask(__name__)

Message = namedtuple('Message', 'text tag')
messages = []
df = pd.DataFrame({'A': [0, 1, 2, 3, 4],
                   'B': [5, 100, 25, -1, 9],
                   'C': ['a', 'b', 'c--', 'd', 'e']})


@app.route('/get_recommendation/<user_id>', methods=("POST", "GET"))
def get_recommendation(user_id):
    recommendations = my_module.get_recommendation(int(user_id))
    return recommendations.to_html(index=False)


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


# @app.route('/main', methods=['GET'])
# def main():
#     return render_template('main.html', messages=messages)
#
#
# @app.route('/add_message', methods=['POST'])
# def add_message():
#     text = request.form['text']
#     tag = request.form['tag']
#     messages.append(Message(text,tag))
#     return redirect(url_for('main'))
