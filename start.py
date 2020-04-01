from my_module import RecommendationAlgoritm

from flask import Flask, render_template, current_app


app = Flask(__name__)

rec_alg = RecommendationAlgoritm()


@app.route('/get_recommendation/<int:user_id>', methods=["POST","GET"])
def get_recommendation(user_id):
    global rec_alg
    recommendations = rec_alg.get_recommendation(user_id)
    return render_template('main.html',  tables=[recommendations.to_html(classes='data', index=False)],
                           titles=recommendations.columns.values)
@app.route('/train', methods=["POST"])
def train_model():
    global rec_alg
    rec_alg.train_model()

# if __name__ == '__main__':
#     app.run()
#     print("!!!!!!!!!!!!!!!!!!!!!!!!!!1",  current_app.name)