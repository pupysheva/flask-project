from my_module import RecommendationAlgoritm

from flask import Flask


app = Flask(__name__)

rec_alg = RecommendationAlgoritm()


@app.route('/get_recommendation/<int:user_id>', methods=("POST", "GET"))
def get_recommendation(user_id):
    recommendations = rec_alg.get_recommendation(user_id)
    return recommendations.to_html(index=False)
