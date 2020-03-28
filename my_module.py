import pickle
import pandas as pd
import time
from itertools import product


class RecommendationAlgoritm:
    def __init__(self):
        now = time.time()

        with open('resources/ml-20m/model_svd.pkl', 'rb') as f:
            self.svd = pickle.load(f)

        self.data_with_user = pd.read_pickle('resources/ml-20m/data_with_user.pkl')
        self.movies_df = pd.read_pickle('resources/ml-20m/movies.pkl')
        print('\n Время чтения:', time.time() - now)

    def get_films_rated_by_user(self, user_id):
        user_ratings = self.data_with_user[self.data_with_user.u_id == user_id]
        user_ratings.columns = ['u_id',	'i_id', 'rating']
        rated_df = self.movies_df[self.movies_df['i_id'].isin(user_ratings['i_id'])].\
            merge(pd.DataFrame(self.data_with_user).reset_index(drop=True), how='inner', left_on='i_id', right_on='i_id')
        rated_df = rated_df.loc[rated_df.u_id == user_id].sort_values(by='rating', ascending=False)
        return rated_df

    def get_recommendation(self, user_id):
        user_id = [user_id]

        now = time.time()
        all_movies = self.data_with_user.i_id.unique()
        recommendations = pd.DataFrame(list(product(user_id, all_movies)), columns=['u_id', 'i_id'])

        # Получение прогноза оценок для user_id
        pred_train = self.svd.predict(recommendations)
        recommendations['prediction'] = pred_train

        sorted_user_predictions = recommendations.sort_values(by='prediction', ascending=False)
        print(sorted_user_predictions.head(10))

        user_ratings = self.data_with_user[self.data_with_user.u_id == user_id[0]]
        user_ratings.columns = ['u_id', 'i_id', 'rating']
        # Топ 20 фильмов для рекомендации
        recommendations = self.movies_df[~self.movies_df['i_id'].isin(user_ratings['i_id'])]. \
            merge(pd.DataFrame(sorted_user_predictions).reset_index(drop=True), how='inner', left_on='i_id',
                  right_on='i_id'). \
            sort_values(by='prediction', ascending=False)
        print('\nВремя алгоритма', time.time() - now)

        return recommendations.head(20)

    def _train_model(self):
        return 0
