import pickle
import pandas as pd
import numpy as np
import time
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error

from DIPLOMv1 import SVD


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
        # print(sorted_user_predictions.head(10))

        user_ratings = self.data_with_user[self.data_with_user.u_id == user_id[0]]
        user_ratings.columns = ['u_id', 'i_id', 'rating']
        # Топ 20 фильмов для рекомендации
        recommendations = self.movies_df[~self.movies_df['i_id'].isin(user_ratings['i_id'])]. \
            merge(pd.DataFrame(sorted_user_predictions).reset_index(drop=True), how='inner', left_on='i_id',
                  right_on='i_id'). \
            sort_values(by='prediction', ascending=False)
        print('\nВремя алгоритма', time.time() - now)

        return recommendations.head(20)

    def train_model(self):
        test_user = 0
        train_user = self.data_with_user.sample(frac=0.8)
        val_user = self.data_with_user.drop(train_user.index.tolist()).sample(frac=0.5, random_state=8)
        test_user = self.data_with_user.drop(train_user.index.tolist()).drop(val_user.index.tolist())

        lr, reg, factors = (0.02, 0.016, 64)
        epochs = 10  # epochs = 50

        svd = SVD(learning_rate=lr, regularization=reg, n_epochs=epochs, n_factors=factors,
                  min_rating=0.5, max_rating=5)

        svd.fit(X=train_user, X_val=val_user, early_stopping=False, shuffle=False)  # early_stopping=True

        pred = svd.predict(test_user)
        mae = mean_absolute_error(test_user["rating"], pred)
        rmse = np.sqrt(mean_squared_error(test_user["rating"], pred))
        print("Test MAE:  {:.2f}".format(mae))
        print("Test RMSE: {:.2f}".format(rmse))
        print('{} factors, {} lr, {} reg'.format(factors, lr, reg))

        self.svd = svd
