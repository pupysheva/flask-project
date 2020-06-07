#!/usr/bin/python
# utf-8
import pickle
import pandas as pd
import numpy as np
import time
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model.pkl_generator import generate_if_need
from sqlalchemy import create_engine, DateTime, Integer, String, Float
from model import SVD
from logger import log


class RecommendationAlgorithm:
    def __init__(self, from_pkl):
        if from_pkl:
            log('read in PKL ...', (RecommendationAlgorithm, self.__init__))
            now = time.time()
            (self.data_with_user, self.movies_df, self.svd) = generate_if_need()
            log('Время чтения PKL файлов {} seconds'.format(time.time() - now), (RecommendationAlgorithm, self.__init__))
        else:
            log("read in DB ...", (RecommendationAlgorithm, self.__init__))
            now = time.time()
            db_url = 'sqlite:///./movies_recom_db.db'
            engine = create_engine(db_url)

            if not engine.dialect.has_table(engine, "Movies"):
                self.movies_df = generate_if_need(svd="ignore", data_with_user="ignore")[1]
                self.save(in_db=True, engine=engine, table_name="Movies")
            else:
                self.movies_df = pd.read_sql_table("Movies", con=engine)

            if not engine.dialect.has_table(engine, "Ratings"):
                self.data_with_user = generate_if_need(svd="ignore", movies_df=self.movies_df)[0]
                self.save(in_db=True, engine=engine, table_name="Ratings")
            else:
                df_generator = pd.read_sql_table("Ratings",
                                                 con=engine,
                                                 columns=["u_id", "i_id", "rating"],
                                                 chunksize=1000000)
                now_r = time.time()
                list_of_dfs = []
                self.data_with_user = pd.DataFrame()
                for chunk_df in df_generator:
                    list_of_dfs.append(chunk_df)
                self.data_with_user = pd.concat(list_of_dfs, ignore_index=True)
                log('Время соединения chunk`ов: {} seconds'.format(time.time() - now_r), (RecommendationAlgorithm, self.__init__))

            self.svd = generate_if_need(movies_df=self.movies_df, data_with_user=self.data_with_user)[2]
            log('Время чтения DB: {} seconds'.format(time.time() - now), (RecommendationAlgorithm, self.__init__))
        self.data_with_user_i_id_unique = self.data_with_user.i_id.unique()
        self.get_recommendation_cache = {}
        self.data_with_user_u_id_unique = self.data_with_user.u_id.unique()

    def save(self, engine, table_name, in_pkl=False, in_db=False):
        if in_pkl:
            with open('resources/ml-20m/model_svd.pkl', 'wb') as f:
                pickle.dump(self.svd, f)
            self.data_with_user.to_pickle('resources/ml-20m/data_with_user.pkl')
            self.movies_df.to_pickle('resources/ml-20m/movies.pkl')
        if in_db:
            if table_name == "Movies":
                self.movies_df.to_sql(name=table_name,
                                      con=engine,
                                      if_exists='replace',
                                      index=False,
                                      dtype={"i_id": Integer,
                                             "title": String(50),
                                             "genres": String(150)})
            if table_name == "Ratings":
                self.data_with_user.to_sql(name=table_name,
                                           con=engine,
                                           if_exists='replace',
                                           index=True, #Запишет индексы строк dataFrame в виде столбца
                                           index_label = "id_in_db",
                                           dtype={"user_id": Integer,
                                                  "movie_id": Integer,
                                                  "rating": Float,
                                                  "date_time": DateTime})

    def get_films_rated_by_user(self, user_id):
        user_ratings = self.data_with_user[self.data_with_user.u_id == user_id]
        user_ratings.columns = ['u_id', 'i_id', 'rating']
        rated_df = self.movies_df[self.movies_df['i_id'].isin(user_ratings['i_id'])].\
            merge(pd.DataFrame(self.data_with_user).reset_index(drop=True), how='inner', left_on='i_id', right_on='i_id')
        rated_df = rated_df.loc[rated_df.u_id == user_id].sort_values(by='rating', ascending=False)
        return rated_df

    def get_recommendation(self, user_id, df=None, if_need_print_time=True):
        if df is None:
            res = self.recoms(user_id, self.data_with_user, if_need_print_time)
        else:
            res = self.recoms(user_id, df, if_need_print_time)
        return res

    def recoms(self, user_id, df, if_need_print_time=True):
        now = time.time()
        if user_id in self.data_with_user_u_id_unique:
            # if user_id not in self.get_recommendation_cache:
            #     all_movies = self.data_with_user_i_id_unique
            #     recommendations = pd.DataFrame(list(product([user_id], all_movies)), columns=['u_id', 'i_id'])
            #     # Получение прогноза оценок для user_id
            #     pred_train = self.svd.predict(recommendations)
            #     recommendations['prediction'] = pred_train
            #     user_ratings = self.data_with_user[self.data_with_user.u_id == user_id]
            #     user_ratings.columns = ['u_id', 'i_id', 'rating']
            #     # Топ 20 фильмов для рекомендации
            #     recommendations = self.movies_df[~self.movies_df['i_id'].isin(user_ratings['i_id'])]. \
            #         merge(pd.DataFrame(recommendations).reset_index(drop=True), how='inner', left_on='i_id',
            #               right_on='i_id'). \
            #         sort_values(by='prediction', ascending=False)
            #     self.get_recommendation_cache[user_id] = recommendations.head(20)
            all_movies = self.data_with_user_i_id_unique
            recommendations = pd.DataFrame(list(product([user_id], all_movies)), columns=['u_id', 'i_id'])
            # Получение прогноза оценок для user_id
            pred_train = self.svd.predict(recommendations)
            recommendations['prediction'] = pred_train
            user_ratings = df[df.u_id == user_id]
            user_ratings.columns = ['u_id', 'i_id', 'rating']
            # Топ 20 фильмов для рекомендации
            recommendations = self.movies_df[~self.movies_df['i_id'].isin(user_ratings['i_id'])]. \
                merge(pd.DataFrame(recommendations).reset_index(drop=True), how='inner', left_on='i_id',
                      right_on='i_id'). \
                sort_values(by='prediction', ascending=False)
            if if_need_print_time:
                log('Время алгоритма {} seconds'.format(time.time() - now), (RecommendationAlgorithm, self.recoms))
            return recommendations.head(20)
        else:
            return pd.DataFrame()

    def train_model(self, thread=None):
        log('Start train...', (RecommendationAlgorithm, self.train_model))
        if thread is not None: thread.set_progress(0.01)
        log('Progress set 0.01.', (RecommendationAlgorithm, self.train_model))

        train_data = self.data_with_user.sample(frac=0.8)
        if thread is not None: thread.set_progress(0.09)
        log('self.data_with_user.sample(frac=0.8)', (RecommendationAlgorithm, self.train_model))
        val_data = self.data_with_user.drop(train_data.index.tolist()).sample(frac=0.5, random_state=8)
        if thread is not None: thread.set_progress(0.19)
        log('self.data_with_user.drop(train_user.index.tolist()).sample(frac=0.5, random_state=8)', (RecommendationAlgorithm, self.train_model))
        test_data = self.data_with_user.drop(train_data.index.tolist()).drop(val_data.index.tolist())
        log('self.data_with_user.drop(train_user.index.tolist()).drop(val_user.index.tolist())', (RecommendationAlgorithm, self.train_model))

        log('Деление на выборки: {} {} {}'.format(
            len(np.unique(self.data_with_user["u_id"])),
            len(np.unique(train_data["u_id"])),
            len(np.unique(test_data["u_id"]))),
        (RecommendationAlgorithm, self.train_model))

        learning_rate, reg, features = (0.02, 0.015, 64)
        epochs = 10

        if thread is not None: thread.set_progress(0.25)
        log('start SVD create', (RecommendationAlgorithm, self.train_model))
        svd = SVD(learning_rate=learning_rate, regularization=reg, n_epochs=epochs, n_factors=features,
                  min_rating=0.5, max_rating=5)
        log('finish SVD create. Start fit...', (RecommendationAlgorithm, self.train_model))

        if thread is not None:
            thread.set_progress(0.50)
            svd.fit(Data=train_data, Data_val=val_data, progress=lambda p: thread.set_progress(p * 0.25 + 0.50))
        else:
            svd.fit(Data=train_data, Data_val=val_data)

        log('finish svd.fit. Start predict', (RecommendationAlgorithm, self.train_model))

        if thread is not None: thread.set_progress(0.75)
        pred = svd.predict(test_data)
        log('finish svd.predict. Start mean and sqrt', (RecommendationAlgorithm, self.train_model))
        if thread is not None: thread.set_progress(0.99)
        mae = mean_absolute_error(test_data["rating"], pred)
        mse = mean_squared_error(test_data["rating"], pred)
        rmse = np.sqrt(mse)
        log('finish print results...', (RecommendationAlgorithm, self.train_model))
        log("Test MAE:  {:.2f}".format(mae), (RecommendationAlgorithm, self.train_model))
        log("Test RMSE: {:.2f}".format(rmse), (RecommendationAlgorithm, self.train_model))
        log('{} factors, {} lr, {} reg'.format(learning_rate, reg, features), (RecommendationAlgorithm, self.train_model))

        self.svd = svd
        if thread is not None: thread.set_progress(1.00)
        return train_data, test_data

    def train_for_ranking_test(self):
        # Деление на выборки специально для ranking_test
        l = self.data_with_user_u_id_unique
        now = time.time()
        for i,u in enumerate(l):
            log(i, (RecommendationAlgorithm, self.train_for_ranking_test))
            u_rated = self.data_with_user[self.data_with_user["u_id"] == u]
            user_train = u_rated.head(12)
            mask = u_rated['i_id'].isin(user_train["i_id"].unique())
            user_test = u_rated[~mask]
            if i == 0:
                train_df = user_train
                test_df = user_test
            else:
                train_df = train_df.append(user_train, ignore_index=True)
                test_df = test_df.append(user_test, ignore_index=True)
        log(time.time() - now, (RecommendationAlgorithm, self.train_for_ranking_test))
        train_df.to_pickle("./tests/train.pkl")
        test_df.to_pickle("./tests/test.pkl")
        learning_rate, reg, features = (0.02, 0.015, 64)
        epochs = 10
        svd = SVD(learning_rate=learning_rate, regularization=reg, n_epochs=epochs, n_factors=features,
                  min_rating=0.5, max_rating=5)
        svd.fit(Data=train_df)
        self.svd = svd
        return train_df, test_df
