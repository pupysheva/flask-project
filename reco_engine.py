#!/usr/bin/python
# utf-8
import pickle
import pandas as pd
import numpy as np
import time
from datetime import datetime
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model.pkl_generator import generate_if_need
from sqlalchemy import create_engine, DateTime, Integer, String, Float
from model import SVD


class RecommendationAlgorithm:
    def __init__(self, from_pkl):

        if from_pkl:
            print("read in PKL ...")
            now = time.time()
            (self.data_with_user, self.movies_df, self.svd) = generate_if_need()
            print('\n Время чтения PKL файлов:', time.time() - now)

        else:
            print("read in DB ...")
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
                                                 columns=["u_id", "i_id", "rating", "timestamp"],
                                                 chunksize=1000000)
                now_r = time.time()
                # Create empty list
                list_of_dfs = []
                # Create empty dataframe
                self.data_with_user = pd.DataFrame()
                # Start Chunking
                for chunk_df in df_generator:
                    # Start Appending Data Chunks from SQL Result set into List
                    list_of_dfs.append(chunk_df)
                # Start appending data from list to dataframe
                self.data_with_user = pd.concat(list_of_dfs, ignore_index=True)
                print('\nВремя соединения chunk`ов', time.time() - now_r)
                self.data_with_user.drop(columns=['timestamp'], inplace=True)

            self.svd = generate_if_need(movies_df=self.movies_df, data_with_user=self.data_with_user)[2]
            print('\nВремя чтения BD:', time.time() - now)
        self.data_with_user_i_id_unique = self.data_with_user.i_id.unique()
        self.get_recommendation_cache = {}
    
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
                                                  "timestamp": DateTime})

    def get_films_rated_by_user(self, user_id):
        user_ratings = self.data_with_user[self.data_with_user.u_id == user_id]
        user_ratings.columns = ['u_id', 'i_id', 'rating']
        rated_df = self.movies_df[self.movies_df['i_id'].isin(user_ratings['i_id'])].\
            merge(pd.DataFrame(self.data_with_user).reset_index(drop=True), how='inner', left_on='i_id', right_on='i_id')
        rated_df = rated_df.loc[rated_df.u_id == user_id].sort_values(by='rating', ascending=False)
        return rated_df

    def get_recommendation(self, user_id, if_need_print_time = True):
        now = time.time()
        if user_id not in self.get_recommendation_cache:
            all_movies = self.data_with_user_i_id_unique
            recommendations = pd.DataFrame(list(product([user_id], all_movies)), columns=['u_id', 'i_id'])

            # Получение прогноза оценок для user_id
            pred_train = self.svd.predict(recommendations)
            recommendations['prediction'] = pred_train
            # sorted_user_predictions = recommendations.sort_values(by='prediction', ascending=False)
            # print(sorted_user_predictions.head(10))

            user_ratings = self.data_with_user[self.data_with_user.u_id == user_id]
            user_ratings.columns = ['u_id', 'i_id', 'rating']
            # Топ 20 фильмов для рекомендации
            recommendations = self.movies_df[~self.movies_df['i_id'].isin(user_ratings['i_id'])]. \
                merge(pd.DataFrame(recommendations).reset_index(drop=True), how='inner', left_on='i_id',
                      right_on='i_id'). \
                sort_values(by='prediction', ascending=False)
            self.get_recommendation_cache[user_id] = recommendations.head(20)
        if if_need_print_time:
            print('\nВремя алгоритма', time.time() - now)

        return self.get_recommendation_cache[user_id]

    def train_model(self, thread):
        print(datetime.now(), 'Start train...')
        thread.set_progress(0.01)
        print(datetime.now(), 'Progress set 0.01.')

        test_user = 0
        train_user = self.data_with_user.sample(frac=0.8)
        thread.set_progress(0.09)
        print(datetime.now(), 'self.data_with_user.sample(frac=0.8)')
        val_user = self.data_with_user.drop(train_user.index.tolist()).sample(frac=0.5, random_state=8)
        thread.set_progress(0.19)
        print(datetime.now(), 'self.data_with_user.drop(train_user.index.tolist()).sample(frac=0.5, random_state=8)')
        test_user = self.data_with_user.drop(train_user.index.tolist()).drop(val_user.index.tolist())
        print(datetime.now(), 'self.data_with_user.drop(train_user.index.tolist()).drop(val_user.index.tolist())')

        lr, reg, factors = (0.02, 0.016, 64)
        epochs = 10  # epochs = 50

        thread.set_progress(0.25)
        print(datetime.now(), 'start SVD create')
        svd = SVD(learning_rate=lr, regularization=reg, n_epochs=epochs, n_factors=factors,
                  min_rating=0.5, max_rating=5)
        print(datetime.now(), 'finish SVD create. Start fit...')

        thread.set_progress(0.50)
        svd.fit(Data=train_user, Data_val=val_user, early_stopping=False, shuffle=False, progress=lambda p: thread.set_progress(p * 0.25 + 0.50))  # early_stopping=True
        print(datetime.now(), 'finish svd.fit. Start predict')

        thread.set_progress(0.75)
        pred = svd.predict(test_user)
        print(datetime.now(), 'finish svd.predict. Start mean and sqrt')
        thread.set_progress(0.99)
        mae = mean_absolute_error(test_user["rating"], pred)
        rmse = np.sqrt(mean_squared_error(test_user["rating"], pred))
        print(datetime.now(), 'finish print results...')
        print("Test MAE:  {:.2f}".format(mae))
        print("Test RMSE: {:.2f}".format(rmse))
        print('{} factors, {} lr, {} reg'.format(factors, lr, reg))

        self.svd = svd
        thread.set_progress(1.00)
