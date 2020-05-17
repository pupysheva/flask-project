#!/usr/bin/python
# utf-8
from model.loading_dataset import fetch_ml_ratings
from model import SVD

from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd
import numpy as np
import pickle
import os
import shutil
import time


def create_svd(svd_filename, data_with_user):
    train_user = data_with_user.sample(frac=0.8)
    val_user = data_with_user.drop(train_user.index.tolist()).sample(frac=0.5, random_state=8)
    test_user = data_with_user.drop(train_user.index.tolist()).drop(val_user.index.tolist())

    # lr, reg, factors = (0.007, 0.03, 90)
    lr, reg, factors = (0.02, 0.016, 64)
    epochs = 10  # epochs = 50
    svd = SVD(learning_rate=lr, regularization=reg, n_epochs=epochs, n_factors=factors, min_rating=0.5, max_rating=5)
    svd.fit(Data=train_user, Data_val=val_user, early_stopping=False, shuffle=False)

    pred = svd.predict(test_user)
    mae = mean_absolute_error(test_user["rating"], pred)
    rmse = np.sqrt(mean_squared_error(test_user["rating"], pred))
    print("Test MAE:  {:.2f}".format(mae))
    print("Test RMSE: {:.2f}".format(rmse))
    print('{} factors, {} lr, {} reg'.format(factors, lr, reg))

    with open(svd_filename, 'wb') as f:
        pickle.dump(svd, f)
    return svd


def create_or_load_svd(svd_filename, data_with_user):
    if os.path.exists(svd_filename):
        return pd.read_pickle(svd_filename)
    else:
        return create_svd(svd_filename, data_with_user)


def create_or_load_movies_df(variant):
    return fetch_ml_ratings(target_df='movies', variant=variant)


def create_or_load_data_with_user(variant, movies_df, data_with_user_filename):
    return create_data_with_user(fetch_ml_ratings(target_df='ratings', variant=variant), movies_df, data_with_user_filename)



def create_or_load_dfs(variant, data_with_user, movies_df, data_with_user_filename):
    now = time.time()
    if movies_df is None:
        movies_df = create_or_load_movies_df(variant)
    if data_with_user is None:
        data_with_user = create_or_load_data_with_user(variant, movies_df, data_with_user_filename)
    print('\n', time.time() - now)
    return data_with_user, movies_df



def create_data_with_user(df, movies_df, data_with_user_filename):
    model = df.copy()
    # #  Инициализация пользователя 0
    # n_m = len(model.i_id.unique())
    # my_ratings = np.zeros(n_m)
    # my_ratings[4993] = 5
    # my_ratings[1080] = 5
    # my_ratings[260] = 5
    # my_ratings[4896] = 5
    # my_ratings[1196] = 5
    # my_ratings[1210] = 5
    # my_ratings[2628] = 5
    # my_ratings[5378] = 5
    # print('User ratings:')
    # print('-----------------')
    # for i, val in enumerate(my_ratings):
    #     if val > 0:
    #         print('Rated %d stars: %s' %
    #               (val, movies_df.loc[movies_df.i_id == i].title.values))
    # print("Adding your recommendations!")
    # items_id = [item[0] for item in np.argwhere(my_ratings > 0)]
    # ratings_list = my_ratings[np.where(my_ratings > 0)]
    # user_id = np.asarray([0] * len(ratings_list))
    #
    # user_ratings = pd.DataFrame(
    #     list(zip(user_id, items_id, ratings_list)), columns=['u_id', 'i_id', 'rating'])
    # try:
    #     model = model.drop(columns=['timestamp'])
    # except:
    #     pass
    # data_with_user = model.append(user_ratings, ignore_index=True)
    #
    # now = time.time()
    # data_with_user.to_pickle(data_with_user_filename)
    # print('\n', time.time() - now)
    # Create an empty dictionary.
    my_ratings = {}
    print("TEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
    # Fill the dictionary, pair by pair.
    my_ratings[920] = 5
    my_ratings[1721] = 5
    my_ratings[47382] = 4
    my_ratings[7669] = 5
    my_ratings[4246] = 5
    my_ratings[55052] = 5
    my_ratings[8969] = 5
    my_ratings[41569] = 5
    my_ratings[6373] = 1
    my_ratings[1485] = 1
    my_ratings[7361] = 2
    my_ratings[64969] = 1
    my_ratings[19] = 1

    for i, val in my_ratings.items():
        print('Rated %d %d stars: %s' % (val, i, movies_df.loc[movies_df.i_id == i].title.values))

    print("Adding your recommendations!")
    items_id = list(my_ratings.keys())
    ratings_list = list(my_ratings.values())
    user_id = np.asarray([0] * len(ratings_list))
    user_ratings = pd.DataFrame(list(zip(user_id, items_id, ratings_list)), columns=['u_id', 'i_id', 'rating'])

    try:
        model = model.drop(columns=['timestamp'])
    except:
        pass
    data_with_user = model.append(user_ratings, ignore_index=True)
    return data_with_user

def generate_if_need(path='./resources', svd=None, movies_df=None, data_with_user=None):
    svd_filename = '/'.join((path, 'ml-20m/model_svd.pkl'))
    data_with_user_filename = '/'.join((path, 'ml-20m/data_with_user.pkl'))
    movies_filename = '/'.join((path, 'ml-20m/movies.pkl'))
    variant = '20m'

    # ИЗМЕНИТЬ ЕСЛИ ОБОИХ ФАЙЛОВ НЕТ, ТО НУЖНО СКАЧАТЬ ЕЩЁ РАЗ или взять из zip
    if data_with_user is None or movies_df is None:
        (data_with_user, movies_df) = create_or_load_dfs(variant, data_with_user, movies_df, data_with_user_filename)

    if svd is None:
        svd = create_or_load_svd(svd_filename, data_with_user)

    return (data_with_user, movies_df, svd)
