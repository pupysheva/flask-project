from DIPLOMv1.dataset import fetch_ml_ratings
from DIPLOMv1 import SVD

from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd
import numpy as np
import pickle
import os
import shutil
import time


def creator(
        svd_filename='./ml-20m/model_svd.pkl',
        data_with_user_filename='./ml-20m/data_with_user.pkl',
        movies_filename='./ml-20m/movies.pkl',
        variant='20m'):
    now = time.time()
    df = fetch_ml_ratings(variant=variant)
    movies_df = pd.read_csv('ml-%s/movies.csv' % variant,
                            names=['i_id', 'title', 'genres'], sep=',', encoding='latin-1')
    movies_df.drop([0], inplace=True)
    movies_df['i_id'] = movies_df['i_id'].apply(pd.to_numeric)
    model = df.copy()
    print('\n', time.time()-now)


    #  Инициализация пользователя 0
    n_m = len(model.i_id.unique())
    my_ratings = np.zeros(n_m)
    my_ratings[4993] = 5
    my_ratings[1080] = 5
    my_ratings[260] = 5
    my_ratings[4896] = 5
    my_ratings[1196] = 5
    my_ratings[1210] = 5
    my_ratings[2628] = 5
    my_ratings[5378] = 5
    print('User ratings:')
    print('-----------------')
    for i, val in enumerate(my_ratings):
        if val > 0:
            print('Rated %d stars: %s' %
                  (val, movies_df.loc[movies_df.i_id == i].title.values))
    print("Adding your recommendations!")
    items_id = [item[0] for item in np.argwhere(my_ratings > 0)]
    ratings_list = my_ratings[np.where(my_ratings > 0)]
    user_id = np.asarray([0] * len(ratings_list))

    user_ratings = pd.DataFrame(
        list(zip(user_id, items_id, ratings_list)), columns=['u_id', 'i_id', 'rating'])

    try:
        model = model.drop(columns=['timestamp'])
    except:
        pass
    data_with_user = model.append(user_ratings, ignore_index=True)

    train_user = data_with_user.sample(frac=0.8)
    val_user = data_with_user.drop(train_user.index.tolist()).sample(frac=0.5, random_state=8)
    test_user = data_with_user.drop(train_user.index.tolist()).drop(val_user.index.tolist())

    # lr, reg, factors = (0.007, 0.03, 90)
    lr, reg, factors = (0.02, 0.016, 64)
    epochs = 10  # epochs = 50
    svd = SVD(learning_rate=lr, regularization=reg, n_epochs=epochs, n_factors=factors, min_rating=0.5, max_rating=5)
    svd.fit(Data=train_user, Data_val=val_user, early_stopping=False, shuffle=False)  # early_stopping=True

    pred = svd.predict(test_user)
    mae = mean_absolute_error(test_user["rating"], pred)
    rmse = np.sqrt(mean_squared_error(test_user["rating"], pred))
    print("Test MAE:  {:.2f}".format(mae))
    print("Test RMSE: {:.2f}".format(rmse))
    print('{} factors, {} lr, {} reg'.format(factors, lr, reg))

    now = time.time()
    with open(svd_filename, 'wb') as f:
        pickle.dump(svd, f)
    print('\n', time.time()-now)

    now = time.time()
    data_with_user.to_pickle(data_with_user_filename)
    print('\n', time.time()-now)

    now = time.time()
    movies_df.to_pickle(movies_filename)
    print('\n', time.time()-now)


def generate_if_need(path='./resources', need_autoremove=True):
    svd_filename = '/'.join((path, 'ml-20m/model_svd.pkl'))
    data_with_user_filename = '/'.join((path, 'ml-20m/data_with_user.pkl'))
    movies_filename = '/'.join((path, 'ml-20m/movies.pkl'))
    variant = '20m'
    if os.path.exists(svd_filename) and os.path.exists(data_with_user_filename) and os.path.exists(movies_filename):
        return
    else:
        os.makedirs('/'.join((path, 'ml-%s' % variant)), exist_ok=True)
        creator(svd_filename, data_with_user_filename,
                movies_filename, variant)
        if need_autoremove:
            shutil.rmtree('./ml-%s' % variant, ignore_errors=True)
