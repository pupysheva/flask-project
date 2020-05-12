#!/usr/bin/python
# utf-8
import datetime
import os
import urllib.request
import shutil
import zipfile

import numpy as np
import pandas as pd


__all__ = [
    'fetch_ml_ratings',
]

VARIANTS = {
    '100k': {'rating_filename': 'u.data', 'sep': '\t'},
    '20m': {'ratings': {'filename': 'ratings', 'sep': ','},
            'movies': {'filename': 'movies', 'sep': ','}}
}


def ml_ratings_csv_to_df(ratings_csv_path, variant):
    # Чтение ratings.csv
    names = ['u_id', 'i_id', 'rating', 'timestamp']
    dtype = {'u_id': np.uint32, 'i_id': np.uint32, 'rating': np.float64}
    def date_parser(time):
        return datetime.datetime.fromtimestamp(float(time))
    ratings_df = pd.read_csv(ratings_csv_path, names=names, dtype=dtype, header=0,
                     sep=VARIANTS[variant]['ratings']['sep'], parse_dates=['timestamp'],
                     date_parser=date_parser, engine='python')
    ratings_df.sort_values(by='timestamp', inplace=True)
    ratings_df.reset_index(drop=True, inplace=True)

    return ratings_df


def ml_movies_csv_to_df(movie_csv_path, variant):
    # Чтение movies.csv
    movies_df = pd.read_csv(movie_csv_path, names=['i_id', 'title', 'genres'], sep=VARIANTS[variant]['movies']['sep'],
                            encoding='latin-1', header=0)
    movies_df['i_id'] = movies_df['i_id'].apply(pd.to_numeric)
    return movies_df


def fetch_ml_ratings(target_df, data_dir_path="./resources/", variant='20m'):
    os.makedirs(data_dir_path, exist_ok=True)
    dirname = 'ml-' + variant
    ratings_filename = VARIANTS[variant]['ratings']['filename']
    csv_path_ratings = os.path.join(data_dir_path, dirname, ratings_filename) + '.csv'
    pkl_path_ratings = os.path.join(data_dir_path, dirname, ratings_filename) + '.pkl'

    movies_filename = VARIANTS[variant]['movies']['filename']
    csv_path_movies = os.path.join(data_dir_path, dirname, movies_filename) + '.csv'
    pkl_path_movies = os.path.join(data_dir_path, dirname, movies_filename) + '.pkl'

    zip_path = os.path.join(data_dir_path, dirname) + '.zip'
    url = 'http://files.grouplens.org/datasets/movielens/ml-' + variant + \
              '.zip'

    if target_df == "ratings" and os.path.exists(csv_path_ratings):
        if not os.path.exists(pkl_path_ratings):
            print('read csv...', csv_path_ratings)
            df = ml_ratings_csv_to_df(csv_path_ratings, variant)
            print('save csv.pkl...', pkl_path_ratings)
            df.to_pickle(pkl_path_ratings)
        else:
            print('read csv.pkl...', pkl_path_ratings)
            df = pd.read_pickle(pkl_path_ratings)
        return df
    if target_df == "movies" and os.path.exists(csv_path_movies):
        if not os.path.exists(pkl_path_movies):
            print('read csv...', csv_path_movies)
            df = ml_movies_csv_to_df(csv_path_movies, variant)
            print('save csv.pkl...', pkl_path_movies)
            df.to_pickle(pkl_path_movies)
        else:
            print('read csv.pkl...', pkl_path_movies)
            df = pd.read_pickle(pkl_path_movies)
        return df

    elif os.path.exists(zip_path):
        print('Unzipping data...', zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_dir_path)

        if variant == '10m':
            os.rename(os.path.join(data_dir_path, 'ml-10M100K'),
                      os.path.join(data_dir_path, dirname))

        # for using cache: os.remove(zip_path)

        return fetch_ml_ratings(variant=variant, target_df=target_df)

    else:
        print('Downloading data from', url, 'to', zip_path)
        with urllib.request.urlopen(url) as r, open(zip_path, 'wb') as f:
            shutil.copyfileobj(r, f)

        return fetch_ml_ratings(variant=variant, target_df=target_df)
