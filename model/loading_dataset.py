#!/usr/bin/python
# utf-8
import datetime
import os
import urllib
import shutil
import zipfile

import numpy as np
import pandas as pd


__all__ = [
    'fetch_ml_ratings',
]

VARIANTS = {
    '100k': {'rating_filename': 'u.data', 'sep': '\t'},
    '20m': {'ratings': {'filename': 'ratings.csv', 'sep': ','},
            'movies': {'filename': 'movies.csv', 'sep': ','}}
}


def ml_ratings_csv_to_df(variant, rating_csv_path):
    # Чтение ratings.csv
    names = ['u_id', 'i_id', 'rating', 'timestamp']
    dtype = {'u_id': np.uint32, 'i_id': np.uint32, 'rating': np.float64}
    def date_parser(time):
        return datetime.datetime.fromtimestamp(float(time))
    ratings_df = pd.read_csv(rating_csv_path, names=names, dtype=dtype, header=0,
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
    dirname = 'ml-' + variant
    ratings_filename = VARIANTS[variant]['ratings']['filename']
    csv_path_ratings = os.path.join(data_dir_path, dirname, ratings_filename)

    movies_filename = VARIANTS[variant]['movies']['filename']
    csv_path_movies = os.path.join(data_dir_path, dirname, movies_filename)

    zip_path = os.path.join(data_dir_path, dirname) + '.zip'
    url = 'http://files.grouplens.org/datasets/movielens/ml-' + variant + \
              '.zip'

    if target_df == "ratings" and os.path.exists(csv_path_ratings):
        if not os.path.exists(csv_path_ratings + '.pkl'):
            print('read csv...')
            df = ml_ratings_csv_to_df(csv_path_ratings, variant)
            print('save csv.pkl...')
            df.to_pickle(csv_path_ratings + '.pkl')
        else:
            print('read csv.pkl...')
            df = pd.read_pickle(csv_path_ratings + '.pkl')
        return df
    if target_df == "movies" and os.path.exists(csv_path_movies):
        if not os.path.exists(csv_path_movies + '.pkl'):
            print('read csv...')
            df = ml_movies_csv_to_df(csv_path_movies, variant)
            print('save csv.pkl...')
            df.to_pickle(csv_path_movies + '.pkl')
        else:
            print('read csv.pkl...')
            df = pd.read_pickle(csv_path_movies + '.pkl')
        return df

    elif os.path.exists(zip_path):
        print('Unzipping data...', zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_dir_path)

        if variant == '10m':
            os.rename(os.path.join(data_dir_path, 'ml-10M100K'),
                      os.path.join(data_dir_path, dirname))

        # for using cache: os.remove(zip_path)

        return fetch_ml_ratings(variant=variant)

    else:
        print('Downloading data...')
        with urllib.request.urlopen(url) as r, open(zip_path, 'wb') as f:
            shutil.copyfileobj(r, f)

        return fetch_ml_ratings(variant=variant, target_df=target_df)
