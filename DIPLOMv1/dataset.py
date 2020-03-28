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
    '100k': {'rating_filename': 'u.data', 'sep':'\t'},
    '20m': {'rating_filename': 'ratings.csv', 'sep':','}
}


def ml_ratings_csv_to_df(csv_path, variant):
    names = ['u_id', 'i_id', 'rating', 'timestamp']
    dtype = {'u_id': np.uint32, 'i_id': np.uint32, 'rating': np.float64}

    def date_parser(time):
        return datetime.datetime.fromtimestamp(float(time))

    df = pd.read_csv(csv_path, names=names, dtype=dtype, header=0,
                     sep=VARIANTS[variant]['sep'], parse_dates=['timestamp'],
                     date_parser=date_parser, engine='python')

    df.sort_values(by='timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def fetch_ml_ratings(data_dir_path=None, variant='20m'):
    if data_dir_path is None:
        data_dir_path = './'
        dirname = 'ml-' + variant
        filename = VARIANTS[variant]['rating_filename']
        csv_path = os.path.join(data_dir_path, dirname, filename)
        zip_path = os.path.join(data_dir_path, dirname) + '.zip'
        url = 'http://files.grouplens.org/datasets/movielens/ml-' + variant + \
              '.zip'
    else:
        csv_path = data_dir_path

    if os.path.exists(csv_path):
        if not os.path.exists(csv_path + '.pkl'):
            print('read csv...')
            df = ml_ratings_csv_to_df(csv_path, variant)
            print('save csv.pkl...')
            df.to_pickle(csv_path + '.pkl')
        else:
            print('read csv.pkl...')
            df = pd.read_pickle(csv_path + '.pkl')
     
        return df

    elif os.path.exists(zip_path):
        print('Unzipping data...')

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

        return fetch_ml_ratings(variant=variant)