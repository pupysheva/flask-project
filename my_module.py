import pandas as pd
import time
import pickle
from itertools import product


def get_films_rated_by_user(user_id):
    user_ratings = data_with_user[data_with_user.u_id == user_id]
    user_ratings.columns = ['u_id',	'i_id', 'rating']
    rated_df = movies_df[movies_df['i_id'].isin(user_ratings['i_id'])].\
        merge(pd.DataFrame(data_with_user).reset_index(drop=True), how = 'inner', left_on = 'i_id', right_on = 'i_id')
    rated_df = rated_df.loc[rated_df.u_id==user_id].sort_values(by='rating', ascending = False)
    return rated_df

def get_recommendation(user_id):
    now = time.time()
    with open('resources/ml-20m/model_svd.pkl', 'rb') as f:
        svd = pickle.load(f)

    data_with_user = pd.read_pickle('resources/ml-20m/data_with_user.pkl')
    movies_df = pd.read_pickle('resources/ml-20m/movies.pkl')
    print('\n Время чтения:', time.time() - now)

    user_id = [user_id]

    now = time.time()
    all_movies = data_with_user.i_id.unique()
    recommendations = pd.DataFrame(list(product(user_id, all_movies)), columns=['u_id', 'i_id'])

    # Getting predictions for the selected userID
    pred_train = svd.predict(recommendations)
    recommendations['prediction'] = pred_train

    sorted_user_predictions = recommendations.sort_values(by='prediction', ascending=False)
    print(sorted_user_predictions.head(10))

    user_ratings = data_with_user[data_with_user.u_id == user_id[0]]
    user_ratings.columns = ['u_id', 'i_id', 'rating']
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = movies_df[~movies_df['i_id'].isin(user_ratings['i_id'])]. \
        merge(pd.DataFrame(sorted_user_predictions).reset_index(drop=True), how='inner', left_on='i_id',
              right_on='i_id'). \
        sort_values(by='prediction', ascending=False)  # .drop(['i_id'],axis=1)
    print('\nВремя алгоритма', time.time() - now)

    return recommendations.head(20)
