import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error

from model.loading_dataset import fetch_ml_ratings
from model import SVD
from reco_engine import RecommendationAlgorithm as rec_alg

df = fetch_ml_ratings(target_df = "ratings")
model = df.copy()
movies_df = fetch_ml_ratings(target_df = "movies")

# Adding our own ratings
n_m = len(model.i_id.unique())
#  Initialize my ratings
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
        print('Rated %d stars: %s' % (val, movies_df.loc[movies_df.i_id == i].title.values))

print("Adding your recommendations!")
items_id = [item[0] for item in np.argwhere(my_ratings > 0)]
ratings_list = my_ratings[np.where(my_ratings > 0)]
user_id = np.asarray([0] * len(ratings_list))

user_ratings = pd.DataFrame(list(zip(user_id, items_id, ratings_list)), columns=['u_id', 'i_id', 'rating'])

try:
    model = model.drop(columns=['timestamp'])
except:
    pass
data_with_user = model.append(user_ratings, ignore_index=True)

train_user = data_with_user.sample(frac=0.8)
val_user = data_with_user.drop(train_user.index.tolist()).sample(frac=0.5, random_state=8)
test_user = data_with_user.drop(train_user.index.tolist()).drop(val_user.index.tolist())


lr, reg, factors = (0.02, 0.016, 64)
epochs = 10


svd = SVD(learning_rate=lr, regularization=reg, n_epochs=epochs, n_factors=factors,
          min_rating=0.5, max_rating=5)

svd.fit(X=train_user, X_val=val_user, early_stopping=False, shuffle=False)

pred = svd.predict(test_user)
mae = mean_absolute_error(test_user["rating"], pred)
rmse = np.sqrt(mean_squared_error(test_user["rating"], pred))
print("Test MAE:  {:.2f}".format(mae))
print("Test RMSE: {:.2f}".format(rmse))
print('{} factors, {} lr, {} reg'.format(factors, lr, reg))


# Получить список всех пользователей
user_ids_list = df["u_id"].unique()
print(len(user_ids_list))
items_in_rec = {}
user_with_rec = []


# Расчёт охвата

def calculate_coverage(users):
    for ep, user in enumerate(users):
        print(ep, " из ", len(users))
        recset = rec_alg.get_recommendation(user, data_with_user, movies_df)
        if not recset.empty:
            user_with_rec.append(user)
            m_ids_rec = recset["i_id"].values
            print("m_ids_rec", m_ids_rec)
            for rec in m_ids_rec:
                if rec in items_in_rec:
                    items_in_rec[rec] += 1
                else:
                    items_in_rec[rec] = 1

    print("items_in_rec", items_in_rec, len(items_in_rec.items()))
    no_movies = 27278
    no_movies_in_rec = len(items_in_rec.items())

    no_users = len(user_ids_list)
    no_users_in_rec = len(user_with_rec)

    print("no_movies_in_rec  ", no_movies_in_rec)
    print("no_users_in_rec ", no_users_in_rec)

    user_covarage = float(no_users_in_rec / no_users)
    movie_covarage = float(no_movies_in_rec / no_movies)
    return user_covarage, movie_covarage

now = time.time()
user_covarage, movie_covarage = calculate_coverage(user_ids_list)
print(time.time() - now)

print(user_covarage, movie_covarage)
