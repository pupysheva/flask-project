#!/usr/bin/python
# utf-8
import sys
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
sys.path.append('./')
from model import SVD
from model.pkl_generator import generate_if_need

ratings, movies_df, _ = generate_if_need()
# Разбиение на тестовый, тренировочный и валидационных наборы данных
print(datetime.now(), 'Start train...')
print(datetime.now(), 'Progress set 0.01.')
train_data = ratings.sample(frac=0.8)
print(datetime.now(), 'self.data_with_user.sample(frac=0.8)')
val_data = ratings.drop(train_data.index.tolist()).sample(frac=0.5, random_state=8)
print(datetime.now(), 'self.data_with_user.drop(train_user.index.tolist()).sample(frac=0.5, random_state=8)')
test_data = ratings.drop(train_data.index.tolist()).drop(val_data.index.tolist())
print(datetime.now(), 'self.data_with_user.drop(train_user.index.tolist()).drop(val_user.index.tolist())')

print(len(np.unique(ratings["u_id"])), " - Число пользователей, поставивших оценки")
print(len(np.unique(train_data["u_id"])), " - Число пользователей, попавших в тренировочный набор")
print(len(np.unique(test_data["u_id"])), " - Число пользователей, попавших в тестовый набор")
print(len(np.unique(ratings["i_id"])), " - Число фильмов в data_with_user")
print(len(np.unique(train_data["i_id"])), " - Число фильмов в тренировочном наборе")
print(len(np.unique(test_data["i_id"])), " - Число фильмов в тестовом наборе")

lr, reg, factors = (0.02, 0.015, 64)
epochs = 10
print(datetime.now(), 'start SVD create')
svd = SVD(learning_rate=lr, regularization=reg, n_epochs=epochs, n_factors=factors,
                  min_rating=0.5, max_rating=5)
print(datetime.now(), 'finish SVD create. Start fit...')
svd.fit(Data=train_data, Data_val=val_data)
print(datetime.now(), 'finish svd.fit. Start predict')
pred = svd.predict(test_data)
print(datetime.now(), 'finish svd.predict. Start mean and sqrt')
mae = mean_absolute_error(test_data["rating"], pred)
mse = mean_squared_error(test_data["rating"], pred)
rmse = np.sqrt(mse)
print(datetime.now(), 'finish print results...')
print("Test MAE Funk SVD:  {:.2f}".format(mae))
print("Test MSE Funk SVD: {:.2f}".format(mse))
print("Test RMSE Funk SVD: {:.2f}".format(rmse))
print('{} factors, {} lr, {} reg'.format(factors, lr, reg))


# Тривиальный алгоритм
df_means_ratings_for_movies = ratings.groupby(['i_id'], as_index=False)['rating'].mean()
df_means_ratings_for_movies.rename(columns={'rating': 'mean_rating'}, inplace=True)
pred_trivial = df_means_ratings_for_movies.merge(test_data.reset_index(drop=True), how='inner',
                                                 left_on='i_id',  right_on='i_id')
mae = mean_absolute_error(pred_trivial["rating"], pred_trivial["mean_rating"])
mse = mean_squared_error(pred_trivial["rating"], pred_trivial["mean_rating"])
rmse = np.sqrt(mse)
print("Test MAE Trivial:  {:.2f}".format(mae))
print("Test MSE Funk SVD: {:.2f}".format(mse))
print("Test RMSE Trivial: {:.2f}".format(rmse))


