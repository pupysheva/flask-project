#!/usr/bin/python
# utf-8
import sys
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append('./')
from model import SVD
from model.pkl_generator import generate_if_need



data_with_user, movies_df, _ = generate_if_need()
# Разбиение на тестовый, тренировочный и валидационных наборы данных
print(datetime.now(), 'Start train...')
print(datetime.now(), 'Progress set 0.01.')
train_data = data_with_user.sample(frac=0.8)
print(datetime.now(), 'self.data_with_user.sample(frac=0.8)')
val_user = data_with_user.drop(train_data.index.tolist()).sample(frac=0.5, random_state=8)
print(datetime.now(), 'self.data_with_user.drop(train_user.index.tolist()).sample(frac=0.5, random_state=8)')
test_data = data_with_user.drop(train_data.index.tolist()).drop(val_user.index.tolist())
print(datetime.now(), 'self.data_with_user.drop(train_user.index.tolist()).drop(val_user.index.tolist())')


print("ТЕСТ НА ДЕЛЕНИЕ НА ВЫБОРКИ по юзерам!!!!!!!!!!")
print(len(np.unique(data_with_user["u_id"])), " - Число пользователей, поставивших оценки")
print(len(np.unique(train_data["u_id"])), " - Число пользователей, попавших в тренировочный набор")
print(len(np.unique(test_data["u_id"])), " - Число пользователей, попавших в тестовый набор")


print("ТЕСТ НА ДЕЛЕНИЕ НА ВЫБОРКИ по фильмам!!!!!!!!!!")
print(len(np.unique(data_with_user["i_id"])), " - Число фильмов в data_with_user")
print(len(np.unique(train_data["i_id"])), " - Число фильмов в тренировочном наборе")
print(len(np.unique(test_data["i_id"])), " - Число фильмов в тестовом наборе")

lr, reg, factors = (0.02, 0.016, 64)
epochs = 10

print(datetime.now(), 'start SVD create')
svd = SVD(learning_rate=lr, regularization=reg, n_epochs=epochs, n_factors=factors,
                  min_rating=0.5, max_rating=5)
print(datetime.now(), 'finish SVD create. Start fit...')


svd.fit(Data=train_data, Data_val=val_user, early_stopping=False, shuffle=False)
print(datetime.now(), 'finish svd.fit. Start predict')


pred = svd.predict(test_data)
print(datetime.now(), 'finish svd.predict. Start mean and sqrt')

mae = mean_absolute_error(test_data["rating"], pred)
rmse = np.sqrt(mean_squared_error(test_data["rating"], pred))
print(datetime.now(), 'finish print results...')
print("Test MAE:  {:.2f}".format(mae))
print("Test RMSE: {:.2f}".format(rmse))
print('{} factors, {} lr, {} reg'.format(factors, lr, reg))
