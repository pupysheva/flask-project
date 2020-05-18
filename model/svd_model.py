#!/usr/bin/python
# utf-8
import numpy as np
import time

from .epochs import _calculation_valid_metrics
from .epochs import _init_params
from .epochs import _start_ep
from .timer import timer


class SVD():
    """Реализация алгоритма Simon Funk SVD 
    Attributes:
        lr (float): скорость обучения
        reg (float): коэффициент регуляризации
        n_epochs (int): количество итераций
        n_factors (int): количество признаков
        global_mean (float): среднее по всем оценкам
        user_embeddings (numpy array): матрица признаков пользователей
        movie_embeddings (numpy array): матрица признаков элементов
        user_deviations (numpy array): вектор смециений пользователей
        movie_deviations (numpy array): вектор смециений элемнетов
        early_stopping (boolean): стоит ли прекратить обучение на основе вычисления ошибок на валидационной выборке
        shuffle (boolean): стоит ли перемешивать данные перед каждой эпохой.
    """

    def __init__(self, learning_rate=.02, regularization=0.015, n_epochs=50,
                 n_factors=64, min_rating=0.5, max_rating=5):

        self.lr = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.min_rating = min_rating
        self.max_rating = max_rating

    def _data_conversion(self, Data, is_training=True):
        """Сопоставляет идентификаторы пользователей и элементов с индексами и возвращает массив NumPy.
        Аргументы:
            Data - набор данных для преобразования
            is_training - флаг, который определяет является ли Х обучающим набором
        Возвращает:
            Data (numpy array): сопоставленный набор данных
        """
        Data = Data.copy()

        if is_training:
            u_ids = Data['u_id'].unique().tolist()
            i_ids = Data['i_id'].unique().tolist()

            self.user_dict = dict(zip(u_ids, [i for i in range(len(u_ids))]))
            self.item_dict = dict(zip(i_ids, [i for i in range(len(i_ids))]))

        Data['u_id'] = Data['u_id'].map(self.user_dict)
        Data['i_id'] = Data['i_id'].map(self.item_dict)

        # Tag unknown users/items with -1 (when val)
        Data.fillna(-1, inplace=True)

        Data['u_id'] = Data['u_id'].astype(np.int32)
        Data['i_id'] = Data['i_id'].astype(np.int32)

        Data = Data[['u_id', 'i_id', 'rating']].values

        return Data

    def _stochastic_gradient_descent(self, Data, Data_val, progress):
        """Алгоритм стохастического градиентного спуска
        Args:
            Data (numpy array): обучающий набор, первый столбец должен содержать индексы пользователей, второй - индексы элементов, и третий - рейтинг.
            Data_val (numpy array or `None`): валидационный набор данных
        """
        n_user = len(np.unique(Data[:, 0]))
        n_item = len(np.unique(Data[:, 1]))
        user_embeddings, movie_embeddings, user_deviations, movie_deviations = _init_params(n_user, n_item, self.n_factors)
        for epoch_ix in range(self.n_epochs):
            beginning_time = self._epoch_history(epoch_ix)
            user_embeddings, movie_embeddings, user_deviations, movie_deviations = _start_ep(Data, user_embeddings, movie_embeddings, user_deviations, movie_deviations, self.global_mean,
                                        self.n_factors, self.lr, self.reg)

            if Data_val is not None:
                val_mse, val_rmse, val_mae = _calculation_valid_metrics(Data_val, user_embeddings, movie_embeddings, user_deviations, movie_deviations,
                                                       self.global_mean,
                                                       self.n_factors)
                self._print_metrics(beginning_time, val_mse, val_rmse, val_mae)
            else:
                self._print_metrics(beginning_time)
            progress(float(epoch_ix) / self.n_epochs)
            time.sleep(0.001)

        self.user_embeddings = user_embeddings
        self.movie_embeddings = movie_embeddings
        self.user_deviations = user_deviations
        self.movie_deviations = movie_deviations
        progress(1)

    @timer(text='\nTraining took ')
    def fit(self, Data, Data_val=None, progress=lambda p: None):
        """Обучение модели
        Аргументы:
            Data - обучающий набор
            Data_val - валидационный набор данных
            progress (Callable[[float], None]): функция получения статуса прогресса от 0 до 1
        Возвращает:
            объект Funk SVD: обученная модель
        """
        progress(0.01)
        self.early_stopping = None
        self.shuffle = None
        print('Preprocessing data...\n')
        Data = self._data_conversion(Data)
        progress(0.50)

        if Data_val is not None:
            Data_val = self._data_conversion(Data_val, is_training=False)

        self.global_mean = np.mean(Data[:, 2])
        progress(0.75)
        self._stochastic_gradient_descent(Data, Data_val, lambda p: progress(p * 0.25 + 0.75))
        progress(1)

        return self

    def _predict(self, user_index, item_index):
        """Возвращает прогноз рейтинга модели для данной пары пользователь - элемент.
        Аргументы:
            user_idex - идентификатор пользователя
            item_idex - идентификатор элемента
        Возвращает:
            forecast - рейтинг для данной пары пользователь - элемент
        """
        is_user_known, is_item_known = False, False
        forecast = self.global_mean

        if user_index in self.user_dict:
            is_user_known = True
            cl_ind = self.user_dict[user_index]
            forecast += self.user_deviations[cl_ind]

        if item_index in self.item_dict:
            is_item_known = True
            obj_ind = self.item_dict[item_index]
            forecast += self.movie_deviations[obj_ind]

        if is_user_known and is_item_known:
            forecast += np.dot(self.user_embeddings[cl_ind], self.movie_embeddings[obj_ind])
        return forecast

    def predict(self, Data):
        """Возвращает оценки нескольких заданных пар пользователь - элемент
        Аргументы:
            Data (pandas DataFrame): все пары пользователь - элемент, для которых мы хотим
                предсказывать рейтинги.
        Возврацает:
            forecasts - список с прогнозами для пар пользователь - элемент pairs.
        """
        forecasts = []
        for u_id, i_id in zip(Data['u_id'], Data['i_id']):
            forecasts.append(self._predict(u_id, i_id))
        return forecasts


    def _epoch_history(self, ep_ind):
        beginning = time.time()
        end = '  | ' if ep_ind < 9 else ' | '
        print('Epoch {}/{}'.format(ep_ind + 1, self.n_epochs), end=end)
        return beginning


    def _print_metrics(self, beginning_time, mean_sqrt_error, root_mean_square_error, mean_abs_err):
        end = time.time()
        if mean_sqrt_error is not None:
            print('val_mse: {:.3f}'.format(mean_sqrt_error), end=' - ')
        if root_mean_square_error is not None:
            print('val_rmse: {:.3f}'.format(root_mean_square_error), end=' - ')
        if mean_abs_err is not None:
            print('val_mae: {:.3f}'.format(mean_abs_err), end=' - ')

        print('took {:.1f} sec'.format(end - beginning_time))
