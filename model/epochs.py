#!/usr/bin/python
# utf-8
import numpy as np

from numba import njit, prange

@njit(parallel=True)
def _init_params(coun_users, count_items, count_features):
    """Инициализация смещений и матриц признаков пользователей и элементов.
    Аргументы:
        coun_users (int): количество пользователей
        count_items (int): количество пользователей
        count_features (int): количество признаков
    Возвращает:
        user_embeddings - матрица признаков пользователей
        movie_embeddings - матрица признаков элементов
        user_deviations - вектор смециений пользователей
        movie_deviations - вектор смециений элемнетов
    """
    user_embeddings = np.random.normal(0, .01, (coun_users, count_features))
    movie_embeddings = np.random.normal(0, .01, (count_items, count_features))

    user_deviations = np.zeros(coun_users)
    movie_deviations = np.zeros(count_items)

    return user_embeddings, movie_embeddings, user_deviations, movie_deviations


@njit(parallel=True)
def _start_ep(Data, user_embeddings, movie_embeddings, user_deviations, movie_deviations, g_average, count_features, learning_rate, reg):
    """Запускает эпоху, обновляя вес модели (user_embeddings, movie_embeddings, user_deviations, movie_deviations).
    Аргументы:
        Data (numpy array): тренировочное множество
        user_embeddings (numpy array): матрица признаков пользователей
        movie_embeddings (numpy array): матрица признаков элементов
        user_deviations (numpy array): вектор смециений пользователей
        movie_deviations (numpy array): вектор смециений элемнетов
        g_average (float): среднее по всем оценкам
        count_features (int): количество признаков
        learning_rate (float): скорость обучения
        reg (float): коэффициент регуляризации
    Возвращает:
        user_embeddings - обновленная матрица признаков пользователей
        movie_embeddings - обновленная матрица признаков элементов
        user_deviations - обновленный вектор смециений пользователей
        movie_deviations - обновленный вектор смециений элемнетов
    """
    for i in prange(Data.shape[0]):
        cl_ind, object_ind, target_r = int(Data[i, 0]), int(Data[i, 1]), Data[i, 2]

        # Предсказание текущей оценки
        forecast = g_average + user_deviations[cl_ind] + movie_deviations[object_ind]

        for feature in range(count_features):
            forecast += user_embeddings[cl_ind, feature] * movie_embeddings[object_ind, feature]

        err = target_r - forecast

        # Обновление отклонений
        user_deviations[cl_ind] += learning_rate * (err - reg * user_deviations[cl_ind])
        movie_deviations[object_ind] += learning_rate * (err - reg * movie_deviations[object_ind])

        # Обновление матриц пользователей и фильмов
        for feature in prange(count_features):
            puf = user_embeddings[cl_ind, feature]
            qif = movie_embeddings[object_ind, feature]

            user_embeddings[cl_ind, feature] += learning_rate * (err * qif - reg * puf)
            movie_embeddings[object_ind, feature] += learning_rate * (err * puf - reg * qif)

    return user_embeddings, movie_embeddings, user_deviations, movie_deviations


@njit(parallel=True)
def _calculation_valid_metrics(Data_val, user_embeddings, movie_embeddings, user_deviations, movie_deviations, g_average, count_features):
    """Вычисляет метрики проверки (ошибка, rmse и mae).
    Аргументы:
        Data_val (numpy array): валидационный набор данных
        user_embeddings (numpy array): матрица признаков пользователей
        movie_embeddings (numpy array): матрица признаков элементов
        user_deviations (numpy array): вектор смециений пользователей
        movie_deviations (numpy array): вектор смециений элемнетов
        g_average (float): среднее по всем оценкам
        count_features (int): количество признаков
    Возвращает:
        Кортеж - mean_sqrt_error, root_mean_square_error and mean_abs_err.
    """
    errors_for_test = []
    for i in range(Data_val.shape[0]):
        cl_ind, object_ind, target_r = int(Data_val[i, 0]), int(Data_val[i, 1]), Data_val[i, 2]
        forecast = g_average
        if cl_ind > -1:
            forecast += user_deviations[cl_ind]
        if object_ind > -1:
            forecast += movie_deviations[object_ind]
        if (cl_ind > -1) and (object_ind > -1):
            for feature in range(count_features):
                forecast += user_embeddings[cl_ind, feature] * movie_embeddings[object_ind, feature]
        errors_for_test.append(target_r - forecast)

    errors_for_test = np.array(errors_for_test)
    mean_sqrt_error = np.square(errors_for_test).mean()
    root_mean_square_error = np.sqrt(mean_sqrt_error)
    mean_abs_err = np.absolute(errors_for_test).mean()
    return (mean_sqrt_error, root_mean_square_error, mean_abs_err)
