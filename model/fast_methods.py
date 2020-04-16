import numpy as np

from numba import njit, prange


@njit
def _shuffle(Data):
    np.random.shuffle(Data)
    return Data


@njit(parallel=True)
def _initialization(n_user, n_item, n_factors):
    """Инициализация смещений и матриц признаков пользователей и элементов.
    Args:
        n_user (int): количество пользователей
        n_item (int): количество пользователей
        n_factors (int): количество признаков
    Returns:
        user_embeddings (numpy array): матрица признаков пользователей
        movie_embeddings (numpy array): матрица признаков элементов
        user_deviations (numpy array): вектор смециений пользователей
        movie_deviations (numpy array): вектор смециений элемнетов
    """
    user_embeddings = np.random.normal(0, .1, (n_user, n_factors))
    movie_embeddings = np.random.normal(0, .1, (n_item, n_factors))

    user_deviations = np.zeros(n_user)
    movie_deviations = np.zeros(n_item)

    return user_embeddings, movie_embeddings, user_deviations, movie_deviations


@njit(parallel=True)
def _run_epoch(Data, user_embeddings, movie_embeddings, user_deviations, movie_deviations, global_mean, n_factors, lr, reg):
    """Запускает эпоху, обновляя вес модели (user_embeddings, movie_embeddings, user_deviations, movie_deviations).
    Args:
        Data (numpy array): тренировочное множество
        user_embeddings (numpy array): матрица признаков пользователей
        movie_embeddings (numpy array): матрица признаков элементов
        user_deviations (numpy array): вектор смециений пользователей
        movie_deviations (numpy array): вектор смециений элемнетов
        global_mean (float): среднее по всем оценкам
        n_factors (int): количество признаков
        lr (float): скорость обучения
        reg (float): коэффициент регуляризации
    Returns:
        user_embeddings (numpy array): обновленная матрица признаков пользователей
        movie_embeddings (numpy array): обновленная матрица признаков элементов
        user_deviations (numpy array): обновленный вектор смециений пользователей
        movie_deviations (numpy array): обновленный вектор смециений элемнетов
    """
    for i in prange(Data.shape[0]):
        user, item, rating = int(Data[i, 0]), int(Data[i, 1]), Data[i, 2]

        # Предсказание текущей оценки
        pred = global_mean + user_deviations[user] + movie_deviations[item]

        for factor in range(n_factors):
            pred += user_embeddings[user, factor] * movie_embeddings[item, factor]

        err = rating - pred

        # Обновление отклонений
        user_deviations[user] += lr * (err - reg * user_deviations[user])
        movie_deviations[item] += lr * (err - reg * movie_deviations[item])

        # Обновление матриц пользователей и фильмов
        for factor in prange(n_factors):
            puf = user_embeddings[user, factor]
            qif = movie_embeddings[item, factor]

            user_embeddings[user, factor] += lr * (err * qif - reg * puf)
            movie_embeddings[item, factor] += lr * (err * puf - reg * qif)

    return user_embeddings, movie_embeddings, user_deviations, movie_deviations


@njit(parallel=True)
def _compute_val_metrics(Data_val, user_embeddings, movie_embeddings, user_deviations, movie_deviations, global_mean, n_factors):
    """Вычисляет метрики проверки (ошибка, rmse и mae).
    Args:
        Data_val (numpy array): валидационный набор данных
        user_embeddings (numpy array): матрица признаков пользователей
        movie_embeddings (numpy array): матрица признаков элементов
        user_deviations (numpy array): вектор смециений пользователей
        movie_deviations (numpy array): вектор смециений элемнетов
        global_mean (float): среднее по всем оценкам
        n_factors (int): количество признаков
    Returns:
        (tuple of floats): validation loss, rmse and mae.
    """
    residuals = []

    for i in range(Data_val.shape[0]):
        user, item, rating = int(Data_val[i, 0]), int(Data_val[i, 1]), Data_val[i, 2]
        pred = global_mean

        if user > -1:
            pred += user_deviations[user]

        if item > -1:
            pred += movie_deviations[item]

        if (user > -1) and (item > -1):
            for factor in range(n_factors):
                pred += user_embeddings[user, factor] * movie_embeddings[item, factor]

        residuals.append(rating - pred)

    residuals = np.array(residuals)
    loss = np.square(residuals).mean()
    rmse = np.sqrt(loss)
    mae = np.absolute(residuals).mean()

    return (loss, rmse, mae)