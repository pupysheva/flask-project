#!/usr/bin/python
# utf-8
import time
import sys
import os
from multiprocessing import Process, Queue
sys.path.append('./')
from reco_engine import RecommendationAlgorithm
from logger import log


def init():
    g_rec_alg = RecommendationAlgorithm(from_pkl=True)
    # Получить список всех пользователей
    g_user_ids_list = g_rec_alg.data_with_user["u_id"].unique()
    log('len(g_user_ids_list): {} users'.format(len(g_user_ids_list)), init)
    return g_rec_alg, g_user_ids_list


def pred_thread(rec_alg, users, queue, id_thread):
    log('start thread № {}'.format(id_thread), pred_thread)
    user_with_rec = []
    items_in_rec = {}
    now = time.time()
    for ep, user in enumerate(users):
        if ep % os.cpu_count() == id_thread:
            rec_set = rec_alg.get_recommendation(user, if_need_print_time=False)
            if not rec_set.empty:
                user_with_rec.append(user)
                for rec in rec_set["i_id"].values:
                    if rec in items_in_rec:
                        items_in_rec[rec] += 1
                    else:
                        items_in_rec[rec] = 1
        if ep % 1000 == 999:
            log('[{}]: {:>5.1f} % {:>8.6f} seconds/user'.format(id_thread, ep * 100.0 / len(users), (time.time() - now) / 1000), pred_thread)
            now = time.time()
    log('finish thread № {}'.format(id_thread), pred_thread)
    queue.put((user_with_rec, items_in_rec))


def calculate_coverage(g_rec_alg, g_user_ids_list):
    log('start', calculate_coverage)
    g_items_in_rec = {}
    g_user_with_rec = []
    q = Queue()
    for i in range(os.cpu_count()):
        p = Process(target=pred_thread, args=(g_rec_alg, g_user_ids_list, q, i))
        p.start()
    for i in range(os.cpu_count()):
        (user_with_rec, items_in_rec) = q.get()
        g_user_with_rec.extend(user_with_rec)
        for key, value in items_in_rec.items():
            if key in g_items_in_rec:
                g_items_in_rec[key] += value
            else:
                g_items_in_rec[key] = value
    log({'g_items_in_rec': g_items_in_rec, 'len(g_items_in_rec)': len(g_items_in_rec)}, calculate_coverage)
    no_movies = 27278
    no_movies_in_rec = len(g_items_in_rec)

    no_users = len(g_user_ids_list)
    no_users_in_rec = len(g_user_with_rec)

    log({'no_movies_in_rec': no_movies_in_rec, 'no_users_in_rec': no_users_in_rec}, calculate_coverage)

    user_coverage = float(no_users_in_rec / no_users)
    movie_coverage = float(no_movies_in_rec / no_movies)
    log('finish', calculate_coverage)
    return no_movies, no_movies_in_rec, no_users, no_users_in_rec, user_coverage, movie_coverage


def main():
    g_rec_alg, g_user_ids_list = init()
    now = time.time()
    movies, movies_in_rec, users, users_in_rec, user_coverage, movie_coverage = calculate_coverage(g_rec_alg, g_user_ids_list)
    log(time.time() - now, main)
    output = {'movies': str(movies),
              'movies_in_rec': str(movies_in_rec),
              'users': str(users),
              'users_in_rec': str(users_in_rec),
              'user_coverage': str(user_coverage),
              'movie_coverage': str(movie_coverage)}
    log(output, main)
    with open("./tests/coverage_result.log", "w") as file_coverage:
        file_coverage.write('\n'.join(['{}: {}'.format(key, value) for (key, value) in output.items()]))


if __name__ == "__main__":
    main()
