#!/usr/bin/python
# utf-8
import numpy as np
import pandas as pd
import time
import sys
import os
from datetime import datetime
from multiprocessing import Process, Queue
sys.path.append('./')
from reco_engine import RecommendationAlgorithm
from psutil import virtual_memory, swap_memory
import platform


def init():
    g_rec_alg = RecommendationAlgorithm(from_pkl=True)
    # Получить список всех пользователей
    g_user_ids_list = g_rec_alg.data_with_user["u_id"].unique()
    print(len(g_user_ids_list))
    return g_rec_alg, g_user_ids_list


def pred_thread(rec_alg, users, queue, id_thread):
    print('{} start thread {} VIRT: {:>6.0f} MiB SWAP: {:>7.0f} MiB'.format(datetime.now(), id_thread, virtual_memory().used / 2**20, swap_memory().used / 2**20))
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
                print('{} {:>5.1f}% VIRT: {:>6.0f} MiB SWAP: {:>7.0f} MiB {:>8.6f}'.format(datetime.now(), ep * 100.0 / len(users), virtual_memory().used / 2**20, swap_memory().used / 2**20, (time.time() - now) / 1000))
                now = time.time()
    print('{} finish thread {} VIRT: {:>6.0f} MiB SWAP: {:>7.0f} MiB'.format(datetime.now(), id_thread, virtual_memory().used / 2**20, swap_memory().used / 2**20))
    queue.put((user_with_rec, items_in_rec))


def calculate_coverage(g_rec_alg, g_user_ids_list):
    print('{} start calculate_coverage VIRT: {:>6.0f} MiB SWAP: {:>7.0f} MiB'.format(datetime.now(), virtual_memory().used / 2**20, swap_memory().used / 2**20))
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
    print("g_items_in_rec", g_items_in_rec, len(g_items_in_rec.items()))
    no_movies = 27278
    no_movies_in_rec = len(g_items_in_rec.items())

    no_users = len(g_user_ids_list)
    no_users_in_rec = len(g_user_with_rec)

    print("no_movies_in_rec  ", no_movies_in_rec)
    print("no_users_in_rec ", no_users_in_rec)

    user_coverage = float(no_users_in_rec / no_users)
    movie_coverage = float(no_movies_in_rec / no_movies)
    print('{} finish calculate_coverage VIRT: {:>6.0f} MiB SWAP: {:>7.0f} MiB'.format(datetime.now(), virtual_memory().used / 2**20, swap_memory().used / 2**20))
    return no_movies, no_movies_in_rec, no_users, no_users_in_rec, user_coverage, movie_coverage


def main():
    print('{} main coverage total VIRT: {:>6.0f} MiB total SWAP: {:>7.0f} MiB threads: {} {} {}'.format(datetime.now(), virtual_memory().total / 2**20, swap_memory().total / 2**20, os.cpu_count(), platform.system(), platform.release()))
    g_rec_alg, g_user_ids_list = init()
    now = time.time()
    movies, movies_in_rec, users, users_in_rec, user_coverage, movie_coverage = calculate_coverage(g_rec_alg, g_user_ids_list)
    print(time.time() - now)

    print(user_coverage, movie_coverage)

    file_coverage = open("./tests/coverage_result.log", "w")
    file_coverage.write("movies: "+str(movies)+"; movies_in_rec:"+str(movies_in_rec)+"\n")
    file_coverage.write("users: "+str(users)+"; users_in_rec:"+str(users_in_rec)+"\n")
    file_coverage.write("user_coverage: "+str(user_coverage)+"; movie_coverage:"+str(movie_coverage)+"\n")

    file_coverage.close()


if __name__ == "__main__":
    main()
