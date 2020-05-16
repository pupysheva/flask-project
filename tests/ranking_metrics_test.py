#!/usr/bin/python
# utf-8
import numpy as np
import pandas as pd
import time
import sys
sys.path.append('./')
from reco_engine import RecommendationAlgorithm
import os
from datetime import datetime
from multiprocessing import Process, Queue

def init():
    global g_rec_alg
    global g_user_ids_list
    global mean_rating_users
    g_rec_alg = RecommendationAlgorithm(from_pkl=True)
    g_rec_alg.train_model()
    # Получить список всех пользователей
    g_user_ids_list = g_rec_alg.data_with_user["u_id"].unique()
    print(len(g_user_ids_list))

    # средние оценки по юзерам
    mean_rating_users = g_rec_alg.train_user.groupby(['u_id'], as_index=False)['rating'].mean()

def pred_thread(rec_alg, users, queue, id_thread):
    now = time.time()
    for ep, user in enumerate(users):
        if ep % os.cpu_count() == id_thread:
            recs_df = rec_alg.get_recommendation(user)
            t = g_rec_alg.test_user
            temp = t[t[""]>]
    print(datetime.now(), 'finish tread', id_thread)
    queue.put((user_with_rec, items_in_rec))

def calculate_coverage(users):
    g_items_in_rec = {}
    g_user_with_rec = []
    q = Queue()
    for i in range(os.cpu_count()):
        p = Process(target=pred_thread, args=(g_rec_alg, users, q, i))
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
    return no_movies, no_movies_in_rec, no_users, no_users_in_rec, user_coverage, movie_coverage

def main():
    init()
    now = time.time()
    movies, movies_in_rec, users, users_in_rec, user_coverage, movie_coverage = calculate_coverage(g_user_ids_list)
    print(time.time() - now)

    print(user_coverage, movie_coverage)

    file_coverage = open("./tests/coverage_result.log", "w")
    file_coverage.write("movies: "+str(movies)+"; movies_in_rec:"+str(movies_in_rec)+"\n")
    file_coverage.write("users: "+str(users)+"; users_in_rec:"+str(users_in_rec)+"\n")
    file_coverage.write("user_coverage: "+str(user_coverage)+"; movie_coverage:"+str(movie_coverage)+"\n")

    file_coverage.close()

if __name__ == "__main__":
    main()
