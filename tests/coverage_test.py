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
    g_rec_alg = RecommendationAlgorithm(from_pkl=True)
    # Получить список всех пользователей
    g_user_ids_list = g_rec_alg.data_with_user["u_id"].unique()
    print(len(g_user_ids_list))

def pred_thread(rec_alg, users, queue, id_thread):
    user_with_rec = []
    items_in_rec = {}
    now = time.time()
    for ep, user in enumerate(users):
        if ep % os.cpu_count() == id_thread:
            recset = rec_alg.get_recommendation(user, if_need_print_time=False)
            if not recset.empty:
                user_with_rec.append(user)
                for rec in recset["i_id"].values:
                    if rec in items_in_rec:
                        items_in_rec[rec] += 1
                    else:
                        items_in_rec[rec] = 1
            if ep % 100 == 99:
                print(datetime.now(), (time.time() - now) / 100)
                now = time.time()
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

    user_covarage = float(no_users_in_rec / no_users)
    movie_covarage = float(no_movies_in_rec / no_movies)
    return no_movies, no_movies_in_rec, no_users, no_users_in_rec, user_covarage, movie_covarage

def main():
    init()
    now = time.time()
    movies, movies_in_rec, users, users_in_rec, user_covarage, movie_covarage = calculate_coverage(g_user_ids_list)
    print(time.time() - now)

    print(user_covarage, movie_covarage)

    file_covarage = open("./tests/covarage_result.log", "w")
    file_covarage.write("movies: "+str(movies)+"; movies_in_rec:"+str(movies_in_rec)+"\n")
    file_covarage.write("users: "+str(users)+"; users_in_rec:"+str(users_in_rec)+"\n")
    file_covarage.write("user_covarage: "+str(user_covarage)+"; movie_covarage:"+str(movie_covarage)+"\n")

    file_covarage.close()

if __name__ == "__main__":
    main()
