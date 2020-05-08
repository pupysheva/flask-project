#!/usr/bin/python
# utf-8
import numpy as np
import pandas as pd
import time
import sys
sys.path.append('./')
from reco_engine import RecommendationAlgorithm
from multiprocessing import Pool
import os
import threading
from datetime import datetime

rec_alg = RecommendationAlgorithm(from_pkl=True)


# Получить список всех пользователей
user_ids_list = rec_alg.data_with_user["u_id"].unique()
print(len(user_ids_list))
g_items_in_rec = {}
g_user_with_rec = []
g_users = None

lock = threading.Lock()

def pred_thread(id_thread):
    global g_users
    global g_user_with_rec
    global g_items_in_rec
    users = g_users
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
    lock.acquire()
    print(time.time(), 'merge results by thread', id_thread)
    g_user_with_rec.extend(user_with_rec)
    for key, value in items_in_rec.items():
        if key in g_items_in_rec:
            g_items_in_rec[key] += value
        else:
            g_items_in_rec[key] = value
    lock.release()

def calculate_coverage(users):
    global g_users
    global g_user_with_rec
    global g_items_in_rec
    g_users = users
    with Pool(os.cpu_count()) as p:
        p.map(pred_thread, range(os.cpu_count()))

    print("g_items_in_rec", g_items_in_rec, len(g_items_in_rec.items()))
    no_movies = 27278
    no_movies_in_rec = len(g_items_in_rec.items())

    no_users = len(user_ids_list)
    no_users_in_rec = len(g_user_with_rec)

    print("no_movies_in_rec  ", no_movies_in_rec)
    print("no_users_in_rec ", no_users_in_rec)

    user_covarage = float(no_users_in_rec / no_users)
    movie_covarage = float(no_movies_in_rec / no_movies)
    return no_movies, no_movies_in_rec, no_users, no_users_in_rec, user_covarage, movie_covarage

now = time.time()
movies, movies_in_rec, users, users_in_rec, user_covarage, movie_covarage = calculate_coverage(user_ids_list)
print(time.time() - now)

print(user_covarage, movie_covarage)

file_covarage = open("./tests/covarage_result.txt", "w")
file_covarage.write("movies: "+str(movies)+"; movies_in_rec:"+str(movies_in_rec)+"\n")
file_covarage.write("users: "+str(users)+"; users_in_rec:"+str(users_in_rec)+"\n")
file_covarage.write("user_covarage: "+str(user_covarage)+"; movie_covarage:"+str(movie_covarage)+"\n")

file_covarage.close()
