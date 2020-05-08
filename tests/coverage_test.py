#!/usr/bin/python
# utf-8
import numpy as np
import pandas as pd
import time
import sys
sys.path.append('./')
from reco_engine import RecommendationAlgorithm

rec_alg = RecommendationAlgorithm(from_pkl=True)


# Получить список всех пользователей
user_ids_list = rec_alg.data_with_user["u_id"].unique()
print(len(user_ids_list))
items_in_rec = {}
user_with_rec = []


# Расчёт охвата
mean_time = 0
def calculate_coverage(users):
    for ep, user in enumerate(users):
        now = time.time()
        recset = rec_alg.get_recommendation(user, if_need_print_time=False)
        if not recset.empty:
            user_with_rec.append(user)
            m_ids_rec = recset["i_id"].values
            for rec in m_ids_rec:
                if rec in items_in_rec:
                    items_in_rec[rec] += 1
                else:
                    items_in_rec[rec] = 1
        t = time.time() - now
        global mean_time
        mean_time += t
        if ep % 10 == 0 and ep != 0 :
            print(mean_time/10)
            mean_time = 0

    print("items_in_rec", items_in_rec, len(items_in_rec.items()))
    no_movies = 27278
    no_movies_in_rec = len(items_in_rec.items())

    no_users = len(user_ids_list)
    no_users_in_rec = len(user_with_rec)

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