#!/usr/bin/python
# utf-8
import numpy as np
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
    train_data, test_data = g_rec_alg.train_model()
    # Получить список всех пользователей
    g_user_ids_list_for_ped = test_data["u_id"].unique()
    print(len(g_user_ids_list_for_ped))
    # средние оценки по юзерам
    mean_rating_users = train_data.groupby(['u_id'], as_index=False)['rating'].mean()
    return g_rec_alg, g_user_ids_list_for_ped, mean_rating_users, test_data


def pred_thread(g_rec_alg, g_user_ids_list_for_ped, mean_rating_users, test_data, queue, id_thread):
    print('{} start thread {} VIRT: {:>6.0f} MiB SWAP: {:>7.0f} MiB'.format(datetime.now(), id_thread, virtual_memory().used / 2**20, swap_memory().used / 2**20))
    precision_list = []
    recall_list = []
    now = time.time()
    for ep, user in enumerate(g_user_ids_list_for_ped):
        if ep % os.cpu_count() == id_thread:
            mean_u = float(mean_rating_users[mean_rating_users["u_id"] == user]["rating"])
            test_ratings = test_data[test_data["u_id"] == user]
            id_films_liked_by_u = test_ratings[test_ratings["rating"] > mean_u]["i_id"].unique()

            pred_for_u = g_rec_alg.get_recommendation(user, if_need_print_time=False)["i_id"].unique()

            intersection = len(list(set(id_films_liked_by_u) & set(pred_for_u)))

            if len(id_films_liked_by_u) != 0:
                recall = intersection / len(id_films_liked_by_u)
                recall_list.append(recall)
                precision = intersection / len(pred_for_u)
                precision_list.append(precision)
            if ep % 1000 == 999:
                print('{} {:>5.1f}% VIRT: {:>6.0f} MiB SWAP: {:>7.0f} MiB {:>8.6f}'.format(datetime.now(), ep * 100.0 / len(users), virtual_memory().used / 2**20, swap_memory().used / 2**20, (time.time() - now) / 1000))
                now = time.time()

    print('{} finish thread {} VIRT: {:>6.0f} MiB SWAP: {:>7.0f} MiB'.format(datetime.now(), id_thread, virtual_memory().used / 2**20, swap_memory().used / 2**20))
    queue.put((precision_list, recall_list))


def calculate_precision_recall(g_rec_alg, g_user_ids_list_for_ped, mean_rating_users, test_data):
    print('{} start calculate_precision_recall VIRT: {:>6.0f} MiB SWAP: {:>7.0f} MiB'.format(datetime.now(), virtual_memory().used / 2**20, swap_memory().used / 2**20))
    q = Queue()
    g_pres_list = []
    g_recall_list = []
    for i in range(os.cpu_count()):
        p = Process(target=pred_thread, args=(g_rec_alg, g_user_ids_list_for_ped, mean_rating_users, test_data, q, i))
        p.start()
    for i in range(os.cpu_count()):
        (precision_list, recall_list) = q.get()

        g_pres_list.extend(precision_list)
        g_recall_list.extend(recall_list)

        precision_mean = np.array(g_pres_list).mean()
        recall_mean = np.array(g_recall_list).mean()
    print('{} finish calculate_precision_recall VIRT: {:>6.0f} MiB SWAP: {:>7.0f} MiB'.format(datetime.now(), virtual_memory().used / 2**20, swap_memory().used / 2**20))
    return precision_mean, recall_mean


def main():
    print('{} main ranking_metrics total VIRT: {:>6.0f} MiB total SWAP: {:>7.0f} MiB used VIRT: {:>6.0f} MiB used SWAP: {:>7.0f} MiB threads: {} {} {}'.format(datetime.now(), virtual_memory().total / 2**20, swap_memory().total / 2**20, virtual_memory().used / 2**20, swap_memory().used / 2**20, os.cpu_count(), platform.system(), platform.release()))
    g_rec_alg, g_user_ids_list_for_ped, mean_rating_users, test_data = init()
    now = time.time()
    precision, recall = calculate_precision_recall(g_rec_alg, g_user_ids_list_for_ped, mean_rating_users, test_data)
    print(time.time() - now)

    print(precision, recall)

    file_p_r = open("./tests/ranking_metrics.log", "w")
    file_p_r.write("precision: "+str(precision))
    file_p_r.write("recall: "+str(recall))
    file_p_r.close()


if __name__ == "__main__":
    main()



# g_rec_alg = RecommendationAlgorithm(from_pkl=True)
# g_rec_alg.train_model(if_progress_need = False)
# precision_list = []
# recall_list = []
# # средние оценки по юзерам
# mean_rating_users = g_rec_alg.train_data.groupby(['u_id'], as_index=False)['rating'].mean()
# g_user_ids_list_for_ped = g_rec_alg.test_data["u_id"].unique()
# for j, user in enumerate(g_user_ids_list_for_ped):
#     now = time.time()
#     mean_u = float(mean_rating_users[mean_rating_users["u_id"] == user]["rating"])
#     # print("mean_u", mean_u)
#     test_ratings = g_rec_alg.test_data[g_rec_alg.test_data["u_id"] == user]
#     id_films_liked_by_u = test_ratings[test_ratings["rating"] > mean_u]["i_id"].unique()
#     # print("test_ratings", len(id_films_liked_by_u), test_ratings)
#
#
#     pred_for_u = g_rec_alg.get_recommendation(user, if_need_print_time=False)["i_id"].unique()
#     # print("pred_for_u", len(pred_for_u), pred_for_u)
#
#     intersection = len(list(set(id_films_liked_by_u) & set(pred_for_u)))
#     print(intersection)
#
#
#     recall = "нет"
#     precision = "нет"
#     if len(id_films_liked_by_u) != 0:
#         recall = intersection/len(id_films_liked_by_u)
#         recall_list.append(recall)
#         precision = intersection/len(pred_for_u)
#         precision_list.append(precision)
#     print(user, recall, precision, time.time() - now)


