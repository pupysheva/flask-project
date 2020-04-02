#!/usr/bin/env python
# coding: utf-8

from DIPLOMv1.dataset import fetch_ml_ratings
from DIPLOMv1 import SVD

from sklearn.metrics import mean_absolute_error

#20m dataset
import pandas as pd
import numpy as np
import pickle

import time
now = time.time()
# print("Downloading 20-m movielens data...")
df = fetch_ml_ratings(variant = '20m')

# genre_cols = [
#     "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
#     "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
#     "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
# ]
# movies_cols = ['i_id', 'title', 'release_date', "video_release_date", "imdb_url"] + genre_cols
# movies_df = pd.read_csv('ml-100k/u.item', sep='|', names=movies_cols, encoding='latin-1')
# movies_df['i_id'] = movies_df['i_id'].apply(pd.to_numeric)


movies_df = pd.read_csv('ml-20m/movies.csv',names= ['i_id', 'title', 'genres'], sep=',', encoding='latin-1')
movies_df.drop([0], inplace=True)
movies_df['i_id'] = movies_df['i_id'].apply(pd.to_numeric)







model = df.copy()

print('\n', time.time()-now)


# In[2]:


from sklearn.metrics import mean_squared_error, mean_absolute_error



train = model.sample(frac=0.8)
val = model.drop(train.index.tolist()).sample(frac=0.5, random_state=8)
test = model.drop(train.index.tolist()).drop(val.index.tolist())

iterations = 100

def sample_params():
    lr = np.random.uniform(low = 0.001, high = 0.1,  size = 1)[0]
    reg = np.random.uniform(low = 0.001, high = 0.1,  size = 1)[0]
#     factors = np.random.randint(low = 100, high = 500,  size = 1)[0]
    factors = 64
    return lr, reg, factors


# In[3]:


# lr, reg, factors = (0.007, 0.03, 90)
lr, reg, factors = (0.02, 0.016, 64)
svd = SVD(learning_rate=lr, regularization=reg, n_epochs=200, n_factors=factors,
          min_rating=0.5, max_rating=5)

svd.fit(X=train, X_val=val, early_stopping=True, shuffle=False)

pred = svd.predict(test)
mae = mean_absolute_error(test["rating"], pred)
rmse = np.sqrt(mean_squared_error(test["rating"], pred))
print("Test MAE:  {:.2f}".format(mae))
print("Test RMSE: {:.2f}".format(rmse))
print('{} factors, {} lr, {} reg'.format(factors, lr, reg))


# In[4]:


#Adding our own ratings

n_m = len(model.i_id.unique())

#  Initialize my ratings
my_ratings = np.zeros(n_m)


my_ratings[4993] = 5
my_ratings[1080] = 5
my_ratings[260] = 5
my_ratings[4896] = 5
my_ratings[1196] = 5
my_ratings[1210] = 5
my_ratings[2628] = 5
my_ratings[5378] = 5

print('User ratings:')
print('-----------------')

for i, val in enumerate(my_ratings):
    if val > 0:
        print('Rated %d stars: %s' % (val, movies_df.loc[movies_df.i_id==i].title.values))
print("Adding your recommendations!")
items_id = [item[0] for item in np.argwhere(my_ratings>0)]
ratings_list = my_ratings[np.where(my_ratings>0)]
user_id = np.asarray([0] * len(ratings_list))

user_ratings = pd.DataFrame(list(zip(user_id, items_id, ratings_list)), columns=['u_id', 'i_id', 'rating'])


# In[5]:


try:
    model = model.drop(columns=['timestamp'])
except:
    pass
data_with_user = model.append(user_ratings, ignore_index=True)

train_user = data_with_user.sample(frac=0.8)
val_user = data_with_user.drop(train_user.index.tolist()).sample(frac=0.5, random_state=8)
test_user = data_with_user.drop(train_user.index.tolist()).drop(val_user.index.tolist())


# In[6]:


# lr, reg, factors = (0.007, 0.03, 90)
lr, reg, factors = (0.02, 0.016, 64)
epochs = 10 #epochs = 50

svd = SVD(learning_rate=lr, regularization=reg, n_epochs=epochs, n_factors=factors,
          min_rating=0.5, max_rating=5)

svd.fit(X=train_user, X_val=val_user, early_stopping=False, shuffle=False)#early_stopping=True

pred = svd.predict(test_user)
mae = mean_absolute_error(test_user["rating"], pred)
rmse = np.sqrt(mean_squared_error(test_user["rating"], pred))
print("Test MAE:  {:.2f}".format(mae))
print("Test RMSE: {:.2f}".format(rmse))
print('{} factors, {} lr, {} reg'.format(factors, lr, reg))



now = time.time()
with open ('ml-20m/model_svd.pkl', 'wb') as f:
    pickle.dump(svd, f)
print('\n', time.time()-now) 


now = time.time()
data_with_user.to_pickle('ml-20m/data_with_user.pkl')
print('\n', time.time()-now) 

now = time.time()
movies_df.to_pickle('ml-20m/movies.pkl')
print('\n', time.time()-now) 
