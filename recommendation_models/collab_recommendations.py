#!/usr/bin/env python
# coding: utf-8

# # Collaborative Filtering Recommendations
# 
# Process: 
# - Input precomputed rating predictions 
# - Limit to user
# - Merge with movie ratings to get weighted average of ratings of each movie
# - Sort first on similarity score (prediction) and secondarily on weighted average (first merge with movies_ratings) if same prediction
# 
# If precision = True, then instead of using precomputed predictions, fit KNN baseline model on train ratings data and then generate predictions for test ratings data. Produce test predictions for calculating precision and recall. 
#     
# Parameters:
# - user_id: ID of user to generate recommendations for
# - ratings: ratings data for each user (movies rated + star ratings)
# - movies_ratings: df of movieIds with weighted average of count and average rating. Used to secondarily sort if same prediction from recommendation model
# - df2: 
#     - precision = False: pregenerated collaborative filtering predictions.
#     - precision = True: test set of ratings data to generate predicitons on for precision, recall calculation
# - keep_movies1, keep_movies2, : [] -- dummy parameter so that this funciton as the same inputs as the other recommendation models
# - df1, movieIds, content_recommendation_system, collab_recommenation_system = False: dummy parameters so that this funciton as the same inputs as the other recommendation models

# In[ ]:


import pandas as pd
import os
import numpy as np
import datetime as datetime
import operator
import scipy.spatial.distance as distance
from sklearn import metrics 
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import fastparquet
import pickle
import scipy
import sklearn
from surprise import SVD, Dataset, Reader, KNNBaseline


# In[1]:


def collab_recommendations(user_id, df1, ratings, movieIds, movies_ratings, keep_movies1, df2,
                           keep_movies2, content_recommedation_system = False, collab_recommendation_system = False,
                           top_n = 10, precision = False):
    
    # generate recommendations on train/test set
    if precision: 
        test_ratings = df2.copy()
        # set parameters for KNN model
        user_based = {'name': 'pearson_baseline',
               'shrinkage': 0  # no shrinkage
               }
        collab_ratings = ratings[['userId','movieId','rating']]
        # set scale between min and max rating 
        min_rat = collab_ratings.rating.min()
        max_rat = collab_ratings.rating.max()
        reader = Reader(rating_scale=(min_rat,max_rat))
        # fit on train set
        data = Dataset.load_from_df(collab_ratings, reader)
        trainset = data.build_full_trainset()
        algo = KNNBaseline(sim_options=user_based)
        algo.fit(trainset)

        # predict on test set
        test_ratings = test_ratings[['userId','movieId','rating']]
        testset = [tuple(x) for x in test_ratings.to_numpy()]
        predictions = algo.test(testset)
        
        # return predictions on test set 
        collab_predictions = pd.DataFrame(predictions)
        collab_predictions=collab_predictions[['uid','iid','est']]
        collab_predictions= collab_predictions.rename(columns = {'est':'prediction', 'uid':'userId', 'iid':'movieId'}
                                                     )[['userId','movieId','prediction']]
        collab_predictions[['userId','movieId']] = collab_predictions[['userId','movieId']].astype(int)
        
    # use precomputed
    else:
        collab_predictions = df2.copy()

    # get recommendations from collab filtering model 
    collab_rec = collab_predictions[collab_predictions.userId == user_id]
    # merge with movie ratings + sort on prediction and secondarily on weighted average of ratings
    collab_rec = pd.merge(collab_rec, movies_ratings, on = 'movieId')
    collab_rec = collab_rec.sort_values(['prediction', 'weighted_avg'], ascending = [False, True])
    
    return collab_rec

