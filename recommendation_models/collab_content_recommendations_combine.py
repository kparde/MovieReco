#!/usr/bin/env python
# coding: utf-8

# # Combine collaborative filtering and content recommendation results
# 
# - Produce half recommendations from collaborative filtering and half from the content model (half based on specified number of recommendations top_n)
# - Recommendations sorted on prediction (similarity score for content, predicted rating for collab) and secondarily on weighted average of ratings if prediction is the same
# - Content model recommendations out of movies not included in collaborative filtering recommendations _for that user_ 
# - Re-sort based on weighted average of movie ratings such that we produce the most "credible"/recognizable results first to gain the user's trust before presenting long tail recommendations
# 
# Parameters:
# - user_id: ID of user to generate recommendations for
# - df1: sparse matrix of movie attributes in one hot encoded fashion with attributes from for content model 
# - ratings: ratings data for each user (movies rated + star ratings)
# - movieIds: list of all movie Ids (rows of sparse matrix)
# - movies_ratings: df of movieIds with weighted average of count and average rating. Used to secondarily sort if same prediction from recommendation model
# - df2: 
#     - precision = False: pregenerated collaborative filtering predictions.
#     - precision = True: test set of ratings data to generate predicitons on for precision, recall calculation
# - keep_movies1, keep_movies2: [] -- dummy parameter so that this funciton as the same inputs as the other recommendation models
# - content_recommendation_system: recommendation system to use to generate recs for content model
#     - Module of a function in another script
# - top_n: number of recommendations total to produce
# - precision: True if want to generate test rating predictions from collaboartive filtering rather than using precomputed predictions. Else False. 

# In[1]:


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


# In[82]:


# not using keep_movies1 or keep_movies2 
# collab_predictions as df2
def collab_content_combine(user_id, df1, ratings, movieIds, movies_ratings, keep_movies1, df2,
                           keep_movies2, content_recommendation_system, collab_recommendation_system,
                           top_n = 10, precision = False):
    
    collab_rec = collab_recommendation_system(user_id, df1, ratings, movieIds, movies_ratings, keep_movies1, df2,
                                              keep_movies2, content_recommendation_system, top_n = 10, precision = precision)

    # find movies in full set that are not in collaborative filtering predictions for this user
    keep_movies = set(movieIds).difference(set(collab_rec.movieId.unique()))
    
    # generate recommendations from content model with movies not in collab filtering
    content_rec = content_recommendation_system(user_id, df1, ratings, movieIds, movies_ratings, keep_movies)
    
    # concat half top recommendations from each model 
    recommendations = pd.concat([collab_rec.head(int(top_n/2)), content_rec.head(int(top_n/2))])
    
    # resort based on weighted average: present popular movies first 
    recommendations = recommendations.sort_values('weighted_avg', ascending = False)
    
    return recommendations

