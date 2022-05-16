#!/usr/bin/env python
# coding: utf-8

# ## Content Recommendations
# Generate personalized recommendations based on the content of movies that the user has previously liked (rated positively). Recommend movies similar to those movies
# 
# Process: 
# - Limit ratings to specified user
# - Normalize ratings for user. If rating < mean, normalized ratings will be negative, which is worse than 0 aka the movie hasn't been rated at all
# - Create user profile: sum of ratings for each attribute
# - Generate recommendations: cosine similarity between movie and user profile 
#     - Do not normalize movie profile. Else penalizing movies with more information 
#     - Ex. Some movies do not have any actors or directors vectors because we excluded actors/directors only in 1 movie since that is not helpful for comparison. If we normalized movies, we would promote movies without actors/directors because their vectors are shorter. 
# - Remove movies already watched/rated from recommendations
# - Limit recommendations to movies listed in keep_movies parameter
#     - Only produce recommendations from a subset of movies (ex with vs without tags)
#     - Still generate user profiles based on ALL movies rated so full look at feature preferences   
#     - Could do this before matrix multiplication to find all similarities, but process to identify and exclude all indices of movies in sparse matrix takes too long. Faster to do at the end. 
# - Sort first on similarity score (prediction) and secondarily on weighted average (first merge with movies_ratings) if same prediction
#       
#       
# Parameters:
# - user_id: ID of user to generate recommendations for
# - df: sparse matrix of movie attributes in one hot encoded fashion
#     - Depending on the attributes included in df, recommendations will be based on those attributes
# - ratings: ratings data for each user (movies rated + star ratings)
# - movieIds: list of all movieIds (rows of sparse matrix)
# - keep_movies: subset of movies (list of movie ids) that we want to limit our recommendations to
#     - This is used in the combination models where we generate recommendations for X movies based on Y features and for the rest of the movies based on Z features
#     - df will still include all movies because want to generate profiles based on all movies. Filter down after generate recommendations 
#     - If [] will default produce all recommendations 
# - movies_ratings: df of movieIds with weighted average of count and average rating. Used to secondarily sort if same prediction from recommendation model
# - df2, keep_movies2, recommendation_system, recommendation_system2, top_n, precision: these are all dummy parameters so that this funciton as the same inputs as other content models. This way we can use the same code in the EvaluationFunctions notebook with the same parameters

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
import sklearn
import fastparquet
import scipy


# In[1]:


def user_content_recommendations(user_id, df, ratings, movieIds, movies_ratings, keep_movies = [],
                                 df2 = False, keep_movies2 = [], recommendation_system = False, 
                                 recommendation_system2 = False, top_n = False, precision = False):
    
    # limit ratings to specific user 
    ratings_user = ratings[ratings.userId == user_id]
    ratings_user = ratings_user.sort_values('movieId')
    # record movies rated/watched
    watched = ratings_user.movieId.unique()
    watched_index = [movieIds.index(i) for i in watched]
    # limit movies dataframe (df) to those rated movies
    movies_user = df[watched_index, :]
        
    # normalize user ratings: subtract mean rating from ratings
        # if rating < mean, normalized rating < 0. Worse than 0 aka not rating the movie at all
    mean_rating = np.mean(ratings_user.rating)
    ratings_user.rating = ratings_user.rating - mean_rating
    
    # generate user profile: multiple item profile by user ratings -> sum of ratings for each movie attribute
    profile = scipy.sparse.csr_matrix(movies_user.T.dot(ratings_user.rating.values))
    
    # normalize profile to account for different numbers of ratings
    profile = sklearn.preprocessing.normalize(profile, axis = 1, norm = 'l2')
    
    # find similarity between profile and movies 
    # cosine similarity *except* movies not normalized 
    recommendations = df.dot(profile.T).todense()
    
    # merge recommendations back with movie Ids
    recommendations = pd.DataFrame(recommendations)
    recommendations = pd.merge(recommendations, pd.Series(movieIds).to_frame(), left_index = True, right_index = True)
    recommendations.columns = ['prediction', 'movieId']
    
    # remove watched movies from recommendations
    recommendations = recommendations[~recommendations.movieId.isin(watched)]
    
    # remove movies not in keep_movies options
    if len(keep_movies) > 0:
        recommendations = recommendations[recommendations.movieId.isin(keep_movies)]
        
    # merge with movie ratings
    recommendations = pd.merge(recommendations, movies_ratings, on = 'movieId')
    
    # sort by similarity and weighted average
    recommendations = recommendations.sort_values(['prediction', 'weighted_avg'], ascending = False)

    return recommendations

