#!/usr/bin/env python
# coding: utf-8

# # Main Streamlit HomePage
# 5 pages:
# - Visualizations   
# - Top rated movies with filtering 
# - Item-Item recommendations
# - Personalized recommendations
# - User profile creation: provide ratings    
#    
# Process:
# 1. Import pages as modules
# 2. Set up data with cached functions
#     - Call data functions from the various page modules so that all loaded in once when the app initially loads, thus decreasing wait time when switch between pages
#     - Cached so that the data is not reloaded at every user selection 
# 3. Create empty user profiles (lists) to be filled if the user creates a new profile
#     - One profile per session to simulate a log in experience
# 4. Create side navigation bar and call page module's write() function according to user selection 
#     - If create a profile, return the updated objects so can be used in personalized recommendations

# #### To Run:
# 1. Convert notebook to py file
#     - Run in command line: py -m jupyter nbconvert --to script main_app.ipynb
#     - Also convert all pages notebooks
# 2. Run streamlit app
#     - Run in command line: streamlit run main_app.py

# In[1]:


import streamlit as st 
import pandas as pd
import os
import numpy as np
import operator
import fastparquet
import re
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import sklearn


# ### Import Individual Pages

# In[ ]:


import pages.home_page
import pages.non_user_recommendations
import pages.item_item_rec_app
import pages.personalized_rec_app
import pages.profile_add_app
import pages.EDA_Streamlit_page


# ## Set up data
# - df: data for displaying on page with movie attributes. Created in recommendation_data_display.ipynb
# - x_unique: lists of unique values of features that can be filtered
# - ratings_df: ratings data
# - df_dummies1, df_dummies2: 2 sparse matricies with relevant features for the two versions of personalized recommendations
# - movieIds: list of all movieIds to mark rows of sparse matrices 
# - movieIds_notags, movieIds_tags: list of all movieIds without tags and with tags so can limit recommendations produced by two personalization models accordingly
# 
# Much of this data is needed for multiple of the pages, so more efficient to only load once in the main app and then feed into the respective functions. Cached so that only run once, not every time the user makes a selection.

# In[2]:


@st.cache(allow_output_mutation=True)
def data_setup():
    # read in data created in recommendation_data_display.ipynb
    df = pd.read_parquet('processed_files/recommendation_display.parq')

    # get unique lists of all filter values for user selections 
    genres_unique, actors_df, directors_df, countries_unique, language_unique, tags_unique, decades_unique, movies_unique = pages.non_user_recommendations.unique_lists(df)
    
    # data for personalized recommendations
    ratings_df, movieIds, df_dummies1, df_dummies2, collab_predictions, movieIds_tags, movieIds_notags = pages.personalized_rec_app.load_data()
        
    return df, genres_unique, actors_df, directors_df, countries_unique, language_unique, tags_unique, decades_unique, movies_unique, ratings_df, movieIds, df_dummies1, df_dummies2, collab_predictions, movieIds_tags, movieIds_notags


# In[5]:


df, genres_unique, actors_df, directors_df, countries_unique, language_unique, tags_unique, decades_unique, movies_unique, ratings_df, movieIds, df_dummies1, df_dummies2, collab_predictions, movieIds_tags, movieIds_notags = data_setup()


# ## Set up Empty New User Profile
# - Mutable list objects so that preserved between runs of script. Cached funciton that creates empty lists will not be rerun, so you can add things to profiles multiple times and the profile (list) will be preserved
# - Generate new unique user ID for this session (max existing ID + 1)
#     - User only gets one ID per session to create a new profile with. Simulating a login experience. 

# In[ ]:


# function creates empty lists, so not overwritten when page refreshes
# only works with mutable data types
@st.cache(allow_output_mutation=True)
def list_create():
    return [], [], [], [], []


# In[ ]:


@st.cache(allow_output_mutation=True)
def empty_profile_create(ratings_df):

    # empty lists to hold user input: will persist across user refresh because of function
    new_ratings, new_users, new_movies, new_titles, userId_new = list_create()    

    # generate a new user id 
    # append to list because changes every time the page is run. Only want first max entry. 
    userId_new.append(int(ratings_df.userId.max() + 1))
    
    return new_ratings, new_users, new_movies, new_titles, userId_new


# In[ ]:


new_ratings, new_users, new_movies, new_titles, userId_new = empty_profile_create(ratings_df)


# # Main Function: Navigation between Pages
# - Side radio menu that user can select page based on. Defaulted to home page.    
# - Once select page, call write() function within each page with appropriate data arguments to generate that page     
# - Return any updated profile data so can be repassed in for use in personalized recommendations 

# In[ ]:


PAGES = ['Home', 'Top Movie Visualizations', 'Top Rated Movies', 'Movie Based Recommendations',
         'Personalized Recommendations', 'Add Profile']


# In[12]:


def main(df, genres_unique, actors_df, directors_df, countries_unique, language_unique, tags_unique, decades_unique,
         movies_unique, df_dummies1, df_dummies2, collab_predictions, ratings_df, movieIds, movieIds_notags, movieIds_tags,
         new_ratings, new_users, new_movies, new_titles, userId_new):
    
    # set up side navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", PAGES)    
    
    # depending on user selection, call write function
    if selection == 'Home':
        pages.home_page.write()
    if selection == 'Top Movie Visualizations':
        pages.EDA_Streamlit_page.write()
    if selection == 'Top Rated Movies':
        pages.non_user_recommendations.write(df, genres_unique, actors_df, 
                                             directors_df, countries_unique, language_unique, tags_unique, decades_unique)
    if selection == 'Movie Based Recommendations':
        pages.item_item_rec_app.write(df, df_dummies1, df_dummies2, movieIds, movieIds_notags, movieIds_tags)
    if selection == 'Personalized Recommendations':
        pages.personalized_rec_app.write(df, genres_unique, actors_df, directors_df, countries_unique,
                                         language_unique, tags_unique, decades_unique,
                                         new_ratings, new_users, new_movies, ratings_df, movieIds,
                                        collab_predictions, df_dummies1, df_dummies2, movieIds_notags, movieIds_tags)
        
    if selection == 'Add Profile':
        new_ratings, new_users, new_movies, new_titles = pages.profile_add_app.write(df, new_ratings, new_users,
                                                                                     new_movies, new_titles, userId_new,
                                                                                     ratings_df)
        
    # return any updated profile data so can be repassed in for use in personalized recommendations
    return new_ratings, new_users, new_movies, new_titles


# In[ ]:


new_ratings, new_users, new_movies, new_titles = main(df, genres_unique, actors_df, directors_df, countries_unique,
                                                      language_unique, tags_unique, decades_unique, movies_unique,
                                                      df_dummies1, df_dummies2, collab_predictions, ratings_df, 
                                                      movieIds, movieIds_notags, movieIds_tags,
                                                      new_ratings, new_users, new_movies, new_titles, userId_new)

