#!/usr/bin/env python
# coding: utf-8

# # Non-Personalized recommendations based on (filtered) top rated movies
# **Use case**: users without a profile can input desired movie attributes and view the top rated movies in those categories.  
# Sort on weighted average: # of reviews * average rating 
# 
# Process: 
# - Allow users to choose multiple selections from filters using AND logic 
#     - For directors and actors, allow them to enter free text and display the top 3 most similar values in dataset for them to choose between 
#         - If display full alphabetized lists, too much data for streamlit to handle and app crashes
# - Once hit view recommendations, display sorted movies in the chosen categories   
#         
# Note: if run this locally outside of app, data paths will be incorrect. Assuming running in streamlit, in which case main_app.py calls these scripts from the root folder, which is where the datasets live.   
# Also, data is being passed in from main_app, so not all required data is loaded/created in this script

# In[1]:


import pandas as pd
import os
import numpy as np
import datetime as datetime
import operator
import streamlit as st
import fastparquet
import re
from fuzzywuzzy import fuzz


# ## Get Unique Lists of Filter Options
# Options for users to choose from in the streamlit dropdowns. Sorted alphabetically      
# For actors, directors get versions with upcase and downcase. Uppercased for display and downcased for checking against user input        
# Called in data prep section of main app.  
# 

# In[2]:


def cat_list_expand(df, var):
    
    # expand lists such that one entry per row 
    expanded = df[[var, 'movieId']]
    expanded = pd.DataFrame({
        col:np.repeat(expanded[col].values, expanded[var].str.len()) for col in expanded.columns.drop(var)}
    ).assign(**{var:np.concatenate(expanded[var].values)})[expanded.columns]

    return expanded


# In[42]:


@st.cache(allow_output_mutation=True)
def unique_lists(df):
    
    # unique lists. Sort alphabetically
    genres_unique = np.sort(cat_list_expand(df, 'Genres').Genres.unique())
    countries_unique  = np.sort(cat_list_expand(df, 'Filming Countries')['Filming Countries'].unique())
    language_unique = np.sort(cat_list_expand(df, 'Language(s)')['Language(s)'].unique())
    tags_unique = np.sort(cat_list_expand(df, 'Tags').Tags.unique())
    decades_unique = np.sort(df.decade.unique())
    movies_unique = np.sort(df.title_year.unique())
    
    # actors and directors: user input fuzzy string matching.
    # Get version with lower case for user matching + upper case for display
    actors_df = pd.merge(cat_list_expand(df, 'actors_downcased').actors_downcased,
                         cat_list_expand(df, 'Actors').Actors, left_index = True, right_index = True)
    # drop duplicated rows so unique
    actors_df = actors_df[actors_df.duplicated() == False]
    actors_df.columns = ['actors_downcased', 'actors_upcased']

    directors_df = pd.merge(cat_list_expand(df, 'directors_downcased').directors_downcased,
                            cat_list_expand(df, 'Director(s)')['Director(s)'], left_index = True, right_index = True)
    directors_df = directors_df[directors_df.duplicated() == False]    
    directors_df.columns = ['directors_downcased', 'directors_upcased']
    
    return genres_unique, actors_df, directors_df, countries_unique, language_unique, tags_unique, decades_unique, movies_unique


# ## Display in Streamlit with filter options
# Display:
# - Title
# - Year
# - Description
# - Duration
# - Genres
# - Actors
# - Directors
# - Production Company
# - Country
# - Language
# - Genome Tags
# - Number of ratings
# - Average rating    
#    
# Filter by: (unless otherwise stated, multiselect so can select multiple with AND logic)
# - Genres
# - Decade of release
#     - Single select dropdown. AND logic, so can't have a movie in multiple decade
# - Country
# - Language
# - Genome Tags
# - Actors
# - Directors
#     - Actors, directors with free text input -> string matching -> choose among 3 best options in dropdown
#     - Allow multiple entries and selections if separate by a comma
# 
# Default table is highest rated movies without filters    
#      
#      
# Fuzzy string matching for actors and directors: token sort
# - Tokenize phrases, sort alphabetically, join, find Levenshtein distance
# - Alphabetical sort is important because user input: The Notebook but movie title: Notebook, The    
# - Cached function so that once user types in the box, the app won't redo the fuzzy matching process every time a user changes a filter afterwards

# In[1]:


@st.cache(allow_output_mutation=True)
def fuzzy_matching(user_input, original_df, var):
    # downcase input
    user_input = user_input.lower()
    # split into list based on commas
    user_input = user_input.split(', ')

    # fuzzy string matching to find similarity ratio between user input and actual actors (downcased)
        # works for misspellings as well 
        # limit to 70% similarity 
    options = []
    sim_df = original_df.copy()
    for i in user_input:
        # find similarity ratio between input and all unique actors (downcased)
        sim_df['sim'] = sim_df[var + '_downcased'].apply(lambda row: fuzz.token_sort_ratio(row, i))
        # get top 3 with similarity > 70%
        options.append(sim_df[sim_df.sim > 70].sort_values('sim', ascending = False
                                                          ).head(3)[var + '_upcased'].unique())
    # flatten options list
    options = [item for sublist in options for item in sublist]    

    return options


# In[2]:


def write(df_display, genres_unique, actors_df, directors_df, countries_unique,
          language_unique, tags_unique, decades_unique):
    
    # user instructions 
    st.title('Top Rated Movie Recommendations')
    st.header('View the top rated movies with your desired attributes')
    st.write('Enter filters and select **Display Recommendations** \n' + 
             'If you wish to see overall top rated movies, select **Display Recommendations** without any filters')
    st.write('Please note filters use AND logic')
    
    # get user inputs: multiple selection possible per category except decade
    # input sorted list of unique options 
    genre_input = st.multiselect('Select genre(s)', genres_unique)
    decade_input = st.selectbox('Select film decade', ['Choose an option'] + list(decades_unique)) # single option select 
    country_input = st.multiselect('Select filming country(s)', countries_unique)
    language_input = st.multiselect('Select language(s)', language_unique)
    tag_input = st.multiselect('Select genome tags(s)', tags_unique)
    
    # actors, directors get text inputs - dropdowns too many values for streamlit to handle
    # allow multiple entries with a commoa 
    actor_input = st.text_input('Type actor(s) names separated by commas. Select intended actor(s) from dropdown that appears')
    if actor_input != '':
        
        options = fuzzy_matching(actor_input, actors_df, 'actors')
        
        # list actors that are similar to what they typed and accept user selection(s)
        if len(options) > 0:
            actor_input = st.multiselect('Select Actor(s)', options)
        else:
            st.write("Sorry, we can't find any matching actors")

    else:
        actor_input = []

    director_input = st.text_input('Type director(s) names separated by commas. ' + 
                                   'Select intended director(s) from dropdown that appears')
    if director_input != '':
        
        options = fuzzy_matching(director_input, directors_df, 'directors')

        # list directors that are similar to what they typed and accept user selection(s)
        if len(options) > 0:
            director_input = st.multiselect('Select Director(s)', options)
        else:
            st.write("Sorry, we can't find any matching directors")

    else:
        director_input = []

    # display recommendations once hit button
    if st.button('Display Recommendations'):
        
        # for decade, only filter if chose an option (no NA default for selectbox)
        if decade_input != 'Choose an option':
            df_filtered = df_display[(df_display.decade ==  decade_input)]
        else:
            df_filtered = df_display.copy()
        # filter dataframe with rest of filters, sort and get top 10. Drop columns we don't want to display.
        df_filtered = df_filtered[(df_filtered.Genres.map(set(genre_input).issubset)) & 
                                 (df_filtered['Filming Countries'].map(set(country_input).issubset)) &
                                 (df_filtered['Language(s)'].map(set(language_input).issubset)) & 
                                 (df_filtered.Tags.map(set(tag_input).issubset))  & 
                                 (df_filtered['Actors'].map(set(actor_input).issubset)) &
                                 (df_filtered['Director(s)'].map(set(director_input).issubset)) 
                                 ].sort_values('weighted_avg', ascending = False
                                              ).head(10).drop(columns = ['weighted_avg','actors_downcased', 
                                                                         'directors_downcased', 'title_downcased', 
                                                                         'title_year', 'movieId', 'decade', 'tags_num'])
        # if no valid movies with combination of filters, notify. Else display dataframe
        if len(df_filtered) > 0:
            st.write(df_filtered)
        else:
            st.write('Found no movies that match your selections')

