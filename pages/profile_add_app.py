#!/usr/bin/env python
# coding: utf-8

# # Create New Profile
# **Use Case:** new user without existing profile. Enter movies and ratings so that they can receieve personalized recommendations       
# Details:
# - Persistant across session: if come back to page later (without clearing cache), will be on the same profile
#     - Cannot create a new profile in 1 session. Makes sense because proxying a login experience. 
# - Persistant between sessions: save data so when return to app later, can pull up an old profile. 
# - Add to LISTS instead of dataframes because mutable: persistant across sessions 
#     - Thus can enter ID created in personalization page to get recommendations    
# - User must add at least 2 DISTINCT ratings, else will produce no recommendations in the recommendation tab 
#     - Because of normalization - rating becomes 0 if only 1 distinct value (mean = self)
#     
# Process:
# - User free text input movie title
#     - Too many to display as a dropdown. Streamlit crashes
# - Look for movies in system with > 70% similarity. Display top 10
# - User selects from dropdown (1 selection only possible)
# - User input rating and hit submit
# - If hit view profile & save without 2 distinct ratings (at least 2 movies with 2 different ratings), ask to enter more. 
# - Otherwise, display what they've entered + concat to ratings dataframe and save 

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


# In[ ]:


def write(df, new_ratings, new_users, new_movies, new_titles, userId_new, ratings):
    
    # user instructions 
    st.title('Create a New User Profile')
    st.write('(1) Type the title of a movie you have watched')
    st.write('(2) Select a title from the dropdown of potentially matching movies in our system')
    st.write('(3) Provide a rating from 0.5-5.0 stars (higher better)')
    st.write('(4) Click **Submit** to submit this rating. Repeat as many times as desired.                \n ' +
            'In order for us to generate recommendations, you must rate *at least two* movies ' + 
             'with *at least two* different star values.')
    st.write('(5) Click **View & Save Profile** to view your profile and save it for next time. ' + 
             'Please wait for the save to complete.')
    st.write('Enter your user ID on the Personalized Recommendation pages. Feel free to return and add more movies later.')
    st.write('')
    st.write('**Your User ID is: ' + str(userId_new[0]) + '**')
        
    # get user input movie title, free text - too many movies for a full drop down 
    user_text = st.text_input("Enter a movie you have watched")
    # downcase input
    user_text = user_text.lower()

    # if no entry (initial state):
    if user_text == '':
        st.write('Waiting for input')
    # once enter text:
    else:
        # fuzzy string matching to find similarity ratio between user input and actual movie title (downcased)
            # works for misspellings as well 
            # limit to 70% similarity 
        options = df.copy()
        # find similarity ratio between input and all unique movies (downcased)
        options['sim'] = options.title_downcased.apply(lambda row: fuzz.ratio(row, user_text))
        # get top 10 with similarity > 70%. Display full title with (year) in case multiple with same title
        options = options[options.sim > 70].sort_values('sim', ascending = False).head(10).title_year.unique()
        
        # find movies with titles similar to what they typed
        if len(options) > 0:

            # user select out of possible options. Accept one option only. 
            user_title = st.selectbox('Select Movie', ['<select>'] + list(options))

            # once input something, ask for rating
            if user_title != '<select>':

                # ask for rating
                user_rating = st.selectbox('Rate this Movie', [i/2 for i in range(1,11)])

                # once hit submit, add to lists
                if st.button('Submit'):

                    # find ID of movie they selected based on year title 
                    user_movieid = int(df[df.title_year == user_title].movieId.values[0])

                    # add to persistant lists for this profile
                    new_movies.append(user_movieid)
                    new_ratings.append(user_rating)
                    new_titles.append(user_title)
                    new_users.append(userId_new[0])
                    
        # if nothing with > 70% similiarity, then can't find a matching movie
        else:
            st.write("Sorry, we can't find any matching movies")
            
    # view your profile and save 
    if st.button('View & Save Profile'):
        
        # if they've entered fewer than 2 distinct ratings, notify and do not save profile yet
            # set such that looking for distinct ratings
        if len(set(new_ratings)) < 2:
            st.write("You haven't entered enough ratings! Please rate at least two movies with at least two different star values")
        else:
            # create dataframe from lists for profile display
            d = {'movieId':new_movies, 'title':new_titles, 'rating':[str(round(i, 1)) for i in new_ratings]}
            profile = pd.DataFrame(d)
            st.write('Here is your profile')
            st.write(profile)

            # create dataframe from lists for ratings profile save 
            d = {'rating':new_ratings, 'userId':new_users, 'movieId':new_movies}
            new_ratings = pd.DataFrame(d)

            # sometimes duplicate movies from user profile adds - average ratings. Else matrix multiplication won't work
            new_ratings = new_ratings.groupby(['userId', 'movieId']).rating.mean()  
            new_ratings = new_ratings.reset_index(drop = False)

            # in case saved/viewed profile previously,
            # delete instances of this ID in the saved dataset so don't create duplicates
            ratings = ratings[ratings.userId != userId_new[0]]

            # concat with original ratings
            ratings = pd.concat([ratings, new_ratings], sort = False)
            ratings = ratings.reset_index(drop = True)

            # save. Alert user when done
            st.write('Saving your profile. Please wait...')
            ratings.to_parquet('processed_files/ratings_sample_useradd_collab.parq',
                               engine = 'fastparquet', compression = 'GZIP', index = False)
            st.write('Done!')
    
    # return so can be used in this current run
    return new_ratings, new_users, new_movies, new_titles

