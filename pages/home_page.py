#!/usr/bin/env python
# coding: utf-8

# # Home Page For Streamlit App
# List and describe each of the pages

# In[ ]:


import streamlit as st


# In[1]:


def write():
    st.title('Welcome to the Movie Recommender!')
    st.header('Select your destination on the left menu')
    st.write("""
    - **Top Movie Visualizations**: View visualizations of the movies in our catalog to understand what makes a great movie.
    - **Top Rated Movies**: Apply filters to find the top rated movies with your desired attributes
    - **Movie Based Recommendations**: Enter a movie that you have previously enjoyed to view similar movies
    - **Personalized Recommendations**: Enter your user ID to find your personalized top movies + apply filters
    - **Add Profile**: If you are not in our system, create a new profile to enable personalized recommendations
    """)

