#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import os
import numpy as np
from scipy.sparse import csc_matrix
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as datetime
import operator
import fastparquet
import streamlit as st
from PIL import Image


# # Change the directory name
# - image rec folder is also uploaded

# In[24]:


def write():
    #directory_name ='/Users/shwetaanand/DEVHOME/Images_rec'
    directory_name = '/DropBox/DS5500/project2/DS5500-Movie-Recommendation-System/pages/Images_rec'
    
    st.markdown("<h1 style='text-align: center; color: red;'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: blue;'>Select Check Box - to display EDA</h1>", unsafe_allow_html=True)

    st.write("\n")
    st.write("\n")


    if st.checkbox('Top 10 Genre Combinations'):
        image_file = 'Top 10 Genre Combinations'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

    if st.checkbox('Top 10 Actors/Actresses'):
        image_file = 'Top 10 Actors or Actresses'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

    if st.checkbox('Top 10 Directors'):
        image_file = 'Top 10 Directors'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image,  use_column_width=True)

    if st.checkbox('Top 10 Filming Countries'):
        image_file = 'Top 10 Filming Countries'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

    if st.checkbox('Top 10 (Relevant greater than 50%) Genome Tags'):
        image_file = 'Top 10 (Relevant greater than 50%) Genome Tags'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

    if st.checkbox('Top 10 Highly Rated Actors -  Minimum 100 Ratings'):
        image_file = 'Top 10 Highly Rated Actors - Minimum 100 Ratings'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

    if st.checkbox('Top 10 Highly Rated Directors - Minimum 100 Ratings'):
        image_file = 'Top 10 Highly Rated Directors - Minimum 100 Ratings'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

    if st.checkbox('Top 10 Highly Rated Movies - Minimum 100 Ratings'):
        image_file = 'Top 10 Highly Rated Movies - Minimum 100 Ratings'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

    if st.checkbox('Top 10 Highly Rated Movies - Weighted Average'):
        image_file = 'Top 10 Highly Rated Movies - Weighted Average'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

    if st.checkbox('Genres Frequency'):
        image_file = 'Frequency_Genres'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

    if st.checkbox('Genres by Average Rating'):
        image_file = 'Genres by Average Rating'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

    if st.checkbox('Distribution of Ratings'):
        image_file = 'Distribution of Ratings'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)


    if st.checkbox('Number of Ratings (millions)'):
        image_file = 'Number of Ratings (millions)'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

    if st.checkbox('Distribution of Number of Tags per Movie'):
        image_file = 'Distribution of Number of Tags per Movie'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

    if st.checkbox('Distribution of Number Ratings by Movie- Many (Unpopular Movies)'):
        image_file = 'Distribution of Number Ratings by Movie- Many (Unpopular Movies)'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

    if st.checkbox('Distribution of Tag Relevance Scores'):
        image_file = 'Distribution of Tag Relevance Scores'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

    if st.checkbox('Release Year Frequency Counts'):
        image_file = 'Release Year Frequency Counts'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

