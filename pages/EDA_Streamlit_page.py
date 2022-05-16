#!/usr/bin/env python
# coding: utf-8

# In[7]:


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

# In[8]:





# In[6]:


def write():
    
    #directory_name ='/Users/shwetaanand/DEVHOME/Images_rec'
    directory_name = 'pages/Images_rec'

    st.markdown("<h1 style='text-align: center; color: red;'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    st.text("")
    st.text("")
    st.markdown("<h1 style='text-align: center; color: blue;'>Select Check Box - to display EDA</h1>", unsafe_allow_html=True)
    st.text("")
    st.text("")

    if st.checkbox('Movie count by Genre'):
        image_file = 'Movie count by Genre'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

    if st.checkbox('Top 10 Genre Combinations'):
        image_file = 'Top 10 Genre Combinations'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)


    if st.checkbox('Top 10 Most Common Actors'):
        image_file = 'Top 10 Actors or Actresses'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

    if st.checkbox('Top 10 Most Common Directors'):
        image_file = 'Top 10 Directors'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

    if st.checkbox('Top 10 Filming Countries'):
        image_file = 'Top 10 Filming Countries'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)            


    if st.checkbox('Top 10 Highly Rated Actors*'):
        image_file = 'Top 10 Most Rated Actors'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)


    if st.checkbox('Top 10 Highly Rated Directors*'):
        image_file = 'Top 10 Most Rated Directors'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

    if st.checkbox('Top 10 Highly Rated Movies*'):
        image_file = 'Top 10 Most Rated Movies'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)

    if st.checkbox('Number Of Movie Released Per Year'):
        image_file = 'Number Of Movie Released Per Year'
        suffix='.png'
        img_file =os.path.join(directory_name, image_file + suffix)
        image = Image.open(img_file)
        st.image(image, use_column_width=True)                      
    st.text("")
    st.text("")
    st.text('*Weighted Average Rating between average ratings and number of ratings ')
    st.text("")
    st.text("")

