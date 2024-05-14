import os
import pickle
import torch
import streamlit as st
from streamlit_option_menu import option_menu

import numpy as np
import pandas as pd
from sklearn import preprocessing

# Set page configuration
st.set_page_config(page_title="ML Application Models",
                   layout="centered")

# loading the saved models
Models = {
    'HGR': {},
}

Models_dir = "Models"
for model_name in os.listdir(Models_dir):
    with open(os.path.join(Models_dir, model_name), 'rb') as file:
        model = pickle.load(file)
        Models['HGR'][model_name] = model

# sidebar for navigation
with st.sidebar:
    selected = option_menu('ML Models',
                           [
                               'Hand Gesture Recognition',
                            ],
                           default_index=0)


# Decison Tree Page
if selected == 'Hand Gesture Recognition':

    # page title
    st.title('Hand Gesture Recognition Model')
    # st.write('This is a Decison Tree model for Breast Cancer')

    # dataset link
    st.markdown(
        """
        <a href="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic" target="_blank">
            <button style='background-color: #262730;
            border: 0px;
            border-radius: 10px;
            color: white;
            padding: 10px 15px;
            text-align: center;
            text-decoration: none;
            font-size: 16px;
            margin-bottom: 1rem;
            cursor: pointer;'>Breast Cancer Dataset</button>
        </a>
        """, 
        unsafe_allow_html=True,
    )

    # code block
    st.title('Notebook')
    code = '''
    def hello():
        print("Hello, Streamlit!")
    '''
    st.code(code, language='python')
    code = '''
    def hello():
        print("Hello, Streamlit!")
    '''
    st.code(code, language='python')
    code = '''
    def hello():
        print("Hello, Streamlit!")
    '''
    st.code(code, language='python')
