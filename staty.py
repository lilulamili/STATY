#----------------------------------------------------------------------------------------------

####################
# IMPORT LIBRARIES #
####################

import streamlit as st
import sys
from streamlit import cli as stcli
import pandas as pd
import numpy as np
import plotly as dd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager
import plotly.graph_objects as go
import functions as fc
import use_case_indData
import use_case_panelData
import use_case_TextData
import use_case_simpleData
import use_case_home
import use_case_geo
import use_case_timeSeries
import use_case_faqs
import os 
import platform
import time
from streamlit import caching
import streamlit.components.v1 as components
from PIL import Image

#----------------------------------------------------------------------------------------------

###############
# PAGE CONFIG #
###############


staty_favi =  "default data/favicon.png"



# Define page setting
st.set_page_config(
    page_title = "STATY",
    page_icon = staty_favi,
    layout = "centered",
    initial_sidebar_state = "collapsed",
    
)   

hide_footer_style = """
<style>
.reportview-container .main footer {visibility: hidden;}    
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)

# Title
components.html("""
    <body 
    style = "margin-left: 0px;">

        <h1 
        style = "color:#38bcf0;
        font-family: sans-serif;
        font-weight: 750;
        font-size: 40px;
        line-height: 1;"
        >STATY
        </h1>

    </body>

    """,
    height = 70
)


st.markdown("Your data can tell great stories! Let STATY do all data the juggling for you, and focus your attention on the outcomes of powerful models.")
st.sidebar.title("Menu")




#----------------------------------------------------------------------------------------------

#######
# APP #
#######

# Run different code for different use case
PAGES = {
    "Home": use_case_home,
    "Uni- and bivariate data": use_case_simpleData,
    "Multivariate data": use_case_indData,
    "Panel data": use_case_panelData,
    "Web scraping and text data": use_case_TextData,
    "Geospatial data": use_case_geo,
    "Time series data":use_case_timeSeries,
    "FAQs": use_case_faqs
}
#st.sidebar.subheader("Navigation")
use_case = st.sidebar.radio("Navigation", ["Home", "Uni- and bivariate data", "Multivariate data", "Panel data", "Time series data", "Web scraping and text data", "Geospatial data", "FAQs"])
st.sidebar.markdown("")

page = PAGES[use_case]
page.app()


st.sidebar.markdown("We‘d love to hear your [feedback] (https://qfreeaccountssjc1.az1.qualtrics.com/jfe/form/SV_4G6JfNt6sHORUt8)!")

st.sidebar.markdown("Report a [bug] (https://drive.google.com/file/d/1iDLJMyYxwRGXtpQUj6H1Ose5DJ0AnICj/view?usp=sharing)!")

st.sidebar.markdown("")
st.sidebar.markdown("Your :rocket: to data science!")

# Hide footer
hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


# Decoration bar color
decoration_bar_style ="""
    <style>
    .css-kywgdc {
        background-image:  linear-gradient(90deg, #38bcf0, #dcf3fc);
    }
    </style>
"""
st.markdown(decoration_bar_style, unsafe_allow_html=True) 

# Progress bar color
progress_bar_style ="""
    <style>
    .stProgress > div > div > div > div {
        background-color: #38bcf0;
    }
    </style>
"""
st.markdown(progress_bar_style, unsafe_allow_html=True) 


# st.info color
info_container_style ="""
    <style>
    .stAlert .st-al {
        background-color: rgba(220, 243, 252, 0.4);
        color: #262730;
    }
    </style>
"""
st.markdown(info_container_style, unsafe_allow_html=True) 
