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
import use_case_MultivariateData
import use_case_PanelData
import use_case_WebScrapingTextData
import use_case_UnivariateBivariateData
import use_case_Home
import use_case_GeospatialData
import use_case_TimeSeriesData
import use_case_FAQs
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


st.markdown("Your data can tell great stories! Let STATY do all the data juggling for you, and focus your attention on critical reflection of issues around your data.") 
#Let STATY do all the data juggling for you, and focus your attention on the outcomes of powerful models and on critical reflection.
st.sidebar.title("Menu")




#----------------------------------------------------------------------------------------------

#######
# APP #
#######

# Run different code for different use case
PAGES = {
    "Home": use_case_Home,
    "Uni- and bivariate data": use_case_UnivariateBivariateData,
    "Multivariate data": use_case_MultivariateData,
    "Panel data": use_case_PanelData,
    "Web scraping and text data": use_case_WebScrapingTextData,
    "Geospatial data": use_case_GeospatialData,
    "Time series data":use_case_TimeSeriesData,
    "FAQs": use_case_FAQs
}
#st.sidebar.subheader("Navigation")
use_case = st.sidebar.radio("Navigation", ["Home", "Uni- and bivariate data", "Multivariate data", "Panel data", "Time series data", "Web scraping and text data", "Geospatial data", "FAQs"])
st.sidebar.markdown("")

page = PAGES[use_case]
page.app()


st.sidebar.markdown("We‘d love to hear your [feedback] (https://qfreeaccountssjc1.az1.qualtrics.com/jfe/form/SV_4G6JfNt6sHORUt8)!")

st.sidebar.markdown("Report a [bug] (https://drive.google.com/file/d/1iDLJMyYxwRGXtpQUj6H1Ose5DJ0AnICj/view?usp=sharing)!")

st.sidebar.write("Check the [documentation] (https://quant-works.de/staty/STATY_Documentation_v0.1.pdf)!")

st.sidebar.markdown("")
st.sidebar.markdown("Your :rocket: to data science!   \n Licensed under the [Apache License, Version 2.0] (https://www.apache.org/licenses/LICENSE-2.0.html)")
 

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
    div[data-testid*="stDecoration"] {
        background-image:  linear-gradient(90deg, #38bcf0, #dcf3fa);
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
