#----------------------------------------------------------------------------------------------

####################
# IMPORT LIBRARIES #
####################

import streamlit as st
import os
import pybase64
import platform
import sys
import functions as fc
from PIL import Image

#----------------------------------------------------------------------------------------------

def app():

    
    #------------------------------------------------------------------------------------------
    # SETTINGS

    settings_expander=st.sidebar.beta_expander('Settings')
    with settings_expander:
        st.caption("**Help**")
        sett_hints = st.checkbox('Show learning hints', value=False)
        st.caption("**Appearance**")
        sett_wide_mode = st.checkbox('Wide mode', value=False)
        sett_theme = st.selectbox('Theme', ["Light", "Dark"])
        #sett_info = st.checkbox('Show methods info', value=False)
        #sett_prec = st.number_input('Set the number of diggits for the output', min_value=0, max_value=8, value=2)
    st.sidebar.markdown("")

    # Load background image
    
        
    home_bg_dark =  "default data/bg_image_dark.jpg"
    home_bg_light = "default data/bg_image_light.jpg"
    bg_ext = "jpg"

    # Check if wide mode
    if sett_wide_mode:
        #fc.wide_mode_func()
        st.empty()

    # Check theme
    if sett_theme == "Dark":
        st.markdown(
        f"""
        <style>
        .reportview-container .main {{
            color: white;
            background: url(data:image/{bg_ext};base64,{pybase64.b64encode(open(home_bg_dark, "rb").read()).decode()});
            background-color: rgb(38, 40, 43);
            background-position: center top;
        }}
        span[id*="MainMenu"]  {{
            display: none !important;
            color: white;
        }}
        section[data-testid*="stSidebar"] > div {{
            background-color: rgb(49, 51, 63);
        }}
        section[data-testid*="stSidebar"] > div > div > div > div {{
        background-color: rgb(49, 51, 63);
        }}
        section[data-testid*="stSidebar"] > div > button {{
            background-color: rgb(38, 40, 43);
        }}
        section[data-testid*="stSidebar"] > div > div > button {{
            color: white;
        }}
        div[data-testid*="stMarkdownContainer"] > h1 {{
            color: white;
        }}
        div[data-testid*="stMarkdownContainer"] > h2 {{
            color: white;
        }}
        div[data-testid*="stMarkdownContainer"] > h3 {{
            color: white;
        }}
        div[data-testid*="stMarkdownContainer"] > small {{
            color: white;
        }}
        div[data-testid*="stMarkdownContainer"] {{
            color: white;
        }}
        div[data-testid*="stTable"] > table {{
            color: white;
        }}
        div[data-testid*="stTable"] > table > thead > tr > th {{
            border-color: white;
        }}
        div[data-testid*="stTable"] > table > tbody > tr > th {{
            border-color: white;
        }}
        div[data-testid*="stTable"] > table > tbody > tr > td {{
            border-top: 1px solid rgba(255, 255, 255, 0.7);
            border-bottom: 1px solid rgba(255, 255, 255, 0.7);
        }}
        .stDataFrame > div {{
            background-color: white;
            color: rgb(38, 39, 48);
        }}
        div[width*="2"] {{
            background-color: rgb(38, 40, 43);
        }}
        label[data-baseweb*="radio"] > div {{
            color: white;
        }}
        .reportview-container > section > div  {{
            color: white;
        }}
        .element-container {{
            color: white;
        }}
        .streamlit-expander {{
            border-color: white;
            border-top: 0px;
            border-left: 0px;
            border-right: 0px;
        }}
        .streamlit-expanderHeader {{
            color: white;
            padding-left: 0px;
            font-size: 1rem;
        }}
        .streamlit-expanderHeader > svg > path {{
            color: white;
        }}
        .streamlit-expanderContent {{
        padding-left: 5px;
        }}
        .element-container > div > div > h2 {{
            color: white;
        }}
        .row-widget > label {{
            color: white;
        }}
        .row-widget > button {{
            color: black;
        }}
        .row-widget > label > div {{
            color: white;
        }}
        .stSlider > label {{
            color: white;
        }}
        .stTextArea > label {{
            color: white;
        }}
        p > a {{
        color: white !important;
        }}
        </style> 

        """,
            unsafe_allow_html=True,
        )
    if sett_theme == "Light":
        st.markdown(
            f"""
        <style>
        span[id*="MainMenu"]  {{
            display: none !important;
        }}  
        .reportview-container .main {{
            color: white;
            background: url(data:image/{bg_ext};base64,{pybase64.b64encode(open(home_bg_light, "rb").read()).decode()});
            background-position: center top;
        }}
        div[data-testid*="stMarkdownContainer"] > h1 {{
            color: rgb(38, 39, 48);
        }}
        label[data-baseweb*="radio"] > div {{
            color: rgb(38, 39, 48);
        }}
        .reportview-container > section > div  {{
            color: rgb(38, 39, 48);
        }}
        .element-container {{
            color: rgb(38, 39, 48);
        }}
        .streamlit-expander {{
            border-color: rgba(38, 39, 48, 0.2);
            border-top: 0px;
            border-left: 0px;
            border-right: 0px;
        }}
        .streamlit-expanderContent {{
        padding-left: 5px;
        }}
        .streamlit-expanderHeader {{
            color: rgb(38, 39, 48);
            padding-left: 0px;
            font-size: 1rem;
        }}
        .element-container > div > div > h2 {{
            color: rgb(38, 39, 48);
        }}
        .row-widget > label {{
            color: rgb(38, 39, 48);
        }}
        .row-widget > button {{
            color: rgb(38, 39, 48);
        }}
        .row-widget > label > div {{
            color: rgb(38, 39, 48);
        }}
        .stSlider > label {{
            color: rgb(38, 39, 48);
        }}
        </style> 
        
        """,
            unsafe_allow_html=True,
        )  

    # st.markdown(
    #     f"""
    #     <style>
    #     .reportview-container {{
    #         background: url(data:image/{bg_ext};base64,{pybase64.b64encode(open(home_bg, "rb").read()).decode()})
    #     }}
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )


    #st.markdown("You don’t need any programming skills to get started - STATY will do data mining and machine learning for you, present you great visualisations of your data and will boost your curiosity to explore more. ")
    st.write("To get started, open the menu on the left, or check the intro video!")
    
    
    staty_video =open("default data/Staty_web.mp4", 'rb')        
    staty_video_bytes=staty_video.read()
    st.video(staty_video_bytes)
    #st.caption("Music by Sophonic Media, http://instrumentalsfree.com")

    
    col1, col2 = st.beta_columns(2)
    with col1:
        st.write("STATY is an ongoing educational project designed and developed by [Oskar Kärcher](mailto:o.kaercher@hs-osnabrueck.de?subject=Staty-App) and [Danijela Markovic](mailto:d.markovic-bredthauer@hs-osnabrueck.de?subject=Staty-App) with the aim of improving data literacy among undergraduate and graduate students.")
    st.write("STATY is still under development, and some features may not yet work properly! STATY is therefore provided 'asis' without any warranties of any kind.")
    #st.write("Disclaimer: STATY and the related tools and data are provided for educational purposes only. Note, the project is still under development, and some features may not yet work properly! \n  Licensed under the [Apache License, Version 2.0] (https://www.apache.org/licenses/LICENSE-2.0.html).")
    
    
        
    if sett_theme == "Dark":
        image = Image.open("default data/HS-OS-Logo_dark.png")
    else:
        image = Image.open("default data/HS-OS-Logo_light.png")
    with col2:
        st.image(image)
    
  

    
