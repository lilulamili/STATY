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

    settings_expander=st.sidebar.expander('Settings')
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

    st.write("STATY is an educational project designed and developed by [Danijela Markovic](mailto:staty@quant-works.de?subject=Staty-App-Contact%20Danijela) and [Oskar Kärcher](mailto:staty@quant-works.de?subject=Staty-App-Contact%20Oskar) with the aim of improving data literacy among students of natural and social sciences.")
    col1, col2 = st.columns([2,1])
    with col1:            
        st.write("STATY is provided 'as is' without any warranties of any kind! STATY is under development, and is subject to change!")
        #st.write("Disclaimer: STATY and the related tools and data are provided for educational purposes only. Note, the project is still under development, and some features may not yet work properly! \n  Licensed under the [Apache License, Version 2.0] (https://www.apache.org/licenses/LICENSE-2.0.html).")
    
    staty_expander=st.expander('STATY - get more info')
    with staty_expander:
        st.write("")
        st.markdown("**Background and motivation**")
        st.markdown("The digital transformation has a significant impact on the competencies in the working world of tomorrow. The need for data literacy and critical data awareness has become obvious and calls for an effective response in higher education of natural and social sciences. However, active usage of data tools in undergraduate and graduate programs requires that both students and teachers have either good programming skills, or that they have to familiarize themselves with sometimes cumbersome software solutions. Students with excellent programming skills are an exception, few gain some programming skills at the cost of understanding the methodology, while the majority of students leave universities without basic data literacy skills.")

        st.write("STATY is growing out of the effort to bring more data insights to university education across all disciplines of the natural and social sciences. It is motivated by the belief that fostering data literacy, creativity and critical thinking are more effective towards innovation, than bringing endless units of introduction to programming to students who find  learning programming an overwhelming task. By providing easy access to the methods of classical statistics and machine learning, STATY’s approach is to inspire students to explore issues  they are studying in the curriculum directly on real data, practice interpreting the results and check the source code to see how it is done or to improve the code. ")
        st.write("")

        st.write("STATY can be used in the process of teaching and learning data science, demonstrations of theoretical concepts across various disciplines, active learning, promotion of teamwork, research and beyond.")
        #st.markdown("**STATY**")        
        #st.write("-	is providing easy access to the methods of classical statistics and machine learning.")
        #st.write("-	is shifting the focus on results interpretation and critical questioning.")
        #st.write("-	is a companion in the process of teaching and learning data science. ")
        #st.write("-	can be used in teaching and studying whenever  one wants to explore current data relevant to the subject.")
        #st.write("-	is a great help for all who want to apply the methods of classical statistics and machine learning in the context of various studies.")
        st.write("")

        st.markdown("**About us**")    
        st.markdown("[Oskar Kärcher](mailto:staty@quant-works.de?subject=Staty-App-Contact%20Oskar) (Dr.) is a data scientist, mathematician, musician and entrepreneur.") 
        st.markdown("[Danijela Markovic](https://www.hs-osnabrueck.de/prof-dr-ing-danijela-markovic-bredthauer/) (Prof. Dr.) is a professor of Quantitative Methods at Osnabrück University of Applied Sciences, ex-business consultant, fashion designer, creative thinker and scientist in the areas of climate and global change, conservation ecology, water resources and flight meteorology.")
        st.write("")

        st.markdown("**Acknowledgments**")
        st.write("STATY was financed through the Fund for Improvement of Study Conditions (Studienqualitätsmittel) provided by the state of Lower Saxony.")
        st.write("Development of STATY has been rocketed by great Python libraries including  [altair] (https://altair-viz.github.io/), [plotly] (https://plotly.com/python/), [statsmodels](https://www.statsmodels.org/stable/index.html), [scikit-learn](https://scikit-learn.org/stable/), [scikit-optimize](https://scikit-optimize.github.io/stable/) and [streamlit](https://streamlit.io/)! The full list of libararies used is available [here](https://quant-works.de/staty/requirements.txt).")
        st.write("")

        st.markdown("**Interesting to know**")
        st.write("Before STATY, we have never written a single line of code in Python. The research environments in which we lived so far are using languages such as Fortran, IDL, R and C++. To check Python’s suitability for STATY we did a short experiment. We tried to develop a STATY prototype with all the basic app-features such as data upload per drag and drop, user selection of variables and machine learning methods, model evaluation and comparison of the  goodness of fit statistics. Despite absolutely zero experience with Python, the app was ready in 2,5h! This experiment has given us a clear path towards STATY development – Python! We didn’t regret it - the first version of STATY was developed in 2,5 months! This said, some lines of code are probably awfully un-elegant!") 
        st.write("")

        st.markdown("**Citation**")
        st.markdown("If you use STATY for your research, please cite the libraries of the corresponding STATY module. The key libraries of each STATY module are provided in the FAQ's.  \n You can cite STATY using STATY's URL and the access date.")
        st.markdown("If you intend to use STATY for commercial projects, please [contact us](mailto:staty@quant-works.de?subject=Staty-App-Contact%20commercial%20project).")
        st.write("")
        
        st.markdown("**Disclaimer**")
        st.write("STATY is still under development, and some features may not yet work properly!   \n STATY is provided 'as is' without any warranties of any kind!")
        st.write("")
        st.write("")


    if sett_theme == "Dark":
        image = Image.open("default data/HS-OS-Logo_dark.png")
    else:
        image = Image.open("default data/HS-OS-Logo_light.png")
    with col2:
        st.image(image)
  

    
