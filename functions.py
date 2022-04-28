# FUNCTIONS FOR DATA ANALYSIS APP

#############
# FUNCTIONS #
#############

import pandas as pd
import numpy as np
import scipy
from scipy import stats
import sklearn
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
import plotly.graph_objects as go
import streamlit as st
from random import randint
from sklearn.feature_extraction.text import CountVectorizer

#----------------------------------------------------------------------------------------------
#FUNCTION FOR WIDE MODE

def wide_mode_func():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

#----------------------------------------------------------------------------------------------
#FUNCTION FOR DARK THEME

def theme_func_dark():
    st.markdown(
        f"""
    <style>
    .reportview-container .main {{
        background-color: rgb(38, 40, 43);
        color: white;
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
    section[data-testid*="stSidebar"] > div > div > div > div > div > ul > li > div > div > div {{
        background-color: rgb(49, 51, 63);
    }}
    section[data-testid*="stSidebar"] > div > button {{
        background-color: rgb(38, 40, 43);
    }}
    section[data-testid*="stSidebar"] > div > div > button {{
        color: white;
    }}
    div[data-testid*="stFileUploader"] > label {{
        color: white !important;
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
        color: white;
        font-weight: 550;
        border-top: 1px solid rgba(255, 255, 255, 0.7);
        border-left: 1px solid rgba(255, 255, 255, 0.7);
    }}
    div[data-testid*="stTable"] > table > tbody > tr > th {{
        border-color: white;
        color: white;
        font-weight: 550;
        border-left: 1px solid rgba(255, 255, 255, 0.7);
    }}
    div[data-testid*="stTable"] > table > tbody > tr > td {{
        border-top: 1px solid rgba(255, 255, 255, 0.7);
        border-bottom: 1px solid rgba(255, 255, 255, 0.7);
        border-left: 1px solid rgba(255, 255, 255, 0.7);
        border-right: 1px solid rgba(255, 255, 255, 0.7);
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
    .streamlit-expanderContent {{
        padding-left: 5px;
    }}
    .streamlit-expanderHeader {{
        color: white;
        padding-left: 0px;
        font-size: 1rem;
    }}
    .streamlit-expanderHeader > svg > path {{
        color: white;
    }}
    .element-container > div > div > h2 {{
        color: white;
    }}
    .stNumberInput > label {{
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
    div[attribute*="Settings"] {{
        font-size: 0.8rem !important;
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

#----------------------------------------------------------------------------------------------
#FUNCTION FOR LIGHT THEME

def theme_func_light():
    st.markdown(
        f"""
    <style>
    span[id*="MainMenu"]  {{
        display: none !important;
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
    div[data-testid*="stTable"] > table > thead > tr > th {{
        color: rgb(38, 39, 48);
        font-weight: 550;
    }}
    div[data-testid*="stTable"] > table > tbody > tr > th {{
        color: rgb(38, 39, 48);
        font-weight: 550;
    }}
    </style> 
    
    """,
        unsafe_allow_html=True,
    )

#----------------------------------------------------------------------------------------------
#FUNCTION FOR DOWNLOAD BUTTON THEME

def theme_func_dl_button():
    st.markdown(
        f"""
    <style>
    #button_dl {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background-color: rgb(255, 255, 255);
        color: rgb(38, 39, 48) !important;
        padding: .25rem .75rem;
        margin: 0px;
        position: relative;
        text-decoration: none;
        border-radius: 0.25rem;
        border-width: 1px;
        border-style: solid;
        border-color: rgb(230, 234, 241);
        border-image: initial;
    }} 
    #button_dl:hover {{
        border-color: #38bcf0;
        color: #38bcf0 !important;
    }}
    #button_dl:active {{
        box-shadow: none;
        background-color: #38bcf0;
        color: white;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

#----------------------------------------------------------------------------------------------
#FUNCTION FOR DETERMINING VARIABLE CATEGORY

def is_binary(data):
    df_cat = pd.DataFrame(index = ["cat"], columns = list(data))
    for i in data.columns:
        col = data[i]
        col.dropna(inplace=True)

        if col.dtypes == "float64" and col.unique().size > 2:
            df_cat.loc["cat"][i] = "numeric"

        elif col.dtypes == "float64" and col.unique().size == 2 and col.size > 2:
            df_cat.loc["cat"][i] = "binary"
        
        elif col.dtypes == "float64" and col.unique().size == 2 and col.size <= 2:
            df_cat.loc["cat"][i] = "numeric"
        
        elif col.dtypes == "float64" and col.unique().size == 1 and col.size == 1:
            df_cat.loc["cat"][i] = "numeric/single"
        
        elif col.dtypes == "float64" and col.unique().size == 1 and col.size > 1:
            df_cat.loc["cat"][i] = "numeric"

        elif col.dtypes == "float32" and col.unique().size > 2:
            df_cat.loc["cat"][i] = "numeric"

        elif col.dtypes == "float32" and col.unique().size == 2 and col.size > 2:
            df_cat.loc["cat"][i] = "binary"
        
        elif col.dtypes == "float32" and col.unique().size == 2 and col.size <= 2:
            df_cat.loc["cat"][i] = "numeric"
        
        elif col.dtypes == "float32" and col.unique().size == 1 and col.size == 1:
            df_cat.loc["cat"][i] = "numeric/single"
        
        elif col.dtypes == "float64" and col.unique().size == 1 and col.size > 1:
            df_cat.loc["cat"][i] = "numeric"

        elif col.dtypes == "bool":
            df_cat.loc["cat"][i] = "bool/binary"

        elif col.dtypes == "object" and col.unique().size > 2:
            df_cat.loc["cat"][i] = "string/categorical"

        elif col.dtypes == "object" and col.unique().size == 2:
            df_cat.loc["cat"][i] = "string/binary"
        
        elif col.dtypes == "object" and col.unique().size == 1 and col.size == 1:
            df_cat.loc["cat"][i] = "string/single"
        
        elif col.dtypes == "object" and col.unique().size == 1 and col.size > 1:
            df_cat.loc["cat"][i] = "string/categorical"

        elif col.dtypes == "int64" and col.unique().size == 2 and col.size > 2:
            df_cat.loc["cat"][i] = "binary"
        
        elif col.dtypes == "int64" and col.unique().size == 2 and col.size <= 2:
            df_cat.loc["cat"][i] = "numeric"

        elif col.dtypes == "int64" and col.unique().size == col.size and col.size > 1:
            df_cat.loc["cat"][i] = "numeric"
        
        elif col.dtypes == "int64" and col.unique().size == col.size and col.size == 1:
            df_cat.loc["cat"][i] = "numeric/single"

        elif col.dtypes == "int64" and col.unique().size > 2 and col.unique().size < col.size:
            df_cat.loc["cat"][i] = "categorical"
        
        elif col.dtypes == "int32" and col.unique().size == 2 and col.size > 2:
            df_cat.loc["cat"][i] = "binary"
        
        elif col.dtypes == "int32" and col.unique().size == 2 and col.size <= 2:
            df_cat.loc["cat"][i] = "numeric"
        
        elif col.dtypes == "int32" and col.unique().size == col.size and col.size > 1:
            df_cat.loc["cat"][i] = "numeric"
        
        elif col.dtypes == "int32" and col.unique().size == col.size and col.size == 1:
            df_cat.loc["cat"][i] = "numeric/single"
        
        elif col.dtypes == "int32" and col.unique().size > 2 and col.unique().size < col.size:
            df_cat.loc["cat"][i] = "categorical"

        else:
            df_cat.loc["cat"][i] = "other"

    return df_cat.loc["cat"]

#----------------------------------------------------------------------------------------------
#FUNCTION FOR DETERMINING UNIQUE VALUES PER VARIABLE

def is_unique(data):
    df_uni = pd.DataFrame(index = ["count"], columns = list(data))
    for i in data.columns:
        col = data[i]
        col.dropna(inplace=True)
        df_uni.loc["count"][i] = col.unique().size

    return df_uni.loc["count"]

#----------------------------------------------------------------------------------------------
#FUNCTION FOR DETERMINING MODE ACROSS ALL VARIABLE TYPES

def get_mode(data):
    df_mode = pd.DataFrame(index = ["mode", "n_unique"], columns = list(data))
    for i in data.columns:
        col = data[i]
        col.dropna(inplace=True)
        
        if col.dtypes == "float64" and col.unique().size > 2:
            df_mode.loc["mode"][i] = np.nan

        elif col.dtypes == "float32" and col.unique().size > 2:
            df_mode.loc["mode"][i] = np.nan

        elif col.dtypes == "float64" and col.unique().size == 2:
            df_mode.loc["mode"][i] = col.mode().iloc[0]
            if col.mode().unique().size > 1:
                df_mode.loc["mode"][i] = str(col.mode().iloc[0]) +" **"
                df_mode.loc["n_unique"][i] = "True"

        elif col.dtypes == "float32" and col.unique().size == 2:
            df_mode.loc["mode"][i] = col.mode().iloc[0]
            if col.mode().unique().size > 1:
                df_mode.loc["mode"][i] = str(col.mode().iloc[0]) +" **"
                df_mode.loc["n_unique"][i] = "True"

        elif col.dtypes == "bool":
            df_mode.loc["mode"][i] = col.mode().iloc[0]
            if col.mode().unique().size > 1:
                df_mode.loc["mode"][i] = str(col.mode().iloc[0]) +" **"
                df_mode.loc["n_unique"][i] = "True"

        elif col.dtypes == "object":
            if col.isnull().all() == False:
                df_mode.loc["mode"][i] = col.mode().iloc[0]
                if col.mode().unique().size > 1:
                    df_mode.loc["mode"][i] = str(col.mode().iloc[0]) +" **"
                    df_mode.loc["n_unique"][i] = "True"
            else: 
                df_mode.loc["mode"][i] = np.nan 

        elif col.dtypes == "int64" and col.unique().size > 2:
            df_mode.loc["mode"][i] = col.mode().iloc[0]
            if col.mode().unique().size > 1:
                df_mode.loc["mode"][i] = str(col.mode().iloc[0]) +" **"
                df_mode.loc["n_unique"][i] = "True"

        elif col.dtypes == "int32" and col.unique().size > 2:
            df_mode.loc["mode"][i] = col.mode().iloc[0]
            if col.mode().unique().size > 1:
                df_mode.loc["mode"][i] = str(col.mode().iloc[0]) +" **"
                df_mode.loc["n_unique"][i] = "True"

        elif col.dtypes == "int64" and col.unique().size == 2:
            df_mode.loc["mode"][i] = col.mode().iloc[0]
            if col.mode().unique().size > 1:
                df_mode.loc["mode"][i] = str(col.mode().iloc[0]) +" **"
                df_mode.loc["n_unique"][i] = "True"
        
        elif col.dtypes == "int32" and col.unique().size == 2:
            df_mode.loc["mode"][i] = col.mode().iloc[0]
            if col.mode().unique().size > 1:
                df_mode.loc["mode"][i] = str(col.mode().iloc[0]) +" **"
                df_mode.loc["n_unique"][i] = "True"

        else:
            df_mode.loc["mode"][i] = np.nan

    return df_mode

#----------------------------------------------------------------------------------------------
#FUNCTION FOR DETERMINING MAIN QUANTILES ACROSS ALL VARIABLE TYPES

def get_mainq(data):
    df_mainq = pd.DataFrame(index = ["min","1%-Q", "10%-Q", "25%-Q", "75%-Q","90%-Q", "99%-Q", "max"], columns = list(data))
    for i in data.columns:
        col = data[i]
        col.dropna(inplace=True)
        
        if col.dtypes == "float64" or col.dtypes == "int64" or col.dtypes == "float32" or col.dtypes == "int32":
            df_mainq.loc["min"][i] = col.min()
            df_mainq.loc["1%-Q"][i] = col.quantile(q = 0.01)
            df_mainq.loc["10%-Q"][i] = col.quantile(q = 0.1)
            df_mainq.loc["25%-Q"][i] = col.quantile(q = 0.25)
            df_mainq.loc["75%-Q"][i] = col.quantile(q = 0.75)
            df_mainq.loc["90%-Q"][i] = col.quantile(q = 0.9)
            df_mainq.loc["99%-Q"][i] = col.quantile(q = 0.99)
            df_mainq.loc["max"][i] = col.max()

        elif col.dtypes == "bool":
            df_mainq.loc["min"][i] = col.min()
            df_mainq.loc["1%-Q"][i] = col.quantile(q = 0.01)
            df_mainq.loc["10%-Q"][i] = col.quantile(q = 0.1)
            df_mainq.loc["25%-Q"][i] = col.quantile(q = 0.25)
            df_mainq.loc["75%-Q"][i] = col.quantile(q = 0.75)
            df_mainq.loc["90%-Q"][i] = col.quantile(q = 0.9)
            df_mainq.loc["99%-Q"][i] = col.quantile(q = 0.99)
            df_mainq.loc["max"][i] = col.max()

        elif col.dtypes == "object":
            df_mainq.loc["min"][i] = np.nan
            df_mainq.loc["1%-Q"][i] = np.nan
            df_mainq.loc["10%-Q"][i] = np.nan
            df_mainq.loc["25%-Q"][i] = np.nan
            df_mainq.loc["75%-Q"][i] = np.nan
            df_mainq.loc["90%-Q"][i] = np.nan
            df_mainq.loc["99%-Q"][i] = np.nan
            df_mainq.loc["max"][i] = np.nan

        else:
            df_mainq.loc["min"][i] = np.nan
            df_mainq.loc["1%-Q"][i] = np.nan
            df_mainq.loc["10%-Q"][i] = np.nan
            df_mainq.loc["25%-Q"][i] = np.nan
            df_mainq.loc["75%-Q"][i] = np.nan
            df_mainq.loc["90%-Q"][i] = np.nan
            df_mainq.loc["99%-Q"][i] = np.nan
            df_mainq.loc["max"][i] = np.nan

    return df_mainq

#---------------------------------------------------------------------------------------------
#FUNCTION FOR DETERMINING MEASURES OF SHAPE ACROSS ALL VARIABLE TYPES

def get_shape(data):
    df_shape = pd.DataFrame(index = ["skewness", "kurtosis"], columns = list(data))
    for i in data.columns:
        col = data[i]
        col.dropna(inplace=True)
        
        if col.dtypes == "float64" or col.dtypes == "int64" or col.dtypes == "float32" or col.dtypes == "int32":
            df_shape.loc["skewness"][i] = col.skew()
            df_shape.loc["kurtosis"][i] = col.kurtosis()

        elif col.dtypes == "bool" or col.dtypes == "object":
            df_shape.loc["skewness"][i] = np.nan
            df_shape.loc["kurtosis"][i] = np.nan

        else:
            df_shape.loc["min"][i] = np.nan
            df_shape.loc["1%-Q"][i] = np.nan
            df_shape.loc["10%-Q"][i] = np.nan
            df_shape.loc["25%-Q"][i] = np.nan
            df_shape.loc["75%-Q"][i] = np.nan
            df_shape.loc["90%-Q"][i] = np.nan
            df_shape.loc["99%-Q"][i] = np.nan
            df_shape.loc["max"][i] = np.nan

    return df_shape

#---------------------------------------------------------------------------------------------
#FUNCTION FOR IMPUTATION

def data_impute(data, numeric_method, other_method):
    nrows = data.shape[0]
    for i in data.columns:
        
        # Count rows without NAs
        if data[i].shape[0]-data[i].count() > 0:
            nrows_woNa = data[i].count()
        else:
            nrows_woNa = 0

        # Check if rows with NAs exist in the corresponding column
        if nrows - nrows_woNa > 0:
            # Check if type is numeric
            if data[i].dtypes == "float64" or data[i].dtypes == "float32":
                if numeric_method == "Mean":
                    data[i] = data[i].replace(np.nan, data[i].mean())
                if numeric_method == "Median":
                    data[i] = data[i].replace(np.nan, data[i].median())
                if numeric_method == "Random value":
                    data[i] = data[i].apply(lambda x: x if not np.isnan(x) else np.random.choice(data[i][~data[i].isna()]))
            # Check if type is other
            elif data[i].dtypes != "float64" and data[i].dtypes != "float32":
                if other_method == "Mode":
                    data[i] = data[i].replace(np.nan, data[i].mode()[0])
                if other_method == "Random value":
                    data[i] = data[i].apply(lambda x: x if not pd.isnull(x) else np.random.choice(data[i][~data[i].isna()]))

    return data

#---------------------------------------------------------------------------------------------
#FUNCTION FOR IMPUTATION (PANEL DATA)

def data_impute_panel(data, numeric_method, other_method, group_num, group_other, entity, time):

    nrows = data.shape[0]
    for i in data.columns:
        
        # Count rows without NAs
        if data[i].shape[0]-data[i].count() > 0:
            nrows_woNa = data[i].count()
        else:
            nrows_woNa = 0

        # Check if rows with NAs exist in the corresponding column
        if nrows - nrows_woNa > 0:
            # Check if type is numeric
            if data[i].dtypes == "float64" or data[i].dtypes == "float32":
                # No grouping
                if group_num == "None":
                    if numeric_method == "Mean":
                        data[i] = data[i].replace(np.nan, data[i].mean())
                    if numeric_method == "Median":
                        data[i] = data[i].replace(np.nan, data[i].median())
                    if numeric_method == "Random value":
                        data[i] = data[i].apply(lambda x: x if not np.isnan(x) else np.random.choice(data[i][~data[i].isna()]))
                # Group by entity
                if group_num == "Entity":
                    data_grouped = data.groupby(entity)
                    if numeric_method == "Mean":
                        data_grouped_mean = data_grouped.mean()
                        for x in data[i].index:
                            if np.isnan(data[i][x]):
                                data[i][x] = data_grouped_mean[i][data.loc[x][entity]]
                    if numeric_method == "Median":
                        data_grouped_median = data_grouped.median()
                        for x in data[i].index:
                            if np.isnan(data[i][x]):
                                data[i][x] = data_grouped_median[i][data.loc[x][entity]]
                    if numeric_method == "Random value":
                        for x in data[i].index:
                            if np.isnan(data[i][x]):
                                entity_data = data[data[entity] == data.loc[x][entity]][i]
                                if entity_data[~entity_data.isna()].size != 0:
                                    data[i][x] = np.random.choice(entity_data[~entity_data.isna()])
                # Group by time
                if group_num == "Time":
                    data_grouped = data.groupby(time)
                    if numeric_method == "Mean":
                        data_grouped_mean = data_grouped.mean()
                        for x in data[i].index:
                            if np.isnan(data[i][x]):
                                data[i][x] = data_grouped_mean[i][data.loc[x][time]]
                    if numeric_method == "Median":
                        data_grouped_median = data_grouped.median()
                        for x in data[i].index:
                            if np.isnan(data[i][x]):
                                data[i][x] = data_grouped_median[i][data.loc[x][time]]
                    if numeric_method == "Random value":
                        for x in data[i].index:
                            if np.isnan(data[i][x]):
                                time_data = data[data[time] == data.loc[x][time]][i]
                                if time_data[~time_data.isna()].size != 0:
                                    data[i][x] = np.random.choice(time_data[~time_data.isna()])

            # Check if type is other
            elif data[i].dtypes != "float64" and data[i].dtypes != "float32":
                # No grouping
                if group_other == "None":
                    if other_method == "Mode":
                        data[i] = data[i].replace(np.nan, data[i].mode()[0])
                    if other_method == "Random value":
                        data[i] = data[i].apply(lambda x: x if not pd.isnull(x) else np.random.choice(data[i][~data[i].isna()]))
                # Group by entity
                if group_other == "Entity":
                    if other_method == "Mode":
                        for x in data[i].index:
                            if pd.isnull(data[i][x]):
                                entity_data = data[data[entity] == data.loc[x][entity]][i]
                                if entity_data[~entity_data.isna()].size != 0:
                                    data[i][x] =  entity_data.mode()[0]
                    if other_method == "Random value":
                        for x in data[i].index:
                            if pd.isnull(data[i][x]):
                                entity_data = data[data[entity] == data.loc[x][entity]][i]
                                if entity_data[~entity_data.isna()].size != 0:
                                    data[i][x] =  np.random.choice(entity_data[~entity_data.isna()])
                # Group by time
                if group_other == "Time":
                    if other_method == "Mode":
                        for x in data[i].index:
                            if pd.isnull(data[i][x]):
                                time_data = data[data[time] == data.loc[x][time]][i]
                                if time_data[~time_data.isna()].size != 0:
                                    data[i][x] =  time_data.mode()[0]
                    if other_method == "Random value":
                        for x in data[i].index:
                            if pd.isnull(data[i][x]):
                                time_data = data[data[time] == data.loc[x][time]][i]
                                if time_data[~time_data.isna()].size != 0:
                                    data[i][x] =  np.random.choice(time_data[~time_data.isna()])

    return data

#---------------------------------------------------------------------------------------------
#FUNCTION FOR GROUPED IMPUTATION

def data_impute_grouped(data, numeric_method, other_method, group_num, group_other):

    nrows = data.shape[0]
    for i in data.columns:
        
        # Count rows without NAs
        if data[i].shape[0]-data[i].count() > 0:
            nrows_woNa = data[i].count()
        else:
            nrows_woNa = 0

        # Check if rows with NAs exist in the corresponding column
        if nrows - nrows_woNa > 0:
            # Check if type is numeric
            if data[i].dtypes == "float64" or data[i].dtypes == "float32":
                # No grouping
                if group_num == "None":
                    if numeric_method == "Mean":
                        data[i] = data[i].replace(np.nan, data[i].mean())
                    if numeric_method == "Median":
                        data[i] = data[i].replace(np.nan, data[i].median())
                    if numeric_method == "Random value":
                        data[i] = data[i].apply(lambda x: x if not np.isnan(x) else np.random.choice(data[i][~data[i].isna()]))
                # Grouping
                if group_num != "None":
                    data_grouped = data.groupby(group_num)
                    if numeric_method == "Mean":
                        data_grouped_mean = data_grouped.mean()
                        for x in data[i].index:
                            if np.isnan(data[i][x]):
                                data[i][x] = data_grouped_mean[i][data.loc[x][group_num]]
                    if numeric_method == "Median":
                        data_grouped_median = data_grouped.median()
                        for x in data[i].index:
                            if np.isnan(data[i][x]):
                                data[i][x] = data_grouped_median[i][data.loc[x][group_num]]
                    if numeric_method == "Random value":
                        for x in data[i].index:
                            if np.isnan(data[i][x]):
                                entity_data = data[data[group_num] == data.loc[x][group_num]][i]
                                if entity_data[~entity_data.isna()].size != 0:
                                    data[i][x] = np.random.choice(entity_data[~entity_data.isna()])

            # Check if type is other
            elif data[i].dtypes != "float64" and data[i].dtypes != "float32":
                # No grouping
                if group_other == "None":
                    if other_method == "Mode":
                        data[i] = data[i].replace(np.nan, data[i].mode()[0])
                    if other_method == "Random value":
                        data[i] = data[i].apply(lambda x: x if not pd.isnull(x) else np.random.choice(data[i][~data[i].isna()]))
                # Grouping
                if group_other != "None":
                    if other_method == "Mode":
                        for x in data[i].index:
                            if pd.isnull(data[i][x]):
                                entity_data = data[data[group_other] == data.loc[x][group_other]][i]
                                if entity_data[~entity_data.isna()].size != 0:
                                    data[i][x] =  entity_data.mode()[0]
                    if other_method == "Random value":
                        for x in data[i].index:
                            if pd.isnull(data[i][x]):
                                entity_data = data[data[group_other] == data.loc[x][group_other]][i]
                                if entity_data[~entity_data.isna()].size != 0:
                                    data[i][x] =  np.random.choice(entity_data[~entity_data.isna()])

    return data

#---------------------------------------------------------------------------------------------
#FUNCTION FOR TRANSFORMING VARIABLE TO LOG_VARIABLE

def var_transform_log(data, var_list):
    for i in var_list:

        # transform only if numeric     
        if data[i].dtypes == "float64" or data[i].dtypes == "int64" or data[i].dtypes == "float32" or data[i].dtypes == "int32":
            new_var_name = "log_" + i
            if data[i].min() > 0:
                new_var = np.log(data[i])
                data[new_var_name] = new_var 
            elif data[i].min() <= 0:
                new_var = np.log(data[i]+data[i].min()+1)
                data[new_var_name] = new_var 

    return data

#---------------------------------------------------------------------------------------------
#FUNCTION FOR TRANSFORMING VARIABLE TO SQRT_VARIABLE

def var_transform_sqrt(data, var_list):
    for i in var_list:

        # transform only if numeric     
        if data[i].dtypes == "float64" or data[i].dtypes == "int64" or data[i].dtypes == "float32" or data[i].dtypes == "int32":
            new_var_name = "sqrt_" + i
            if data[i].min() > 0:
                new_var = np.sqrt(data[i])
                data[new_var_name] = new_var
            elif data[i].min() <= 0:
                new_var = np.sqrt(data[i]+data[i].min()+1)
                data[new_var_name] = new_var  

    return data

#---------------------------------------------------------------------------------------------
#FUNCTION FOR TRANSFORMING VARIABLE TO SQUARE_VARIABLE

def var_transform_square(data, var_list):
    for i in var_list:

        # transform only if numeric     
        if data[i].dtypes == "float64" or data[i].dtypes == "int64" or data[i].dtypes == "float32" or data[i].dtypes == "int32":
            new_var_name = "square_" + i
            new_var = np.square(data[i])
            data[new_var_name] = new_var 

    return data

#---------------------------------------------------------------------------------------------
#FUNCTION FOR TRANSFORMING VARIABLE TO CENT_VARIABLE

def var_transform_cent(data, var_list):
    for i in var_list:

        # transform only if numeric     
        if data[i].dtypes == "float64" or data[i].dtypes == "int64" or data[i].dtypes == "float32" or data[i].dtypes == "int32":
            new_var_name = "cent_" + i
            new_var = (data[i] - data[i].mean())
            data[new_var_name] = new_var 

    return data

#---------------------------------------------------------------------------------------------
#FUNCTION FOR TRANSFORMING VARIABLE TO STAND_VARIABLE

def var_transform_stand(data, var_list):
    for i in var_list:

        # transform only if numeric     
        if data[i].dtypes == "float64" or data[i].dtypes == "int64" or data[i].dtypes == "float32" or data[i].dtypes == "int32":
            if data[i].std() != 0:
                new_var_name = "stand_" + i
                new_var = (data[i] - data[i].mean())/data[i].std()
                data[new_var_name] = new_var 

    return data

#---------------------------------------------------------------------------------------------
#FUNCTION FOR TRANSFORMING VARIABLE TO NORM_VARIABLE

def var_transform_norm(data, var_list):
    for i in var_list:

        # transform only if numeric     
        if data[i].dtypes == "float64" or data[i].dtypes == "int64" or data[i].dtypes == "float32" or data[i].dtypes == "int32":
            if (data[i].max()-data[i].min()) != 0:
                new_var_name = "norm_" + i
                new_var = (data[i] - data[i].min())/(data[i].max()-data[i].min())
                data[new_var_name] = new_var 

    return data

#---------------------------------------------------------------------------------------------
#FUNCTION FOR TRANSFORMING VARIABLE TO NUMCAT_VARIABLE

def var_transform_numCat(data, var_list):
    for i in var_list:
        col = data[i]
        col.dropna(inplace = True)
        cats = pd.DataFrame(index = range(col.unique().size), columns = ["cat", "numCat"])
        cats["cat"] = sorted(col.unique())
        cats["numCat"] = range(col.unique().size)

        new_var_name = "numCat_" + i
        new_var = pd.DataFrame(index = data.index, columns = [new_var_name])

        for c in data.index:
            if pd.isnull(data[i][c]) == True:
                new_var.loc[c, new_var_name] = np.nan
            elif pd.isnull(data[i][c]) == False:
                new_var.loc[c, new_var_name] = int(cats[cats["cat"] == data[i][c]]["numCat"])
        
        data[new_var_name] = new_var.astype('int64')

    return data

#---------------------------------------------------------------------------------------------
#FUNCTION FOR VARIABLE MULTIPLICATION

def var_transform_mult(data, var_1, var_2):

    # transform only if numeric     
    if data[var_1].dtypes == "float64" or data[var_1].dtypes == "int64" or data[var_1].dtypes == "float32" or data[var_1].dtypes == "int32":
        if data[var_2].dtypes == "float64" or data[var_2].dtypes == "int64" or data[var_2].dtypes == "float32" or data[var_2].dtypes == "int32":
            new_var_name = "mult_" + var_1 + "_" + var_2
            new_var = data[var_1]*data[var_2]
            data[new_var_name] = new_var 

    return data

#---------------------------------------------------------------------------------------------
#FUNCTION FOR VARIABLE DIVISION

def var_transform_div(data, var_1, var_2):

    # transform only if numeric     
    if data[var_1].dtypes == "float64" or data[var_1].dtypes == "int64" or data[var_1].dtypes == "float32" or data[var_1].dtypes == "int32":
        if data[var_2].dtypes == "float64" or data[var_2].dtypes == "int64" or data[var_2].dtypes == "float32" or data[var_2].dtypes == "int32":
            new_var_name = "div_" + var_1 + "_" + var_2
            new_var = data[var_1]/data[var_2]
            data[new_var_name] = new_var 

    return data

#---------------------------------------------------------------------------------------------
#FUNCTION FOR COMBINED DATA SUMMARY

def data_summary(data):
    
    # Variable types
    df_summary_vt = pd.DataFrame(index = ["data type","category", "unique values", "count", "missing values"], columns = list(data))
    df_summary_vt.loc["data type"] = data.dtypes
    df_summary_vt.loc["category"] = is_binary(data)
    df_summary_vt.loc["unique values"] = is_unique(data)
    df_summary_vt.loc["count"] = data.count()
    df_summary_vt.loc["missing values"] = data.shape[0]-data.count()
        
    # Measures of central tendency
    df_summary_ct = pd.DataFrame(index = ["mean", "mode", "median"], columns = list(data))
    df_summary_ct.loc["mean"] = data.mean()
    df_summary_ct.loc["mode"] = get_mode(data).loc["mode"]
    df_summary_ct.loc["median"] = data.median()

    # Measures of dispersion
    df_summary_mod = pd.DataFrame(index = ["standard deviation", "variance"], columns = list(data))
    df_summary_mod.loc["standard deviation"] = data.std()
    df_summary_mod.loc["variance"] = data.var()

    # Measures of shape
    df_summary_mos = get_shape(data)

    # Main quantiles
    df_summary_mq_full = get_mainq(data)
    df_summary_mq = pd.DataFrame(index = ["min", "1%-Q", "10%-Q", "25%-Q", "75%-Q","90%-Q", "99%-Q", "max"], columns = list(data))
    df_summary_mq.loc["min"] = df_summary_mq_full.loc["min"]
    df_summary_mq.loc["1%-Q"] = df_summary_mq_full.loc["1%-Q"]
    df_summary_mq.loc["10%-Q"] = df_summary_mq_full.loc["10%-Q"]
    df_summary_mq.loc["25%-Q"] = df_summary_mq_full.loc["25%-Q"]
    df_summary_mq.loc["75%-Q"] = df_summary_mq_full.loc["75%-Q"]
    df_summary_mq.loc["90%-Q"] = df_summary_mq_full.loc["90%-Q"]
    df_summary_mq.loc["99%-Q"] = df_summary_mq_full.loc["99%-Q"]
    df_summary_mq.loc["max"] = df_summary_mq_full.loc["max"]

    # Combine dataframes
    summary_collection = {}
    summary_collection["Variable types"] = df_summary_vt
    summary_collection["Measures of central tendency"] = df_summary_ct
    summary_collection["Measures of dispersion"] = df_summary_mod
    summary_collection["Measures of shape"] = df_summary_mos
    summary_collection["Main quantiles"] = df_summary_mq
    summary_collection["ALL"] = pd.concat([df_summary_ct,df_summary_mod,df_summary_mos,df_summary_mq])
    return summary_collection

#------------------------------------------------------------------------------------------
#FUNCTION FOR 2D HISTOGRAM

def compute_2d_histogram(var1, var2, data, density = False):
    H, xedges, yedges = np.histogram2d(data[var1], data[var2], density = density)
    H[H == 0] = np.nan

    # Create a nice variable that shows the bin boundaries
    xedges = pd.Series(['{0:.4g}'.format(num) for num in xedges])
    xedges = pd.DataFrame({"a": xedges.shift(), "b": xedges}).dropna().agg(' - '.join, axis=1)
    yedges = pd.Series(['{0:.4g}'.format(num) for num in yedges])
    yedges = pd.DataFrame({"a": yedges.shift(), "b": yedges}).dropna().agg(' - '.join, axis=1)

    # Cast to long format using melt
    res = pd.DataFrame(H, 
        index = yedges, 
        columns = xedges).reset_index().melt(
                id_vars = 'index'
        ).rename(columns = {'index': 'value2', 
                            'value': 'count',
                            'variable': 'value'})
    

    # Also add the raw left boundary of the bin as a column, will be used to sort the axis labels later
    res['raw_left_value'] = res['value'].str.split(' - ').map(lambda x: x[0]).astype(float)   
    res['raw_left_value2'] = res['value2'].str.split(' - ').map(lambda x: x[0]).astype(float) 
    res['variable'] = var1
    res['variable2'] = var2 
    return res.dropna() # Drop all combinations for which no values where found

#------------------------------------------------------------------------------------------
#FUNCTION FOR NON-ROBUST HAUSMAN-TEST (PANEL DATA MODELLING)

def hausman_test(fixed_effects, random_effects):

    # Extract information for statistic
    b_fe = fixed_effects.params
    b_re = random_effects.params.drop(["const"])
    variance_b_fe = fixed_effects.cov
    variance_b_re = random_effects.cov.drop(["const"])
    variance_b_re = variance_b_re.drop(["const"], axis = 1)
    df = b_fe[np.abs(b_fe) < 1e8].size

    # Wu - Hausman statistic
    chi2 = np.dot((b_fe - b_re).T, np.linalg.inv(variance_b_fe - variance_b_re).dot(b_fe - b_re)) 
    
    # Get p-value
    pval = stats.chi2.sf(chi2, df)

    return chi2, df, pval

#------------------------------------------------------------------------------------------
#FUNCTION FOR DISTRIBUTION FITTING

def fit_scipy_dist(ft_data, Nobins, ft_dist,ft_low,ft_up):
    # ft_data- an array with empirical data
    # Nobins - number of bins in the emp. distribution
    # ft_dist -selected theoretical distributions (scipy)                
        
    # frequency analysis (empirical)
    if ft_low !=None and ft_up !=None:
        y, x = np.histogram(np.array(ft_data),range=(ft_low,ft_up), bins=Nobins)
    else:    
        y, x = np.histogram(np.array(ft_data), bins=Nobins)
    bin_width = x[1]-x[0]
    N = len(ft_data)
    x_mid = (x + np.roll(x, -1))[:-1] / 2.0 
    xlower=x[:-1]
    xupper=x[1:] 
    k= sum(i > 0 for i in y) # number of non-zero classes
                
    # dist name translator (ordinary name: scipy name)
    dist_name={'Alpha':scipy.stats.alpha,'Anglit':scipy.stats.anglit,'Arcsine':scipy.stats.arcsine,'Argus':scipy.stats.argus,'Beta':scipy.stats.beta,'Beta prime':scipy.stats.betaprime,'Bradford':scipy.stats.bradford,'Burr (Type III)':scipy.stats.burr,'Burr (Type XII)':scipy.stats.burr12,'Cauchy':scipy.stats.cauchy,'Chi':scipy.stats.chi,'Chi-squared':scipy.stats.chi2,'Cosine':scipy.stats.cosine,'Crystalball':scipy.stats.crystalball,'Double gamma':scipy.stats.dgamma,'Double Weibull':scipy.stats.dweibull,'Erlang':scipy.stats.erlang,'Exponential':scipy.stats.expon,'Exponentially modified Normal':scipy.stats.exponnorm,'Exponentiated Weibull':scipy.stats.exponweib,'Exponential power':scipy.stats.exponpow,'F':scipy.stats.f,'Fatigue-life (Birnbaum-Saunders)':scipy.stats.fatiguelife,'Fisk':scipy.stats.fisk,'Folded Cauchy':scipy.stats.foldcauchy,'Folded normal':scipy.stats.foldnorm,'Generalized logistic':scipy.stats.genlogistic,'Generalized normal':scipy.stats.gennorm,'Generalized Pareto':scipy.stats.genpareto,'Generalized exponential':scipy.stats.genexpon,'Generalized extreme value':scipy.stats.genextreme,'Gauss hypergeometric':scipy.stats.gausshyper,'Gamma':scipy.stats.gamma,'Generalized gamma':scipy.stats.gengamma,'Generalized half-logistic':scipy.stats.genhalflogistic,'Generalized Inverse Gaussian':scipy.stats.geninvgauss,'Gilbrat':scipy.stats.gilbrat,'Gompertz (or truncated Gumbel)':scipy.stats.gompertz,'Right-skewed Gumbel':scipy.stats.gumbel_r,'Left-skewed Gumbel':scipy.stats.gumbel_l,'Half-Cauchy':scipy.stats.halfcauchy,'Half-logistic':scipy.stats.halflogistic,'Half-normal':scipy.stats.halfnorm,'The upper half of a generalized normal':scipy.stats.halfgennorm,'Hyperbolic secant':scipy.stats.hypsecant,'Inverted gamma':scipy.stats.invgamma,'Inverse Gaussian':scipy.stats.invgauss,'Inverted Weibull':scipy.stats.invweibull,'Johnson SB':scipy.stats.johnsonsb,'Johnson SU':scipy.stats.johnsonsu,'Kappa 4 parameter':scipy.stats.kappa4,'Kappa 3 parameter':scipy.stats.kappa3,'Kolmogorov-Smirnov one-sided test statistic':scipy.stats.ksone,'Kolmogorov-Smirnov two-sided test statistic':scipy.stats.kstwo,'Limiting distribution of scaled Kolmogorov-Smirnov two-sided test statistic':scipy.stats.kstwobign,'Laplace':scipy.stats.laplace,'Asymmetric Laplace':scipy.stats.laplace_asymmetric,'Levy':scipy.stats.levy,'Left-skewed Levy':scipy.stats.levy_l,'Levy-stable':scipy.stats.levy_stable,'Logistic (or Sech-squared)':scipy.stats.logistic,'Log gamma':scipy.stats.loggamma,'Log-Laplace':scipy.stats.loglaplace,'Lognormal':scipy.stats.lognorm,'Loguniform or reciprocal':scipy.stats.loguniform,'Lomax (Pareto of the second kind)':scipy.stats.lomax,'Maxwell':scipy.stats.maxwell,'Mielke Beta-Kappa / Dagum':scipy.stats.mielke,'Moyal':scipy.stats.moyal,'Nakagami':scipy.stats.nakagami,'Non-central chi-squared':scipy.stats.ncx2,'Non-central F distribution':scipy.stats.ncf,'Non-central Student’s t':scipy.stats.nct,'Normal':scipy.stats.norm,'Normal Inverse Gaussian':scipy.stats.norminvgauss,'Pareto':scipy.stats.pareto,'Pearson type III':scipy.stats.pearson3,'Power-function':scipy.stats.powerlaw,'Power log-normal':scipy.stats.powerlognorm,'Power normal':scipy.stats.powernorm,'R-distributed (symmetric beta)':scipy.stats.rdist,'Rayleigh':scipy.stats.rayleigh,'Rice':scipy.stats.rice,'Reciprocal inverse Gaussian':scipy.stats.recipinvgauss,'Semicircular':scipy.stats.semicircular,'Skew-normal':scipy.stats.skewnorm,'Student’s t':scipy.stats.t,'Trapezoidal':scipy.stats.trapezoid,'Triangular':scipy.stats.triang,'Truncated exponential':scipy.stats.truncexpon,'Truncated normal':scipy.stats.truncnorm,'Tukey-Lamdba':scipy.stats.tukeylambda,'Uniform':scipy.stats.uniform,'Von Mises':scipy.stats.vonmises,'Von Mises':scipy.stats.vonmises_line,'Wald':scipy.stats.wald,'Weibull minimum':scipy.stats.weibull_min,'Weibull maximum':scipy.stats.weibull_max,'Wrapped Cauchy':scipy.stats.wrapcauchy}
    
    #------------------
    # Chi-Square Goodness-of-Fit Test:
    chiSquared=[]
    clas_prob_res=[]
    
    ra_bar = st.progress(0.0)
    progress = 0

    for dist in ft_dist:
        dist_LongName=dist
        dist=dist_name[dist]
        dist_ShortName = dist.__class__.__name__[:-4]
        # get dist paramaters    
        params = dist.fit(np.array(ft_data))
        No_para=len(params)
        chi_dof=k-(No_para+1)
            
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        
        #calculate theo. prob for the class & squared emp-theo difs.
        clas_prob=dist.cdf(xupper,loc=loc,scale=scale, *arg)-dist.cdf(xlower,loc=loc,scale=scale, *arg)
        clas_theo_freq=N*clas_prob
        ssd = np.sum((y - clas_theo_freq)**2)
        chi_val= np.sum(((y-clas_theo_freq)**2)/clas_theo_freq)
        chi_p_val=1 -stats.chi2.cdf(chi_val, chi_dof)
                
        clas_prob_res.append(clas_theo_freq/N)  
        #ssd = np.sum((y - clas_prob)**2)
        chiSquared.append([ssd, chi_val, chi_dof, chi_p_val, dist_ShortName, dist_LongName])
        
        progress += 1
        ra_bar.progress(progress/len(ft_dist))
    #------------------
        
    # Output                    
    clas_prob_res=(pd.DataFrame(clas_prob_res)).transpose()
    results = pd.DataFrame(chiSquared, columns = ['SSD','Chi-squared','DOF', 'p-value', 'Distribution abr.', 'Distribution'])
    min_ssd=results['SSD'].idxmin()
    results=results.sort_values(by='p-value',ascending=False) 
    max_p=results.iloc[0]['Distribution']
                
    results=results.sort_values(by='SSD') 
    best_name = results.iloc[0]['Distribution']
    best_dist = getattr(scipy.stats, results.iloc[0]['Distribution abr.'])
    best_params = best_dist.fit(np.array(ft_data))
    loc = best_params[-2]
    scale = best_params[-1]
    
    return results, x_mid,xlower,xupper, min_ssd, max_p, clas_prob_res, best_name, best_dist, best_params                                

#------------------------------------------------------------------------------------------
#FUNCTION FOR UNIVARIATE REGRESSION MODELS

def regression_models(X_ini, Y_ini, expl_var,reg_technique,poly_order):            
    # initialisation
    mlr_reg_inf = pd.DataFrame(index = ["Dep. variable",  "Method", "No. observations"], columns = ["Value"])
    mlr_reg_stats = pd.DataFrame(index = ["R²", "Adj. R²", "Mult. corr. coeff.", "Residual SE", "Log-likelihood", "AIC", "BIC"], columns = ["Value"])
    mlr_reg_anova = pd.DataFrame(index = ["Regression", "Residual", "Total"], columns = ["DF", "SS", "MS", "F-statistic", "p-value"])
    poly_data=pd.DataFrame(X_ini)
    

    if reg_technique=='Polynomial Regression':
        for i in range(1,poly_order):
            var_name=expl_var+str(i+1)
            poly_data[var_name]=np.power(X_ini,(i+1))
        
        mlr_reg_coef = pd.DataFrame(index = sm.add_constant(poly_data).columns, columns = ["coeff", "std err", "t-statistic", "p-value", "lower 95%", "upper 95%"])

    else:
            mlr_reg_coef = pd.DataFrame(index = ["const", str(expl_var)], columns = ["coeff", "std err", "t-statistic", "p-value", "lower 95%", "upper 95%"])

           
    # ['Simple Linear Regression', 'Linear-Log Regression', 'Log-Linear Regression', 'Log-Log Regression','Polynomial Regression']
                
    if reg_technique=='Simple Linear Regression':
        X_data=X_ini
        Y_data=Y_ini
    elif reg_technique=='Linear-Log Regression':
        X_data=np.log(X_ini)
        Y_data=Y_ini
    elif reg_technique=='Log-Linear Regression':
        Y_data=np.log(Y_ini)
        X_data=X_ini
    elif reg_technique=='Log-Log Regression':
        X_data=np.log(X_ini)
        Y_data=np.log(Y_ini)
    elif reg_technique=='Polynomial Regression':
        X_data=poly_data
        Y_data=Y_ini
                

    # Train MLR model (statsmodels)
    X_data_mlr = sm.add_constant(X_data)
    full_model_mlr = sm.OLS(Y_data, X_data_mlr)             
    full_model_fit = full_model_mlr.fit()              
    Y_pred = full_model_fit.predict(X_data_mlr)
    Y_pred = Y_pred.to_numpy()

    # Train MLR model (sklearn)
    full_model_mlr_sk = LinearRegression()
    if reg_technique !='Polynomial Regression':
        X_data_sk=np.array(X_data).reshape(1, -1)
        Y_data_sk=np.array(Y_data).reshape(1, -1)
        full_model_mlr_sk.fit(X_data_sk, Y_data_sk)
    else:
        full_model_mlr_sk.fit(X_data, Y_data)   
        
    # Basic data information
    mlr_reg_inf.loc["Method"] = reg_technique
    mlr_reg_inf.loc["Dep. variable"] = expl_var
    mlr_reg_inf.loc["No. observations"] = full_model_fit.model.nobs
    #st.write(mlr_reg_inf)

    # Statistics
    mlr_reg_stats.loc["R²"] = full_model_fit.rsquared
    mlr_reg_stats.loc["Adj. R²"] = full_model_fit.rsquared_adj
    mlr_reg_stats.loc["Mult. corr. coeff."] = np.sqrt(full_model_fit.rsquared)
    mlr_reg_stats.loc["Residual SE"] = np.sqrt(full_model_fit.mse_resid)
    mlr_reg_stats.loc["Log-likelihood"] = full_model_fit.llf
    mlr_reg_stats.loc["AIC"] = full_model_fit.aic
    mlr_reg_stats.loc["BIC"] = full_model_fit.bic
    #st.write(mlr_reg_stats)

    # ANOVA
    mlr_reg_anova.loc["Regression"]["DF"] = full_model_fit.df_model
    mlr_reg_anova.loc["Regression"]["SS"] = (full_model_fit.ess).round(4)
    mlr_reg_anova.loc["Regression"]["MS"] = (full_model_fit.ess/full_model_fit.df_model).round(4)
    mlr_reg_anova.loc["Regression"]["F-statistic"] = (full_model_fit.fvalue).round(4)
    mlr_reg_anova.loc["Regression"]["p-value"] = (full_model_fit.f_pvalue).round(4)
    mlr_reg_anova.loc["Residual"]["DF"] = full_model_fit.df_resid
    mlr_reg_anova.loc["Residual"]["SS"] = (full_model_fit.ssr).round(4)
    mlr_reg_anova.loc["Residual"]["MS"] = (full_model_fit.ssr/full_model_fit.df_resid).round(4)
    mlr_reg_anova.loc["Residual"]["F-statistic"] = ""
    mlr_reg_anova.loc["Residual"]["p-value"] = ""
    mlr_reg_anova.loc["Total"]["DF"] = full_model_fit.df_resid + full_model_fit.df_model
    mlr_reg_anova.loc["Total"]["SS"] = (full_model_fit.ssr + full_model_fit.ess).round(4)
    mlr_reg_anova.loc["Total"]["MS"] = ""
    mlr_reg_anova.loc["Total"]["F-statistic"] = ""
    mlr_reg_anova.loc["Total"]["p-value"] = ""
    #st.write(mlr_reg_anova)

    # Coefficients
    mlr_reg_coef["coeff"] = full_model_fit.params
    mlr_reg_coef["std err"] = full_model_fit.bse
    mlr_reg_coef["t-statistic"] = full_model_fit.tvalues
    mlr_reg_coef["p-value"] = full_model_fit.pvalues
    mlr_reg_coef["lower 95%"] = full_model_fit.conf_int(alpha = 0.05)[0]
    mlr_reg_coef["upper 95%"] = full_model_fit.conf_int(alpha = 0.05)[1]
    #st.write(mlr_reg_coef)


    return  mlr_reg_inf, mlr_reg_stats, mlr_reg_anova, mlr_reg_coef,X_data, Y_data, Y_pred

#------------------------------------------------------------------------------------------
#FUNCTIONS FOR STOCK DATA ANALYSIS
def  get_stock_list(index_name,web_page,table_no, symbol_col, company_col, sector_col):
    if index_name=='CSI300':
        payload=pd.read_html(web_page,converters={'Index': str})
    elif index_name=='NIKKEI225':
        payload=pd.read_html(web_page,converters={'Code': str})
    elif index_name=='KOSPI':
        payload=pd.read_html(web_page,converters={'Code': str})
    else:
        payload=pd.read_html(web_page)    
    first_table = payload[table_no]
    df = first_table

    df.loc[:,'Index_name'] = index_name

    if index_name=='FTSE100':
        df.iloc[:,1] = df.iloc[:,1] + '.L'
    if index_name=='CSI300':
        df.loc[df['Stock exchange'] == 'Shanghai','Index']=df.loc[df['Stock exchange'] == 'Shanghai','Index']+'.SS'
        df.loc[df['Stock exchange'] == 'Shenzhen','Index']=df.loc[df['Stock exchange'] == 'Shenzhen','Index']+'.SZ'
    if index_name=='NIKKEI225':
        df.loc[:,'Code'] = df.loc[:,'Code'] + '.T'  
    if index_name=='KOSPI':
        df.loc[:,'Code'] = df.loc[:,'Code'] + '.KS'   
    #if index_name=='S&P_TSX60':
    #    df.loc[:,'Symbol'] = df.loc[:,'Symbol'] + '.TO'          
    symbols = df.iloc[:,symbol_col].values.tolist()
    companies = df.iloc[:,company_col].values.tolist()
    sectors = df.iloc[:,sector_col].values.tolist()
    stock_index=df.loc[:,'Index_name'].tolist()
    #symbols_all=symbols_all+symbols
    #company_all=company_all+companies
    #sector_all=sector_all+sectors
    #index_all=index_all+stock_index
    
    return symbols,companies,sectors,stock_index


#----------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
#FUNCTION FOR TEXT PROCESSING

def cv_text(user_text, word_stopwords, ngram_level,user_precision,number_remove):
    
    if ngram_level>1:
        if number_remove==True:
            user_text=''.join(c for c in user_text if not c.isdigit())

    cv = CountVectorizer(analyzer='word', stop_words=set(word_stopwords), ngram_range=(ngram_level, ngram_level))
  
    cv_fit=cv.fit_transform([user_text])
    cv_output= pd.DataFrame(cv_fit.toarray().sum(axis=0), index=cv.get_feature_names(),columns=["Word count"])
    cv_output["Rel. freq."]=100*cv_output["Word count"]/cv_output["Word count"].sum() 
    cv_output["Rel. freq."]=cv_output["Rel. freq."].round(user_precision)
    
    # sort the output:
    cv_output=cv_output.sort_values(by=["Word count"], ascending=False)

    if ngram_level==1:
        if number_remove==True:
        # remove numbers:
            words_stay =[x for x in cv.get_feature_names() if not any(c.isdigit() for c in x)]
            cv_output = cv_output[cv_output.index.isin(words_stay)]       
        
    

    #add column with word length
    cv_output["Word length"]=[len(i) for i in cv_output.index]

    return  cv_output


#---------------------------------------------------------------------------------------------
#FUNCTION FOR LEARNING HINTS

def learning_hints(name):

    #-------------------------------------------------------------------------------------------------

    # DATA EXPLORATION (Multivariate Data & Panel Data)
    #--------------------------------------------------

    # Data Exploration - Summary statistics
    if name == "de_summary_statistics":
        # All options
        learning_hint_options = [
        "What are the values for the central tendency for each variable?",
        "How are the variance and standard deviation related?",
        "How has the 10%-Q to be interpreted for each variable?",
        "How has the 25%-Q to be interpreted for each variable?",
        "How has the median to be interpreted for each variable?",
        "How has the 75%-Q to be interpreted for each variable?",
        "How has the 90%-Q to be interpreted for each variable?",
        "How has the skewness to be interpreted for each variable?",
        "How has the kurtosis to be interpreted for each variable?",
        "Are the minimum and maximum values for each variable within the theoretical ranges?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    #-------------------------------------------------------------------------------------------------

    # DATA EXPLORATION (Panel Data)
    #------------------------------

    # Data Exploration - ANOVA
    if name == "de_anova_boxplot":
        # All options
        learning_hint_options = [
        "What are the pecularities of the boxplots?",
        "Which entities have a narrow distribution?",
        "Which time periods have a narrow distribution?",
        "Which entities have the broadest distribution?",
        "Which time periods have the broadest distribution?",
        "Are there any outliers among entities or time periods?",
        "Are the variances equal among the groups?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    if name == "de_anova_count":
        # All options
        learning_hint_options = [
        "Is the data a balanced or unbalanced panel data set?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    if name == "de_anova_table":
        # All options
        learning_hint_options = [
        "Are there significant differences among the selected groups?",
        "How should the p-value be interpreted?",
        "Are the samples of the single groups drawn from populations with the same mean?",
        "How is the F-statistic calculated?",
        "How are the between and within SS determined?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    if name == "de_anova_residuals":
        # All options
        learning_hint_options = [
        "Are the residuals approximately normally distributed?",
        "Are the residuals left-skewed, right-skewed or symmetric?",
        "Are there any observations that might be outliers?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    #-------------------------------------------------------------------------------------------------

    # DATA VISUALIZATION (Multivariate and Panel Data)
    #-------------------------------------------------
    
    # Data Visualization - Scatterplot
    if name == "dv_scatterplot":
        # All options
        learning_hint_options = [
        "What kind of correlation is probably present?", 
        "Are the two variables strongly correlated?",
        "Is there a linear, non-linear or no relationship between the two variables?",
        "Are the two variables positively or negatively correlated?",
        "Is there a causal relationship between these two variables?",
        "Are there any outliers?",
        "How can outliers be identified in a scatterplot?",
        "How does the y variable behave if the x variable increases?",
        "Would a linear regression make sense for these two variables?",
        "Is there a general trend observable?",
        "Are clusters observable?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Data Visualization - Histogram
    if name == "dv_histogram":
        # All options
        learning_hint_options = [
        "Is the distribution left-skewed, right-skewed or symmetric?", 
        "Is the distribution unimodal, bimodal or multimodal?",
        "How can the histogram be manipulated?",
        "Which category or interval occurs most often?",
        "Can outliers be identified with this histogram?",
        "Is the mean lying in the interval with the highest count?",
        "Is it possible that the mean is stronlgy influenced by very high or low values?",
        "How can the variable be described with the histogram?",
        "How does the interval with the highest count change for differnet numbers of bins?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Data Visualization - Boxplot
    if name == "dv_boxplot":
        # All options
        learning_hint_options = [
        "Between which values lie at least 50% of the data?", 
        "Are there any outliers?",
        "What does the box of a boxplot represent?",
        "How can the distribution of the variable be interpreted with this boxplot?",
        "What are differences compared to a histogram?",
        "How large is the interquartile range?",
        "What is the interpretation of the first quartile Q1?",
        "What is the interpretation of the third quartile Q3?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
        
    # Data Visualization - Boxplot
    if name == "dv_qqplot":
        # All options
        learning_hint_options = [
        "Is the underlying variable normally distributed?", 
        "Is the distribution left-skewed, right-skewed or symmetric?",
        "How can the QQ-plot be interpreted (with the help of the histogram)?",
        "What is the QQ-plot used for?",
        "How can outliers be identified?",
        "Can anything be said about modality?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    #-------------------------------------------------------------------------------------------------

    # MODELLING (Multivariate Data and Panel Data)
    #---------------------------------------------

    # Modelling - Correlation
    if name == "mod_cor":
        # All options
        learning_hint_options = [
        "Which variables are strongly correlated?",
        "Can some variables be excluded due to high correlation?",
        "How are the explanatory variables correlated with the dependent variable?",
        "Is there a risk of multicollinearity?",
        "Are there causal relationships between variables with high correlation?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
   

    #-------------------------------------------------------------------------------------------------

    # MODELLING (Multivariate Data)
    #------------------------------

    # Multiple Linear Regression
    #---------------------------

    # Modelling - MLR - Regression statistics
    if name == "mod_md_MLR_regStat":
        # All options
        learning_hint_options = [
        "What is the difference between R² and adjusted R²?",
        "What is the aim of the least squares method?",
        "What are your arguments for choosing the specific covariance type?",
        "What are the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) used for?",
        "Is a higher log-likelihood better than a lower one?",
        "What does the residual standard error tell you?",
        "For which variables is the multiple correlation coefficient calculated?",
        "How would you interpret the performance of the model?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - MLR - Coefficients
    if name == "mod_md_MLR_coef":
        # All options
        learning_hint_options = [
        "What do the coefficients tell you about impacts on the dependent variable?", 
        "Is it possible to identify the variable with the strongest influence on the dependent variable based on the coefficients?",
        "From which matrix are the standard errors of the coefficients derived?",
        "How is the t-statistic calculated?",
        "What does the p-value for each coefficient tell you?",
        "Are all explanatory variables in the model significant?",
        "How can you determine whether a variable is significant or not?",
        "How can the coefficients be interpreted?",
        "What does the 95% confidence interval for each coefficient tell you?",
        "Are there changes in the signs of the borders of the 95% confidence interval for any explanatory variable?",
        "What would be the prediction of the model if the input for every explanatory variable is zero?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - MLR - ANOVA
    if name == "mod_md_MLR_ANOVA":
        # All options
        learning_hint_options = [
        "Can the ANOVA table be used to determine the R²?",
        "How should the p-value be interpreted?",
        "How is the F-statistic calculated?",
        "Is your model overall significant?",
        "Is your model better than a model that always predicts the mean of the dependent variable?",
        "Is the R² significantly different from zero?",
        "Would a smaller residual sum of squares be better?",
        "What is the null hypothesis of the overall F-test?",
        "What is the alternative hypothesis of the overall F-test?",
        "Are the results of the overall F-test trustworthy?",
        "In which case should the result of the overall F-test be treated carefully?",
        "Would a larger regression sum of squares be better?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - MLR - Heteroskedasticity test
    if name == "mod_md_MLR_hetTest":
        # All options
        learning_hint_options = [
        "What is the difference between the Breusch-Pagan test and White test?", 
        "Is heteroskedasticity present?",
        "What is the dependent variable in the underlying tests?",
        "Does the squared error of the model depend on the explanatory variables?",
        "What are your options in case of heteroskedasticity?",
        "What is the null hypothesis of a heteroskedasticity test?",
        "In which case should the result of the Breusch-Pagan test be treated carefully?",
        "Which test result should rather be used if the errors are not normally distributed?",
        "What is the alternative hypothesis of a heteroskedasticity test?",
        "Which distribution is used for the heteroskedasticity tests?",
        "What is the definition of the test statistic?",
        "On which test should you focus if non-linear forms of heteroskedasticity are present?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - MLR - Variable importance
    if name == "mod_md_MLR_varImp":
        # All options
        learning_hint_options = [
        "Which variable is the most important?",
        "Would it make sense to exclude a variable in the model?",
        "Are the least important variables significant?",
        "What does the value for variable importance tell you?",
        "Which variable importance shows a high variance?",
        "Can the variable importance be logically explained?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - MLR - Observed/Residuals vs Fitted
    if name == "mod_md_MLR_obsResVsFit":
        # All options
        learning_hint_options = [
        "Are there any observations that might be outliers?",
        "Can be concluded that the variances of the residuals/ errors are equal?",
        "Which predictions deviate the furthest from the corresponding observations?",
        "Can be concluded that a linear relationship is reasonable?",
        "Can heteroskedasticity be observed?",
        "Are the residuals/ errors approximately normally distributed?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - MLR - QQ-plot
    if name == "mod_md_MLR_qqplot":
        # All options
        learning_hint_options = [
        "Are the residuals normally distributed?", 
        "Which observations contribute to violating normality?",
        "Are the residuals left-skewed or right-skewed?",
        "Can any outliers be detected?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - MLR - Scale-Location
    if name == "mod_md_MLR_scaleLoc":
        # All options
        learning_hint_options = [
        "Are the residuals randomly spread?", 
        "Can homoskedasticity/ equal variance be concluded?",
        "What does the red line tell you about the residuals?",
        "Can a pattern of the residuals be identified?",
        "Which observations might be outlier?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - MLR - Residuals vs Leverage
    if name == "mod_md_MLR_resVsLev_cooksD":
        # All options
        learning_hint_options = [
        "Which observations might have substantial influence on your model?", 
        "Which observations have a large Cook's distance?",
        "What does a changing spread of the residuals along the leverage imply?",
        "Which observations have both a large leverage and residual error?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # continuous variables

    # Generalized Additive Models
    #----------------------------

    # Modelling - GAM - Regression statistics
    if name == "mod_md_GAM_regStat":
        # All options
        learning_hint_options = [
        "How is the Pseudo R² defined?",
        "What are the Akaike Information Criterion (AIC) and the second-order AIC (AICc) used for?",
        "Is a higher log-likelihood better than a lower one?",
        "What does the generalized cross-validation (GCV) score tell you?",
        "What does the scale score tell you?",
        "How would you interpret the performance of the model?",
        "What is a link function?",
        "What are effective degrees of freedom (Effective DF)?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - GAM - Feature significance
    if name == "mod_md_GAM_featSig":
        # All options
        learning_hint_options = [
        "What is the lambda value for each explanatory variable?",
        "What is the final number of splines used?",
        "How are the effective degrees of freedom distributed among the explanatory variables?",
        "Which splines are statistically significant?",
        "How can effective degrees of freedom (edof) be interpreted for each explanatory variable?",
        "How can the lambda value be interpreted for each explanatory variable?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - GAM - Variable importance
    if name == "mod_md_GAM_varImp":
        # All options
        learning_hint_options = [
        "Which variable is the most important?",
        "Would it make sense to exclude a variable in the model?",
        "What do the values for variable importance tell you?",
        "Which variable importance shows a high variance?",
        "Can the variable importance be logically explained?",
        "Is there a difference between the two methods used for determining variable importance?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - GAM - Partial Dependence Plots
    if name == "mod_md_GAM_partDep":
        # All options
        learning_hint_options = [
        "Is there a connection between the range of the partial dependence curve and the variable importance?",
        "How can the influence of each explanatory variable on the dependent variable be described?",
        "What can be said about parts of the curve with lower observation density?",
        "Is the partial dependence curve reliable along the whole gradient of the respective explanatory variable?",
        "May different influence directions be possible for certain ranges of the explanatory variables?",
        "Is there a rational explanation for the identified influence directions?",
        "Do certain partial dependence curves behave differently than expected?",
        "How would you assess the 95% confidence interval across the gradients of the explanatory variables?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

     # Modelling - GAM - Observed/Residuals vs Fitted
    if name == "mod_md_GAM_obsResVsFit":
        # All options
        learning_hint_options = [
        "Are there any observations that might be outliers?",
        "Can be concluded that the variances of the residuals/ errors are equal?",
        "Which predictions deviate the furthest from the corresponding observations?",
        "Is a trend observable for the residuals?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Random Forest
    #--------------

    # Modelling - RF - Regression statistics
    if name == "mod_md_RF_regStat":
        # All options
        learning_hint_options = [
        "What does the residual standard error tell you?",
        "What is the aim of the loss function least squares?",
        "How is the mean squared error determined?",
        "How would you interpret the out-of-bag (OOB) score?",
        "How would you interpret the performance of the model?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - RF - Variable importance
    if name == "mod_md_RF_varImp":
        # All options
        learning_hint_options = [
        "Which variable is the most important?",
        "Would it make sense to exclude a variable in the model?",
        "What do the values for variable importance tell you?",
        "Which variable importance shows a high variance?",
        "Can the variable importance be logically explained?",
        "Is there a difference between the two methods used for determining variable importance?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - RF - Partial Dependence Plots
    if name == "mod_md_RF_partDep":
        # All options
        learning_hint_options = [
        "Is there a connection between the range of the partial dependence curve and the variable importance?",
        "How can the influence of each explanatory variable on the dependent variable be described?",
        "What can be said about parts of the curve with lower observation density?",
        "Is the partial dependence curve reliable along the whole gradient of the respective explanatory variable?",
        "May different influence directions be possible for certain ranges of the explanatory variables?",
        "Is there a rational explanation for the identified influence directions?",
        "Do certain partial dependence curves behave differently than expected?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - RF - Observed/Residuals vs Fitted
    if name == "mod_md_RF_obsResVsFit":
        # All options
        learning_hint_options = [
        "Are there any observations that might be outliers?",
        "Can be concluded that the variances of the residuals/ errors are equal?",
        "Which predictions deviate the furthest from the corresponding observations?",
        "Is a trend observable for the residuals?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]


    # Boosted Regression Trees
    #-------------------------

    # Modelling - BRT - Regression statistics
    if name == "mod_md_BRT_regStat":
        # All options
        learning_hint_options = [
        "What does the residual standard error tell you?",
        "What is the aim of the loss function least squares?",
        "How is the mean squared error determined?",
        "How would you interpret the performance of the model?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - BRT - Variable importance
    if name == "mod_md_BRT_varImp":
        # All options
        learning_hint_options = [
        "Which variable is the most important?",
        "Would it make sense to exclude a variable in the model?",
        "What do the values for variable importance tell you?",
        "Which variable importance shows a high variance?",
        "Can the variable importance be logically explained?",
        "Is there a difference between the two methods used for determining variable importance?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - BRT - Partial Dependence Plots
    if name == "mod_md_BRT_partDep":
        # All options
        learning_hint_options = [
        "Is there a connection between the range of the partial dependence curve and the variable importance?",
        "How can the influence of each explanatory variable on the dependent variable be described?",
        "What can be said about parts of the curve with lower observation density?",
        "Is the partial dependence curve reliable along the whole gradient of the respective explanatory variable?",
        "May different influence directions be possible for certain ranges of the explanatory variables?",
        "Is there a rational explanation for the identified influence directions?",
        "Do certain partial dependence curves behave differently than expected?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - BRT - Observed/Residuals vs Fitted
    if name == "mod_md_BRT_obsResVsFit":
        # All options
        learning_hint_options = [
        "Are there any observations that might be outliers?",
        "Can be concluded that the variances of the residuals/ errors are equal?",
        "Which predictions deviate the furthest from the corresponding observations?",
        "Is a trend observable for the residuals?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Artificial Neural Networks
    #---------------------------

    # Modelling - ANN - Regression statistics
    if name == "mod_md_ANN_regStat":
        # All options
        learning_hint_options = [
        "What does the residual standard error tell you?",
        "What is the aim of the loss function squared loss?",
        "How is the mean squared error determined?",
        "How would you interpret the performance of the model?",
        "How can the best loss be interpreted?",
        "How often was the dataset seen by the model?"  
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - ANN - Variable importance
    if name == "mod_md_ANN_varImp":
        # All options
        learning_hint_options = [
        "Which variable is the most important?",
        "Would it make sense to exclude a variable in the model?",
        "What does the value for variable importance tell you?",
        "Which variable importance shows a high variance?",
        "Can the variable importance be logically explained?"  
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - ANN - Partial Dependence Plots
    if name == "mod_md_ANN_partDep":
        # All options
        learning_hint_options = [
        "Is there a connection between the range of the partial dependence curve and the variable importance?",
        "How can the influence of each explanatory variable on the dependent variable be described?",
        "What can be said about parts of the curve with lower observation density?",
        "Is the partial dependence curve reliable along the whole gradient of the respective explanatory variable?",
        "May different influence directions be possible for certain ranges of the explanatory variables?",
        "Is there a rational explanation for the identified influence directions?",
        "Do certain partial dependence curves behave differently than expected?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - ANN - Observed/Residuals vs Fitted
    if name == "mod_md_ANN_obsResVsFit":
        # All options
        learning_hint_options = [
        "Are there any observations that might be outliers?",
        "Can be concluded that the variances of the residuals/ errors are equal?",
        "Which predictions deviate the furthest from the corresponding observations?",
        "Is a trend observable for the residuals?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Model comparison
    #-----------------
    
    # Modelling - Model comparisons - Performance metrics
    if name == "mod_md_modCompPerf":
        # All options
        learning_hint_options = [
        "Which algorithm has the highest % VE?",
        "Which algorithm has the lowest MSE?", 
        "Which algorithm has the lowest RMSE?", 
        "Which algorithm has the lowest MAE?", 
        "Which algorithm has the highest EVRS?",  
        "Which algorithm has the lowest SSR?",
        "Which algorithm is the overall best algorithm?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - Model comparisons - Residuals distribution
    if name == "mod_md_modCompRes":
        # All options
        learning_hint_options = [
        "Which algorithm has the best minimum residual value?",
        "Which algorithm has the best 25%-Q value?", 
        "Which algorithm has the best median value?", 
        "Which algorithm has the best 75%-Q value?", 
        "Which algorithm has the best maximum value?",  
        "Which algorithm is the overall best algorithm according to the distribution of the main quantiles?"  
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # binary variables

    # Logistic Regression
    #--------------------

    # Modelling - LR - Regression statistics
    if name == "mod_md_LR_regStat":
        # All options
        learning_hint_options = [
        "What is the aim of the maximum likelihood method?",
        "What are your arguments for choosing the specific covariance type?",
        "What are the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) used for?",
        "Is a higher log-likelihood better than a lower one?",
        "Is your model better than the null model?",
        "How is the log-likelihood ratio calculated?",
        "Why is the log-likelihood ratio considered?",
        "How are the log-likelihood and deviance related?",
        "What is the definition of the pseudo R²?",
        "How would you interpret the performance of the model?",
        "What is the logit?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - LR - Coefficients
    if name == "mod_md_LR_coef":
        # All options
        learning_hint_options = [
        "What do the coefficients tell you about impacts on the dependent variable?", 
        "Is it possible to identify the variable with the strongest influence on the dependent variable based on the coefficients?",
        "From which matrix are the standard errors of the coefficients derived?",
        "How is the t-statistic calculated?",
        "What does the p-value for each coefficient tell you?",
        "Are all explanatory variables in the model significant?",
        "How can you determine whether a variable is significant or not?",
        "How can the coefficients be interpreted?",
        "What does the 95% confidence interval for each coefficient tell you?",
        "Are there changes in the signs of the borders of the 95% confidence interval for any explanatory variable?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - LR - Variable importance
    if name == "mod_md_LR_varImp":
        # All options
        learning_hint_options = [
        "Which variable is the most important?",
        "Would it make sense to exclude a variable in the model?",
        "Are the least important variables significant?",
        "What does the value for variable importance tell you?",
        "Which variable importance shows a high variance?",
        "Can the variable importance be logically explained?"  
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - LR - Threshold/AUC
    if name == "mod_md_LR_thresAUC":
        # All options
        learning_hint_options = [
        "What threshold was determined to seperate the two categories?",
        "How is the threshold for seperating the two categories determined?",
        "How would you interpret the AUC value?",
        "For which AUC value do we have a random model?",
        "Why is it better for the ROC curve to be close to the upper left corner?",
        "How is the ROC curve created?",
        "Are the two categories well seperated by the determined threshold?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - LR - Partial Probability Plots
    if name == "mod_md_LR_partProb":
        # All options
        learning_hint_options = [
        "How are partial probability plots created?",
        "Is there a connection between the range of the partial probability curve and the variable importance?",
        "How can the influence of each single explanatory variable on the dependent variable be described?",
        "Is there a rational explanation for the identified influence directions?",
        "Do certain partial probability curves behave differently than expected?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Generalized Additive Models
    #----------------------------

    # Modelling - GAM - Regression statistics
    if name == "mod_md_GAM_regStat_bin":
        # All options
        learning_hint_options = [
        "What does the AUC ROC tell you?",
        "How is the Pseudo R² defined?",
        "What are the Akaike Information Criterion (AIC) and the second-order AIC (AICc) used for?",
        "Is a higher log-likelihood better than a lower one?",
        "What does the un-biased risk estimator (UBRE) score tell you?",
        "What does the scale score tell you?",
        "How would you interpret the performance of the model?",
        "What is a link function?",
        "What are effective degrees of freedom (Effective DF)?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - GAM - Feature significance
    if name == "mod_md_GAM_featSig_bin":
        # All options
        learning_hint_options = [
        "What is the lambda value for each explanatory variable?",
        "What is the final number of splines used?",
        "How are the effective degrees of freedom distributed among the explanatory variables?",
        "Which splines are statistically significant?",
        "How can effective degrees of freedom (edof) be interpreted for each explanatory variable?",
        "How can the lambda value be interpreted for each explanatory variable?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - GAM - Variable importance
    if name == "mod_md_GAM_varImp_bin":
        # All options
        learning_hint_options = [
        "Which variable is the most important?",
        "Would it make sense to exclude a variable in the model?",
        "What do the values for variable importance tell you?",
        "Which variable importance shows a high variance?",
        "Can the variable importance be logically explained?",
        "Is there a difference between the two methods used for determining variable importance?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - GAM - Threshold/AUC
    if name == "mod_md_GAM_thresAUC":
        # All options
        learning_hint_options = [
        "What threshold was determined to seperate the two categories?",
        "How is the threshold for seperating the two categories determined?",
        "How would you interpret the AUC value?",
        "For which AUC value do we have a random model?",
        "Why is it better for the ROC curve to be close to the upper left corner?",
        "How is the ROC curve created?",
        "Are the two categories well seperated by the determined threshold?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - GAM - Partial Dependence Plots
    if name == "mod_md_GAM_partDep_bin":
        # All options
        learning_hint_options = [
        "Is there a connection between the range of the partial dependence curve and the variable importance?",
        "How can the influence of each explanatory variable on the dependent variable be described?",
        "What can be said about parts of the curve with lower observation density?",
        "Is the partial dependence curve reliable along the whole gradient of the respective explanatory variable?",
        "May different influence directions be possible for certain ranges of the explanatory variables?",
        "Is there a rational explanation for the identified influence directions?",
        "Do certain partial dependence curves behave differently than expected?",
        "How would you assess the 95% confidence interval across the gradients of the explanatory variables?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Random Forest
    #--------------

    # Modelling - RF - Regression statistics
    if name == "mod_md_RF_regStat_bin":
        # All options
        learning_hint_options = [
        "What does the AUC ROC tell you?",
        "What does the AUC PRC tell you?",
        "How would you interpret the out-of-bag (OOB) score?",
        "How would you interpret the performance of the model?"  
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - RF - Variable importance
    if name == "mod_md_RF_varImp_bin":
        # All options
        learning_hint_options = [
        "Which variable is the most important?",
        "Would it make sense to exclude a variable in the model?",
        "What do the values for variable importance tell you?",
        "Which variable importance shows a high variance?",
        "Can the variable importance be logically explained?",
        "Is there a difference between the two methods used for determining variable importance?"  
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - RF - Threshold/AUC
    if name == "mod_md_RF_thresAUC":
        # All options
        learning_hint_options = [
        "What threshold was determined to seperate the two categories?",
        "How is the threshold for seperating the two categories determined?",
        "How would you interpret the AUC value?",
        "For which AUC value do we have a random model?",
        "Why is it better for the ROC curve to be close to the upper left corner?",
        "How is the ROC curve created?",
        "Are the two categories well seperated by the determined threshold?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - RF - Partial Dependence Plots
    if name == "mod_md_RF_partDep_bin":
        # All options
        learning_hint_options = [
        "Is there a connection between the range of the partial dependence curve and the variable importance?",
        "How can the influence of each explanatory variable on the dependent variable be described?",
        "What can be said about parts of the curve with lower observation density?",
        "Is the partial dependence curve reliable along the whole gradient of the respective explanatory variable?",
        "May different influence directions be possible for certain ranges of the explanatory variables?",
        "Is there a rational explanation for the identified influence directions?",
        "Do certain partial dependence curves behave differently than expected?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Boosted Regression Trees
    #-------------------------

    # Modelling - BRT - Regression statistics
    if name == "mod_md_BRT_regStat_bin":
        # All options
        learning_hint_options = [
        "What does the AUC ROC tell you?",
        "What does the AUC PRC tell you?",
        "What is the aim of the loss function deviance?",
        "How are the log-loss and the deviance related?",
        "How would you interpret the performance of the model?"  
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - BRT - Variable importance
    if name == "mod_md_BRT_varImp_bin":
        # All options
        learning_hint_options = [
        "Which variable is the most important?",
        "Would it make sense to exclude a variable in the model?",
        "What do the values for variable importance tell you?",
        "Which variable importance shows a high variance?",
        "Can the variable importance be logically explained?",
        "Is there a difference between the two methods used for determining variable importance?"  
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - BRT - Threshold/AUC
    if name == "mod_md_BRT_thresAUC":
        # All options
        learning_hint_options = [
        "What threshold was determined to seperate the two categories?",
        "How is the threshold for seperating the two categories determined?",
        "How would you interpret the AUC value?",
        "For which AUC value do we have a random model?",
        "Why is it better for the ROC curve to be close to the upper left corner?",
        "How is the ROC curve created?",
        "Are the two categories well seperated by the determined threshold?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - BRT - Partial Dependence Plots
    if name == "mod_md_BRT_partDep_bin":
        # All options
        learning_hint_options = [
        "Is there a connection between the range of the partial dependence curve and the variable importance?",
        "How can the influence of each explanatory variable on the dependent variable be described?",
        "What can be said about parts of the curve with lower observation density?",
        "Is the partial dependence curve reliable along the whole gradient of the respective explanatory variable?",
        "May different influence directions be possible for certain ranges of the explanatory variables?",
        "Is there a rational explanation for the identified influence directions?",
        "Do certain partial dependence curves behave differently than expected?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Artificial Neural Networks
    #---------------------------

    # Modelling - ANN - Regression statistics
    if name == "mod_md_ANN_regStat_bin":
        # All options
        learning_hint_options = [
        "What does the AUC ROC tell you?",
        "What does the AUC PRC tell you?",
        "How would you interpret the average precision?",
        "What is the aim of the loss function log loss?",
        "How are the log-loss and the deviance related?",
        "How would you interpret the performance of the model?",
        "How can the best loss be interpreted?",
        "How often was the dataset seen by the model?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - ANN - Variable importance
    if name == "mod_md_ANN_varImp_bin":
        # All options
        learning_hint_options = [
        "Which variable is the most important?",
        "Would it make sense to exclude a variable in the model?",
        "What do the values for variable importance tell you?",
        "Which variable importance shows a high variance?",
        "Can the variable importance be logically explained?",
        "Is there a difference between the two methods used for determining variable importance?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - ANN - Threshold/AUC
    if name == "mod_md_ANN_thresAUC":
        # All options
        learning_hint_options = [
        "What threshold was determined to seperate the two categories?",
        "How is the threshold for seperating the two categories determined?",
        "How would you interpret the AUC value?",
        "For which AUC value do we have a random model?",
        "Why is it better for the ROC curve to be close to the upper left corner?",
        "How is the ROC curve created?",
        "Are the two categories well seperated by the determined threshold?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - ANN - Partial Dependence Plots
    if name == "mod_md_ANN_partDep_bin":
        # All options
        learning_hint_options = [
        "Is there a connection between the range of the partial dependence curve and the variable importance?",
        "How can the influence of each explanatory variable on the dependent variable be described?",
        "What can be said about parts of the curve with lower observation density?",
        "Is the partial dependence curve reliable along the whole gradient of the respective explanatory variable?",
        "May different influence directions be possible for certain ranges of the explanatory variables?",
        "Is there a rational explanation for the identified influence directions?",
        "Do certain partial dependence curves behave differently than expected?"  
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Model comparison
    #-----------------
    
    # Modelling - Model comparisons - Performance metrics
    if name == "mod_md_modCompThresInd":
        # All options
        learning_hint_options = [
        "Which algorithm has the highest AUC ROC?",
        "Which algorithm has the highest AP?", 
        "Which algorithm has the highest AUC PRC?", 
        "Which algorithm has the lowest LOG-LOSS?",
        "Which algorithm is the overall best algorithm according to threshold-independent metrics?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - Model comparisons - Residuals distribution
    if name == "mod_md_modCompThresDep":
        # All options
        learning_hint_options = [
        "Which algorithm has the highest TPR?",
        "Which algorithm has the lowest FNR?", 
        "Which algorithm has the highest TNR?", 
        "Which algorithm has the lowest FPR?",
        "Which algorithm has the highest TSS?",
        "Which algorithm has the highest PREC?",
        "Which algorithm has the highest F1 score?", 
        "Which algorithm has the highest KAPPA?", 
        "Which algorithm has the highest ACC?",  
        "Which algorithm has the highest BAL ACC?",
        "Which algorithm is the overall best algorithm according to threshold-dependent metrics?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # multi-class variables

    # Modelling - RF - Regression statistics
    if name == "mod_md_RF_regStat_mult":
        # All options
        learning_hint_options = [
        "What does the accuracy tell you?",
        "What does the balanced accuracy tell you?",
        "How would you interpret the out-of-bag (OOB) score?",
        "How would you interpret the performance of the model?"  
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - RF - Variable importance
    if name == "mod_md_RF_varImp_mult":
        # All options
        learning_hint_options = [
        "Which variable is the most important?",
        "Would it make sense to exclude a variable in the model?",
        "What do the values for variable importance tell you?",
        "Which variable importance shows a high variance?",
        "Can the variable importance be logically explained?",
        "Is there a difference between the two methods used for determining variable importance?"  
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - RF - Confusion matrix
    if name == "mod_md_RF_confu_mult":
        # All options
        learning_hint_options = [
        "Which category is predicted worst?",
        "Which category is predicted best?",
        "Is a pattern for wrong predictions observable?",
        "How would you assess the confusion matrix?"  
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - RF - Classification report
    if name == "mod_md_RF_classRep_mult":
        # All options
        learning_hint_options = [
        "Which category has the highest values for the given performance measures?",
        "Which category has the lowest values for the given performance measures?",
        "How is the weighted average determined?",
        "How is the macro average determined?",
        "What is the difference between the precision and the recall?",
        "How is the F1 score related to precision and recall?"  
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Artificial Neural Networks
    #---------------------------

    # Modelling - ANN - Regression statistics
    if name == "mod_md_ANN_regStat_mult":
        # All options
        learning_hint_options = [
        "What does the accuracy tell you?",
        "What does the balanced accuracy tell you?",
        "What is the aim of the loss function log loss?",
        "How would you interpret the performance of the model?",
        "How can the best loss be interpreted?",
        "How often was the dataset seen by the model?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - ANN - Confusion matrix
    if name == "mod_md_ANN_confu_mult":
        # All options
        learning_hint_options = [
        "Which category is predicted worst?",
        "Which category is predicted best?",
        "Is a pattern for wrong predictions observable?",
        "How would you assess the confusion matrix?"  
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - ANN - Classification report
    if name == "mod_md_ANN_classRep_mult":
        # All options
        learning_hint_options = [
        "Which category has the highest values for the given performance measures?",
        "Which category has the lowest values for the given performance measures?",
        "How is the weighted average determined?",
        "How is the macro average determined?",
        "What is the difference between the precision and the recall?",
        "How is the F1 score related to precision and recall?"  
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Model comparison
    #-----------------
    
    # Modelling - Model comparisons - Performance metrics
    if name == "mod_md_modComp_mult":
        # All options
        learning_hint_options = [
        "Which algorithm has the highest ACC?",
        "Which algorithm has the highest BAL ACC?", 
        "Which algorithm has the highest average PREC?", 
        "Which algorithm has the highest average RECALL?",
        "Which algorithm has the highest average F1 score?",
        "Which algorithm is the overall best algorithm according to the metrics?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    #-------------------------------------------------------------------------------------------------

    # VALIDATION (Multivariate Data)
    #-------------------------------

    # continuous

    # Modelling - Validation - Means
    if name == "mod_md_val_means":
        # All options
        learning_hint_options = [
        "Which algorithm has the highest mean % VE?",
        "Which algorithm has the lowest mean MSE?", 
        "Which algorithm has the lowest mean RMSE?", 
        "Which algorithm has the lowest mean MAE?", 
        "Which algorithm has the highest mean EVRS?",  
        "Which algorithm has the lowest mean SSR?",
        "Which algorithm is the overall best algorithm according to validation results?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Validation - SDs
    if name == "mod_md_val_sds":
        # All options
        learning_hint_options = [
        "Which algorithm has the lowest standard deviation for % VE?",
        "Which algorithm has the lowest standard deviation for MSE?", 
        "Which algorithm has the lowest standard deviation for RMSE?", 
        "Which algorithm has the lowest standard deviation for MAE?", 
        "Which algorithm has the lowest standard deviation for EVRS?",  
        "Which algorithm has the lowest standard deviation for SSR?",
        "Which algorithm seems overall most robust according to the standard deviations?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Validation - Boxplot of residuals
    if name == "mod_md_val_resBoxplot":
        # All options
        learning_hint_options = [
        "Which algorithm has the narrowest distribution of residuals?", 
        "Which algorithm has the broadest distribution of residuals?",
        "Which algorithm has the most outliers?",
        "For which algorithm is the box the smallest?",
        "For which algorithm is the box the largest?",
        "Which algorithm has the largest whiskers?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Validation - Boxplot of % VE
    if name == "mod_md_val_VEBoxplot":
        # All options
        learning_hint_options = [
        "Which algorithm has the narrowest distribution of % VE?", 
        "Which algorithm has the broadest distribution of % VE?",
        "Which algorithm has the most outliers?",
        "For which algorithm is the box the smallest?",
        "For which algorithm is the box the largest?",
        "Which algorithm has the highest median?",
        "Which algorithm has the largest whiskers?",
        "Which algorithm has the highest % VE?",
        "Which algorithm has the lowest % VE?"  
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Validation - Variable importance
    if name == "mod_md_val_varImp":
        # All options
        learning_hint_options = [
        "Which variable is the most important?",
        "Does the order of most important variables coincide with the results from the full model?",
        "Would it make sense to exclude a variable in the model?",
        "What does the value for variable importance tell you?",
        "Which variable importance shows a high variance?",
        "Can the variable importance be logically explained?"   
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Validation - Residuals
    if name == "mod_md_val_res":
        # All options
        learning_hint_options = [
        "Which algorithm has the best minimum residual value?",
        "Which algorithm has the best 25%-Q value?", 
        "Which algorithm has the best median value?", 
        "Which algorithm has the best 75%-Q value?", 
        "Which algorithm has the best maximum value?",  
        "Which algorithm is the overall best algorithm according to the distribution of the main quantiles?"   
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # binary

    # Modelling - Validation - Means threshold-independent
    if name == "mod_md_val_means_thresInd":
        # All options
        learning_hint_options = [
        "Which algorithm has the highest mean AUC ROC?",
        "Which algorithm has the highest mean AP?", 
        "Which algorithm has the highest mean AUC PRC?", 
        "Which algorithm has the lowest mean LOG-LOSS?",
        "Which algorithm is the overall best algorithm according to threshold-independent metrics of the validation?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Validation - SDs threshold-independent
    if name == "mod_md_val_sds_thresInd":
        # All options
        learning_hint_options = [
        "Which algorithm has the lowest standard deviation for AUC ROC?",
        "Which algorithm has the lowest standard deviation for AP?", 
        "Which algorithm has the lowest standard deviation for AUC PRC?", 
        "Which algorithm has the lowest standard deviation for LOG-LOSS?",
        "Which algorithm seems overall most robust according to the standard deviations?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Validation - Boxplot of AUC ROC
    if name == "mod_md_val_AUCBoxplot":
        # All options
        learning_hint_options = [
        "Which algorithm has the narrowest distribution of AUC ROC?", 
        "Which algorithm has the broadest distribution of AUC ROC?",
        "Which algorithm has the most outliers?",
        "For which algorithm is the box the smallest?",
        "For which algorithm is the box the largest?",
        "Which algorithm has the highest median?",
        "Which algorithm has the largest whiskers?",
        "Which algorithm has the highest AUC ROC?",
        "Which algorithm has the lowest AUC ROC?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Validation - Boxplot of TSS
    if name == "mod_md_val_TSSBoxplot":
        # All options
        learning_hint_options = [
        "Which algorithm has the narrowest distribution of TSS?", 
        "Which algorithm has the broadest distribution of TSS?",
        "Which algorithm has the most outliers?",
        "For which algorithm is the box the smallest?",
        "For which algorithm is the box the largest?",
        "Which algorithm has the highest median?",
        "Which algorithm has the largest whiskers?",
        "Which algorithm has the highest TSS?",
        "Which algorithm has the lowest TSS?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Validation - Variable importance
    if name == "mod_md_val_varImp_bin":
        # All options
        learning_hint_options = [
        "Which variable is the most important?",
        "Does the order of most important variables coincide with the results from the full model?",
        "Would it make sense to exclude a variable in the model?",
        "What does the value for variable importance tell you?",
        "Which variable importance shows a high variance?",
        "Can the variable importance be logically explained?"    
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Validation - Means threshold-dependent
    if name == "mod_md_val_means_thresDep":
        # All options
        learning_hint_options = [
        "Which algorithm has the highest mean TPR?",
        "Which algorithm has the lowest mean FNR?", 
        "Which algorithm has the highest mean TNR?", 
        "Which algorithm has the lowest mean FPR?",
        "Which algorithm has the highest mean TSS?",
        "Which algorithm has the highest mean PREC?",
        "Which algorithm has the highest mean F1 score?", 
        "Which algorithm has the highest mean KAPPA?", 
        "Which algorithm has the highest mean ACC?",  
        "Which algorithm has the highest mean BAL ACC?", 
        "Which algorithm is the overall best algorithm according to threshold-dependent metrics of the validation?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Validation - SDs threshold-independent
    if name == "mod_md_val_sds_thresDep":
        # All options
        learning_hint_options = [
        "Which algorithm has the lowest standard deviation for TPR?",
        "Which algorithm has the lowest standard deviation for FNR?", 
        "Which algorithm has the lowest standard deviation for TNR?", 
        "Which algorithm has the lowest standard deviation for FPR?",
        "Which algorithm has the lowest standard deviation for TSS?",
        "Which algorithm has the lowest standard deviation for PREC?",
        "Which algorithm has the lowest standard deviation for F1 score?", 
        "Which algorithm has the lowest standard deviation for KAPPA?", 
        "Which algorithm has the lowest standard deviation for ACC?",  
        "Which algorithm has the lowest standard deviation for BAL ACC?",
        "Which algorithm seems overall most robust according to the standard deviations?"  
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # multi-class

    # Modelling - Validation - Means 
    if name == "mod_md_val_means_mult":
        # All options
        learning_hint_options = [
        "Which algorithm has the highest mean ACC?",
        "Which algorithm has the highest mean BAL ACC?", 
        "Which algorithm is the overall best algorithm according to the metrics of the validation?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Validation - SDs 
    if name == "mod_md_val_sds_mult":
        # All options
        learning_hint_options = [
        "Which algorithm has the lowest standard deviation for ACC?",
        "Which algorithm has the lowest standard deviation for BAL ACC?", 
        "Which algorithm seems overall most robust according to the standard deviations?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - Validation - Boxplot of ACC
    if name == "mod_md_val_ACCBoxplot":
        # All options
        learning_hint_options = [
        "Which algorithm has the narrowest distribution of ACC?", 
        "Which algorithm has the broadest distribution of ACC?",
        "Which algorithm has the most outliers?",
        "For which algorithm is the box the smallest?",
        "For which algorithm is the box the largest?",
        "Which algorithm has the highest median?",
        "Which algorithm has the largest whiskers?",
        "Which algorithm has the highest ACC?",
        "Which algorithm has the lowest ACC?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Validation - Boxplot of  BAL ACC
    if name == "mod_md_val_BALACCBoxplot":
        # All options
        learning_hint_options = [
        "Which algorithm has the narrowest distribution of BAL ACC?", 
        "Which algorithm has the broadest distribution of BAL ACC?",
        "Which algorithm has the most outliers?",
        "For which algorithm is the box the smallest?",
        "For which algorithm is the box the largest?",
        "Which algorithm has the highest median?",
        "Which algorithm has the largest whiskers?",
        "Which algorithm has the highest BAL ACC?",
        "Which algorithm has the lowest BAL ACC?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - Validation - Variable importance
    if name == "mod_md_val_varImp_mult":
        # All options
        learning_hint_options = [
        "Which variable is the most important?",
        "Does the order of most important variables coincide with the results from the full model?",
        "Would it make sense to exclude a variable in the model?",
        "What does the value for variable importance tell you?",
        "Which variable importance shows a high variance?",
        "Can the variable importance be logically explained?"    
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    #-------------------------------------------------------------------------------------------------

    # HYPERPARAMETER-TUNING (Multivariate Data)
    #------------------------------------------

    # Random Forest
    #--------------

    # Modelling - Hyperparameter-tuning - Final hyperparameters
    if name == "mod_md_hypeTune_RF_finPara":
        # All options
        learning_hint_options = [
        "What does the specific maximum tree depth mean?",
        "What role does a single tree play in the model?",
        "Are interactions of the explanatory variables incorporated into the model?",
        "What does the specific sample rate mean?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Hyperparameter-tuning - Tuning details
    if name == "mod_md_hypeTune_RF_details":
        # All options
        learning_hint_options = [
        "How would you assess the test data score?", 
        "Is there a big difference between the mean cv score and the test data score?",
        "Did the score vary stronlgy among cross-validation runs?",
        "How does the test data score compare to the full model?",
        "How does the mean cv score compare to the full model?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Boosted Regression Trees
    #--------------------------

    # Modelling - Hyperparameter-tuning - Final hyperparameters
    if name == "mod_md_hypeTune_BRT_finPara":
        # All options
        learning_hint_options = [
        "What does the specific learning rate value mean?",
        "What does the specific maximum tree depth mean?",
        "Are interactions of the explanatory variables incorporated into the model?",
        "What does the specific sample rate mean?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Hyperparameter-tuning - Tuning details
    if name == "mod_md_hypeTune_BRT_details":
        # All options
        learning_hint_options = [
        "How would you assess the test data score?", 
        "Is there a big difference between the mean cv score and the test data score?",
        "Did the score vary stronlgy among cross-validation runs?",
        "How does the test data score compare to the full model?",
        "How does the mean cv score compare to the full model?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Artificial Neural Networks
    #---------------------------

    # Modelling - Hyperparameter-tuning - Final hyperparameters
    if name == "mod_md_hypeTune_ANN_finPara":
        # All options
        learning_hint_options = [
        "What does the specific maximum number of iterations mean?",
        "How is the data transformed in the hidden layers?",
        "What do the hidden layer sizes mean?",
        "What does the specific learning rate value mean?",
        "What does the specific L² regualarization value mean?",
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Hyperparameter-tuning - Tuning details
    if name == "mod_md_hypeTune_ANN_details":
        # All options
        learning_hint_options = [
        "How would you assess the test data score?", 
        "Is there a big difference between the mean cv score and the test data score?",
        "Did the score vary stronlgy among cross-validation runs?",
        "How does the test data score compare to the full model?",
        "How does the mean cv score compare to the full model?",
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    #-------------------------------------------------------------------------------------------------

    # MODELLING (Panel Data)
    #-----------------------

    # Modelling - Information
    if name == "mod_pd_information":
        # All options
        learning_hint_options = [
        "Is the input data balanced or unbalanced?",
        "What is the minimum number of observations for an entity?",
        "What is the minimum number of observation for a time period?",
        "How many observations are available on average for the entities?", 
        "How many observations are available on average for the time periods?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Regression
    if name == "mod_pd_regression":
        # All options
        learning_hint_options = [
        "How is the R² determined?",
        "How is the specific estimation method defined?",
        "What are your arguments for choosing the specific covariance type?",
        "Is a higher log-likelihood better than a lower one?",
        "Is there a difference between SST and SST (overall)?",
        "How is the number for DF model determined?",
        "What are the differences between R² (between), R² (within) and R² (overall)?",
        "Why do the R² and R² (within) coincide/ not coincide?",
        "Why do the R² and R² (overall) coincide/ not coincide?",
        "How is the SST determined?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
        
    # Modelling - Overall performance
    if name == "mod_pd_overallPerf":
        # All options
        learning_hint_options = [
        "How is the % VE determined?", 
        "How is the MSE determined?",
        "How is the RMSE determined?",
        "How is the MAE determined?",
        "How is the MaxErr determined?",
        "How is the EVRS determined?",
        "How is the SSR determined?",
        "Is there a difference between % VE and R²?",
        "How does the distribution of the residuals look like?",
        "How would you interpret the performance of the model?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Coefficients
    if name == "mod_pd_coef":
        # All options
        learning_hint_options = [
        "What do the coefficients tell you about impacts on the dependent variable?", 
        "Is it possible to identify the variable with the strongest influence on the dependent variable based on the coefficients?",
        "From which matrix are the standard errors of the coefficients derived?",
        "How is the t-statistic calculated?",
        "What does the p-value for each coefficient tell you?",
        "Are all explanatory variables in the model significant?",
        "How can you determine whether a variable is significant or not?",
        "How can the coefficients be interpreted?",
        "What does the 95% confidence interval for each coefficient tell you?",
        "Are there changes in the signs of the borders of the 95% confidence interval for any explanatory variable?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - Effects (Fixed and Random Models)
    if name == "mod_pd_effects":
        # All options
        learning_hint_options = [
        "What are the effects of each entity?", 
        "What are the effects of each time period?", 
        "How are the sinlge effects derived?",
        "How can a prediction be made based on the effects and coefficients?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - ANOVA (Pooled)
    if name == "mod_pd_anova":
        # All options
        learning_hint_options = [
        "Can the ANOVA table be used to determine the R²?",
        "How is the F-statistic calculated?",
        "Would a smaller residual sum of squares be better?",
        "Would a larger regression sum of squares be better?",
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Variance decomposition (Random Models)
    if name == "mod_pd_varDecRE":
        # All options
        learning_hint_options = [
        "How is the variance distributed?",
        "How is theta used in the random effects model?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - Statistical tests (Random Models)
    if name == "mod_pd_testRE":
        # All options
        learning_hint_options = [
        "What is the null hypothesis of the overall F-test?",
        "What is the alternative hypothesis of the overall F-test?",
        "Are the results of the overall F-test trustworthy?",
        "How should the p-values be interpreted?",
        "Is the R² significantly different from zero?",
        "Is your model overall significant?",
        "In which case should the result of the overall F-test be treated carefully?",
        "What is the difference between the non-robust and robust F-test?",
        "Do the non-robust and robust F-test lead to the same conclusion?", 
        "Do the p-values of the non-robust and robust F-test coincide?",
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - Statistical tests (Entity Fixed Models + homoskedastic)
    if name == "mod_pd_testEFE_homosk":
        # All options
        learning_hint_options = [
        "What is the null hypothesis of the overall F-test?",
        "What is the alternative hypothesis of the overall F-test?",
        "Are the results of the overall F-test trustworthy?",
        "How should the p-values be interpreted?",
        "Is the R² significantly different from zero?",
        "Is your model overall significant?",
        "In which case should the result of the overall F-test be treated carefully?",
        "What is the difference between the non-robust and robust F-test?",
        "Do the non-robust and robust F-test lead to the same conclusion?", 
        "Do the p-values of the non-robust and robust F-test coincide?",
        "Is the effects model significantly different from a pooled model?",
        "May the random effects model be more appropriate than the entity fixed effects model?",
        "What is the null hypothesis of the Hausman test?",
        "What is the null hypothesis of the F-test for poolability?"  
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Statistical tests (Entity Fixed Models + heteroskedastic)
    if name == "mod_pd_testEFE":
        # All options
        learning_hint_options = [
        "What is the null hypothesis of the overall F-test?",
        "What is the alternative hypothesis of the overall F-test?",
        "Are the results of the overall F-test trustworthy?",
        "How should the p-values be interpreted?",
        "Is the R² significantly different from zero?",
        "Is your model overall significant?",
        "In which case should the result of the overall F-test be treated carefully?",
        "What is the difference between the non-robust and robust F-test?",
        "Do the non-robust and robust F-test lead to the same conclusion?", 
        "Do the p-values of the non-robust and robust F-test coincide?",
        "Is the effects model significantly different from a pooled model?",
        "What is the null hypothesis of the F-test for poolability?"   
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Statistical tests (Pooled Models)
    if name == "mod_pd_test_pooled":
        # All options
        learning_hint_options = [
        "What is the null hypothesis of the overall F-test?",
        "What is the alternative hypothesis of the overall F-test?",
        "Are the results of the overall F-test trustworthy?",
        "How should the p-values be interpreted?",
        "Is the R² significantly different from zero?",
        "Is your model overall significant?",
        "Is your model better than a model that always predicts the mean of the dependent variable?",
        "In which case should the result of the overall F-test be treated carefully?",
        "What is the difference between the non-robust and robust F-test?",
        "Do the non-robust and robust F-test lead to the same conclusion?", 
        "Do the p-values of the non-robust and robust F-test coincide?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Statistical tests (Remaining Models)
    if name == "mod_pd_test":
        # All options
        learning_hint_options = [
        "What is the null hypothesis of the overall F-test?",
        "What is the alternative hypothesis of the overall F-test?",
        "Are the results of the overall F-test trustworthy?",
        "How should the p-values be interpreted?",
        "Is the R² significantly different from zero?",
        "Is your model overall significant?",
        "In which case should the result of the overall F-test be treated carefully?",
        "What is the difference between the non-robust and robust F-test?",
        "Do the non-robust and robust F-test lead to the same conclusion?", 
        "Do the p-values of the non-robust and robust F-test coincide?",
        "Is the effects model significantly different from a pooled model?",
        "What is the null hypothesis of the F-test for poolability?"  
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    #-------------------------------------------------------------------------------------------------
    
    # VALIDATION (Panel Data)
    #------------------------

    # Modelling - Validation - Metrics 
    if name == "mod_pd_val_metrics":
        # All options
        learning_hint_options = [
        "How much do the means of the performance metrics deviate from the full model results?",
        "How much did the values for the different metrics vary around the mean?",
        "Does a specific metric vary a lot?",
        "How would you assess the validation results compared to the full model outputs?" 
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Modelling - Validation - Boxplot of residuals
    if name == "mod_pd_val_resBoxplot":
        # All options
        learning_hint_options = [
        "What is the maximum error across validation runs?",
        "What is the minimum error across validation runs?",
        "What is the median error across validation runs?",
        "Is the box rather narrow?",
        "What are the residual values for the quartiles?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Validation - Boxplot of % VE
    if name == "mod_pd_val_VEBoxplot":
        # All options
        learning_hint_options = [
        "What is the maximum % VE across validation runs?",
        "What is the minimum % VE across validation runs?",
        "What is the median % VE across validation runs?",
        "Is the box rather narrow?",
        "What are the % VE values for the quartiles?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Modelling - Validation - Residuals
    if name == "mod_pd_val_res":
        # All options
        learning_hint_options = [
        "How much do the residual values deviate from the full model results?", 
        ]
        # Randomly select an option 
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    #-------------------------------------------------------------------------------------------------

    # DECOMPOSITION 
    #--------------

    # Decomposition - Correlation
    if name == "decomp_cor":
        # All options
        learning_hint_options = [
        "Which variables are strongly correlated?",
        "Is there a risk of multicollinearity?",
        "Are there causal relationships between variables with high correlation?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Principal Component Analysis (PCA)
    #-----------------------------------

    # Decomposition - PCA - Eigenvalues and explained variance
    if name == "decomp_pca_eigval":
        # All options
        learning_hint_options = [
        "How many components should be included?",
        "How many components have an eigenvalue above 1?",
        "How much variance is explained by the first components?",
        "How can the eigenvalues be interpreted?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Decomposition - PCA - Eigenvectors
    if name == "decomp_pca_eigvec":
        # All options
        learning_hint_options = [
        "What are the loadings for the first component?",
        "Can the first component be labelled based on the loadings?",
        "Which variables have the highest loading for the first components?",
        "What do the values of the eigenvectors represent?",
        "How can the values be interpreted?",
        "How are the eigenvectors used to derive the transformed data?",
        "Are structures identifiable?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Factor Analysis (FA)
    #---------------------

    # Decomposition - FA - Adequacy tests
    if name == "decomp_fa_adeqtests":
        # All options
        learning_hint_options = [
        "What is the null hypothesis of Bartlett's Sphericity test?",
        "What does the KMO value tell you?",
        "Is a Factor Analysis appropriate for the data?",
        "What is the minimum/ maximum value for KMO?",
        "What does a low p-value imply?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Decomposition - FA - Eigenvalues
    if name == "decomp_fa_eigval":
        # All options
        learning_hint_options = [
        "How many components should be included?",
        "How many components have an eigenvalue above 1?",
        "How much variance is explained by the first components?",
        "How can the eigenvalues be interpreted?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Decomposition - FA - Explained variance
    if name == "decomp_fa_explvar":
        # All options
        learning_hint_options = [
        "How are the SS loadings calculated?",
        "What do the SS loadings represent?",
        "How much variance is explained by the first factors?",
        "How many factors should be kept?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Decomposition - FA - Communalities and uniquenesses
    if name == "decomp_fa_comuniq":
        # All options
        learning_hint_options = [
        "How are communalities determined?",
        "How are the uniquenesses determined?",
        "What does a high communality imply?",
        "What does a high uniqueness imply?",
        "Which variables should be kept in the factor analysis?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Decomposition - FA - Loadings
    if name == "decomp_fa_loadings":
        # All options
        learning_hint_options = [
        "What are the loadings for the first factor?",
        "Can the first factor be labelled based on the loadings?",
        "Which variables have the highest loading for the first factor?",
        "What do the loadings represent?",
        "How can the loadings be interpreted?",
        "How are the loadings related to the variance of the variables?",
        "Are structures identifiable?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    #-------------------------------------------------------------------------------------------------

    #-----------------------------------------------------
    # Time series
    #-----------------------------------------------------
    # Time Series Pattern
    if name == "ts_time_series_pattern":
        # All options
        learning_hint_options = [
        "Is the time-series stationary?", 
        "What is autocorrelation?",
        "What is partial autocorrelation?",
        "What does the ACF plot tells you?",
        "What does the PACF plot tells you?",
        "What is Augmented Dickey Fuller test?",
        "Interpret the p-value of the Augmented Dickey Fuller test"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Time series n-order differences    
    if name == "ts_n_order_differences":
        # All options
        learning_hint_options = [
        "What does it mean-the 2.-order difference?",
        "For which differences order the time series gets stationary?",
        "Is the time-series stationary?", 
        "What is autocorrelation?",
        "What is partial autocorrelation?",
        "What does the ACF plot tells you?",
        "What does the PACF plot tells you?",
        "What is Augmented Dickey Fuller test?",
        "Interpret the p-value of the Augmented Dickey Fuller test"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Time series detrending    
    if name == "ts_detrending_hints":
        # All options
        learning_hint_options = [
        "Is the detrended time-series stationary?", 
        "Is there a trend in the data?",
        "Interpret the p-value of the Augmented Dickey Fuller test"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Time series seasonal adjustment  
    if name == "ts_seasonal_hints":
        # All options
        learning_hint_options = [
        "Is seasonally adjusted time-series stationary?", 
        "Is there a seasonal component in the data?",
        "Interpret the p-value of the Augmented Dickey Fuller test"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Time series detrending & seasonal adjustment  
    if name == "ts_detrend_seasonal_hints":
        # All options
        learning_hint_options = [
        "Is detrended and seasonally adjusted time-series stationary?", 
        "Interpret the p-value of the Augmented Dickey Fuller test"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]
    
    # Time series models  
    if name == "ts_models_hints":
        # All options
        learning_hint_options = [
        "What is the difference between AR and MA models?", 
        "What is ARMA and how to find the model paramaters?",
        "What is the difference between seasonal and non-seasonal ARMA models?",
        "How can ACF and PACF help you to get the right model parameters?"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Time series results  
    if name == "ts_model_results_hints":
        # All options
        learning_hint_options = [
        "How good is your model - what statistics you should look at?", 
        "Are the residuals normally distributed?",
        "Is there a heteroscedasticity in your data?",
        "What model coefficients are significance at the level of 5%?",
        "What are AIC, BIC and HQIC and why are these important?",
        "Interpret the one-step ahead predictions"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    #-----------------------------------------------------
    # Uni and bi-variate analyses
    #-----------------------------------------------------

    # Contingency analysis  
    if name == "contingency_hints":
        # All options
        learning_hint_options = [
        "Interpret the corrected Pearson contingency coefficient?", 
        "What is the difference between the Pearson contingency coefficient and the corrected one?",
        "What are the expected frequencies?",
        "Explain the term 'marginal frequencies'?",
        "Is there an association between the variables that you analysed- how do you know?",
        "Check the bar plots of marginal frequencies"
        ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]

    # Correlation analysis  
    if name == "correlation_hints":
        # All options
        learning_hint_options = [
        "What variables manifest strongest correlation?", 
        "What is empirical covariance?",
        "What is the difference between the Pearson and the Spearman correlation?",
        "In case there are large outliers in the dataset, what correlation method is suitable?"
         ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]   

    if name == "reg_hints":
        # All options
        learning_hint_options = [
        "Is there any regression model that fits well to your data?", 
        "Are the residuals normally distributed?",
        "Interpret goodness of the fit measures.",
        "What does $R^2$ measures?"
         ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]       
    
    if name == "fit_hints":
        # All options
        learning_hint_options = [
        "Interpret the p-value in the goodness-of-fit summary table", 
        "What does the comparision of relative frequencies suggest you?",
        "Is there any theoretical distribution that fits well to your data?",
        "Try defining new classes for your empirical distribution"
         ]
        # Randomly select an option
        random_hint = randint(0, len(learning_hint_options)-1)
        learning_hint = learning_hint_options[random_hint]   

    return learning_hint    