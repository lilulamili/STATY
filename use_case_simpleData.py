#----------------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly as dd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager
import plotly.graph_objects as go
import functions as fc
import altair as alt
import scipy
import math
import os
from streamlit import caching
import SessionState
import sys
import platform

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, max_error
import statsmodels.api as sm


# Modelling specifications
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import f
from tqdm import tqdm



#----------------------------------------------------------------------------------------------

def app():

    # Clear cache
    caching.clear_cache()

    # Hide traceback in error messages (comment out for de-bugging)
    sys.tracebacklimit = 0

    # Show altair tooltip when full screen
    st.markdown('<style>#vg-tooltip-element{z-index: 1000051}</style>',unsafe_allow_html=True)

   
    #++++++++++++++++++++++++++++++++++++++++++++
    # DATA IMPORT

    # File upload section
    df_dec = st.sidebar.radio("Get data", ["Use example dataset", "Upload data"])

    if df_dec == "Upload data":
        #st.subheader("Upload your data")
        uploaded_data = st.sidebar.file_uploader("", type=["csv", "txt"])
        if uploaded_data is not None:
            df = pd.read_csv(uploaded_data, sep = ";|,|\t",engine='python')
            st.sidebar.success('Loading data... done!')
        elif uploaded_data is None:
           df = pd.read_csv("/default data/social.csv", sep = ";|,|\t",engine='python')
    else:
        df = pd.read_csv("/default data/social.csv", sep = ";|,|\t",engine='python') 
    st.sidebar.markdown("")
     
    #Basic data info
    n_rows = df.shape[0]
    n_cols = df.shape[1]  

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

    # Check if wide mode
    if sett_wide_mode:
        fc.wide_mode_func()

    # Check theme
    if sett_theme == "Dark":
        fc.theme_func_dark()
    if sett_theme == "Light":
        fc.theme_func_light()

    #++++++++++++++++++++++++++++++++++++++++++++
    # RESET INPUT

    reset_clicked = st.sidebar.button("Reset all your input")
    session_state = SessionState.get(id = 0)
    if reset_clicked:
        session_state.id = session_state.id + 1
    st.sidebar.markdown("")
    
    #------------------------------------------------------------------------------------------

    #++++++++++++++++++++++++++++++++++++++++++++
    # DATA EXPLORATION & VISUALIZATION

    data_title_container = st.beta_container()
    with data_title_container:
        st.header("**Uni- and bivariate data**")
        st.markdown("Let STATY do the data cleaning, variable transformations, visualizations and deliver you the stats you need. Specify your data processing preferences and start exploring your data stories right below... ")

    # Check if enough data is available
    if n_rows > 0 and n_cols > 0:
        st.empty()
    else:
        st.error("ERROR: Not enough data!")
        return

    data_exploration_container = st.beta_container()
    with data_exploration_container:
        st.header("**Data exploration**")
        
        #------------------------------------------------------------------------------------------

        #++++++++++++++++++++++
        # DATA SUMMARY

        # Main panel for data summary (pre)
        #----------------------------------

        dev_expander_raw = st.beta_expander("Explore raw data", expanded = False)
        with dev_expander_raw:

            # Show raw data & data info
            df_summary = fc.data_summary(df) 
            if st.checkbox("Show raw data", value = False, key = session_state.id):      
            # st.dataframe(df.style.apply(lambda x: ["background-color: #ffe5e5" if (not pd.isna(df_summary_mq_full.loc["1%-Q"][i]) and df_summary_vt_cat[i] == "numeric" and (v <= df_summary_mq_full.loc["1%-Q"][i] or v >= df_summary_mq_full.loc["99%-Q"][i]) or pd.isna(v)) else "" for i, v in enumerate(x)], axis = 1))
                st.write(df)
                st.write("Data shape: ", n_rows,  " rows and ", n_cols, " columns")
                #st.info("** Note that NAs and numerical values below/ above the 1%/ 99% quantile are highlighted.") 
            if df[df.duplicated()].shape[0] > 0 or df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0] > 0:
                check_nasAnddupl=st.checkbox("Show duplicates and NAs info", value = False, key = session_state.id) 
                if check_nasAnddupl:      
                    if df[df.duplicated()].shape[0] > 0:
                        st.write("Number of duplicates: ", df[df.duplicated()].shape[0])
                        st.write("Duplicate row index: ", ', '.join(map(str,list(df.index[df.duplicated()]))))
                    if df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0] > 0:
                        st.write("Number of rows with NAs: ", df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0])
                        st.write("Rows with NAs: ", ', '.join(map(str,list(pd.unique(np.where(df.isnull())[0])))))
                
            # Show variable info 
            if st.checkbox('Show variable info', value = False, key = session_state.id): 
                #st.write(df_summary["Variable types"])
                               
                st.write(df_summary["Variable types"])
            # Show summary statistics (raw data)
            if st.checkbox('Show summary statistics (raw data)', value = False, key = session_state.id): 
                #st.write(df_summary["ALL"])
                df_datasumstat=df_summary["ALL"]
                #dfStyler = df_datasumstat.style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector = 'th', props=[('text-align', 'left')])]) 
                
                st.write(df_datasumstat)
                if fc.get_mode(df).loc["n_unique"].any():
                    st.caption("** Mode is not unique.")
                if sett_hints:
                    st.info(str(fc.learning_hints("de_summary_statistics")))

        #++++++++++++++++++++++
        # DATA PROCESSING

        # Settings for data processing
        #-------------------------------------
        
        #st.write("")
        #st.subheader("Data processing")

        dev_expander_dm_sb = st.beta_expander("Specify data processing preferences", expanded = False)
        with dev_expander_dm_sb:
            
            n_rows_wNAs = df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0]
            if n_rows_wNAs > 0:
                a1, a2, a3 = st.beta_columns(3)
            else: a1, a2 = st.beta_columns(2)
            
            sb_DM_dImp_num = None 
            sb_DM_dImp_other = None
            if n_rows_wNAs > 0:
                with a1:
                    #--------------------------------------------------------------------------------------
                    # DATA CLEANING

                    st.markdown("**Data cleaning**")

                    # Delete duplicates if any exist
                    if df[df.duplicated()].shape[0] > 0:
                        sb_DM_delDup = st.selectbox("Delete duplicate rows ", ["No", "Yes"], key = session_state.id)
                        if sb_DM_delDup == "Yes":
                            n_rows_dup = df[df.duplicated()].shape[0]
                            df = df.drop_duplicates()
                    elif df[df.duplicated()].shape[0] == 0:   
                        sb_DM_delDup = "No"    
                        
                    # Delete rows with NA if any exist
                    n_rows_wNAs = df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0]
                    if n_rows_wNAs > 0:
                        sb_DM_delRows_wNA = st.selectbox("Delete rows with NAs ", ["No", "Yes"], key = session_state.id)
                        if sb_DM_delRows_wNA == "Yes": 
                            df = df.dropna()
                    elif n_rows_wNAs == 0: 
                        sb_DM_delRows_wNA = "No"   

                    # Delete rows
                    sb_DM_delRows = st.multiselect("Select rows to delete ", df.index, key = session_state.id)
                    df = df.loc[~df.index.isin(sb_DM_delRows)]

                    # Delete columns
                    sb_DM_delCols = st.multiselect("Select columns to delete ", df.columns, key = session_state.id)
                    df = df.loc[:,~df.columns.isin(sb_DM_delCols)]

                    # Filter data
                    st.markdown("**Data filtering**")
                    filter_var = st.selectbox('Filter your data by a variable...', list('-')+ list(df.columns), key = session_state.id)
                    if filter_var !='-':
                        filter_vals=st.selectbox('Filter your data by a value...', (df[filter_var]).unique(), key = session_state.id)
                        df =df[df[filter_var]==filter_vals]
            
                with a2:
                    #--------------------------------------------------------------------------------------
                    # DATA IMPUTATION

                    # Select data imputation method (only if rows with NA not deleted)
                    if sb_DM_delRows_wNA == "No" and n_rows_wNAs > 0:
                        st.markdown("**Data imputation**")
                        sb_DM_dImp_choice = st.selectbox("Replace entries with NA ", ["No", "Yes"], key = session_state.id)
                        if sb_DM_dImp_choice == "Yes":
                            # Numeric variables
                            sb_DM_dImp_num = st.selectbox("Imputation method for numeric variables ", ["Mean", "Median", "Random value"], key = session_state.id)
                            # Other variables
                            sb_DM_dImp_other = st.selectbox("Imputation method for other variables ", ["Mode", "Random value"], key = session_state.id)
                            df = fc.data_impute(df, sb_DM_dImp_num, sb_DM_dImp_other)
                    else: 
                        st.markdown("**Data imputation**")
                        st.write("")
                        st.info("No NAs in data set!")
                
                with a3:
                    #--------------------------------------------------------------------------------------
                    # DATA TRANSFORMATION

                    st.markdown("**Data transformation**")
                    # Select columns for different transformation types
                    transform_options = df.select_dtypes([np.number]).columns
                    numCat_options = df.columns
                    sb_DM_dTrans_log = st.multiselect("Select columns to transform with log ", transform_options, key = session_state.id)
                    if sb_DM_dTrans_log is not None: 
                        df = fc.var_transform_log(df, sb_DM_dTrans_log)
                    sb_DM_dTrans_sqrt = st.multiselect("Select columns to transform with sqrt ", transform_options, key = session_state.id)
                    if sb_DM_dTrans_sqrt is not None: 
                        df = fc.var_transform_sqrt(df, sb_DM_dTrans_sqrt)
                    sb_DM_dTrans_square = st.multiselect("Select columns for squaring ", transform_options, key = session_state.id)
                    if sb_DM_dTrans_square is not None: 
                        df = fc.var_transform_square(df, sb_DM_dTrans_square)
                    sb_DM_dTrans_stand = st.multiselect("Select columns for standardization ", transform_options, key = session_state.id)
                    if sb_DM_dTrans_stand is not None: 
                        df = fc.var_transform_stand(df, sb_DM_dTrans_stand)
                    sb_DM_dTrans_norm = st.multiselect("Select columns for normalization ", transform_options, key = session_state.id)
                    if sb_DM_dTrans_norm is not None: 
                        df = fc.var_transform_norm(df, sb_DM_dTrans_norm)
                    if df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0] == 0:
                        sb_DM_dTrans_numCat = st.multiselect("Select columns for numeric categorization ", numCat_options, key = session_state.id)
                        if sb_DM_dTrans_numCat is not None: 
                            df = fc.var_transform_numCat(df, sb_DM_dTrans_numCat)
                    else:
                        sb_DM_dTrans_numCat = None
            else:
                with a1:
                    #--------------------------------------------------------------------------------------
                    # DATA CLEANING

                    st.markdown("**Data cleaning**")

                    # Delete duplicates if any exist
                    if df[df.duplicated()].shape[0] > 0:
                        sb_DM_delDup = st.selectbox("Delete duplicate rows ", ["No", "Yes"], key = session_state.id)
                        if sb_DM_delDup == "Yes":
                            n_rows_dup = df[df.duplicated()].shape[0]
                            df = df.drop_duplicates()
                    elif df[df.duplicated()].shape[0] == 0:   
                        sb_DM_delDup = "No"    
                        
                    # Delete rows with NA if any exist
                    n_rows_wNAs = df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0]
                    if n_rows_wNAs > 0:
                        sb_DM_delRows_wNA = st.selectbox("Delete rows with NAs ", ["No", "Yes"], key = session_state.id)
                        if sb_DM_delRows_wNA == "Yes": 
                            df = df.dropna()
                    elif n_rows_wNAs == 0: 
                        sb_DM_delRows_wNA = "No"   

                    # Delete rows
                    sb_DM_delRows = st.multiselect("Select rows to delete ", df.index, key = session_state.id)
                    df = df.loc[~df.index.isin(sb_DM_delRows)]

                    # Delete columns
                    sb_DM_delCols = st.multiselect("Select columns to delete ", df.columns, key = session_state.id)
                    df = df.loc[:,~df.columns.isin(sb_DM_delCols)]

                    # Filter data
                    st.markdown("**Data filtering**")
                    filter_var=st.selectbox('Filter your data by a variable...',  list('-')+ list(df.columns), key = session_state.id)
                    if filter_var !='-':
                        filter_vals=st.selectbox('Filter your data by a value...', (df[filter_var]).unique(), key = session_state.id)
                        df =df[df[filter_var]==filter_vals]
                        
                with a2:
                    #--------------------------------------------------------------------------------------
                    # DATA TRANSFORMATION

                    st.markdown("**Data transformation**")
                    # Select columns for different transformation types
                    transform_options = df.select_dtypes([np.number]).columns
                    numCat_options = df.columns
                    sb_DM_dTrans_log = st.multiselect("Select columns to transform with log ", transform_options, key = session_state.id)
                    if sb_DM_dTrans_log is not None: 
                        df = fc.var_transform_log(df, sb_DM_dTrans_log)
                    sb_DM_dTrans_sqrt = st.multiselect("Select columns to transform with sqrt ", transform_options, key = session_state.id)
                    if sb_DM_dTrans_sqrt is not None: 
                        df = fc.var_transform_sqrt(df, sb_DM_dTrans_sqrt)
                    sb_DM_dTrans_square = st.multiselect("Select columns for squaring ", transform_options, key = session_state.id)
                    if sb_DM_dTrans_square is not None: 
                        df = fc.var_transform_square(df, sb_DM_dTrans_square)
                    sb_DM_dTrans_stand = st.multiselect("Select columns for standardization ", transform_options, key = session_state.id)
                    if sb_DM_dTrans_stand is not None: 
                        df = fc.var_transform_stand(df, sb_DM_dTrans_stand)
                    sb_DM_dTrans_norm = st.multiselect("Select columns for normalization ", transform_options, key = session_state.id)
                    if sb_DM_dTrans_norm is not None: 
                        df = fc.var_transform_norm(df, sb_DM_dTrans_norm)
                    if df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0] == 0:
                        sb_DM_dTrans_numCat = st.multiselect("Select columns for numeric categorization ", numCat_options, key = session_state.id)
                        if sb_DM_dTrans_numCat is not None: 
                            df = fc.var_transform_numCat(df, sb_DM_dTrans_numCat)
                    else:
                        sb_DM_dTrans_numCat = None

                #--------------------------------------------------------------------------------------
                # PROCESSING SUMMARY
                
                if st.checkbox('Show a summary of my data processing preferences ', value = False, key = session_state.id): 
                    st.markdown("Summary of data changes:")

                    #--------------------------------------------------------------------------------------
                    # DATA CLEANING

                    # Duplicates
                    if sb_DM_delDup == "Yes":
                        if n_rows_dup > 1:
                            st.write("-", n_rows_dup, " duplicate rows were deleted!")
                        elif n_rows_dup == 1:
                            st.write("-", n_rows_dup, "duplicate row was deleted!")
                    else:
                        st.write("- No duplicate row was deleted!")
                    # NAs
                    if sb_DM_delRows_wNA == "Yes":
                        if n_rows_wNAs > 1:
                            st.write("-", n_rows_wNAs, "rows with NAs were deleted!")
                        elif n_rows_wNAs == 1:
                            st.write("-", n_rows - n_rows_wNAs, "row with NAs was deleted!")
                    else:
                        st.write("- No row with NAs was deleted!")
                    # Rows
                    if len(sb_DM_delRows) > 1:
                        st.write("-", len(sb_DM_delRows), " rows were manually deleted:", ', '.join(map(str,sb_DM_delRows)))
                    elif len(sb_DM_delRows) == 1:
                        st.write("-",len(sb_DM_delRows), " row was manually deleted:", str(sb_DM_delRows[0]))
                    elif len(sb_DM_delRows) == 0:
                        st.write("- No row was manually deleted!")
                    # Columns
                    if len(sb_DM_delCols) > 1:
                        st.write("-", len(sb_DM_delCols), " columns were manually deleted:", ', '.join(sb_DM_delCols))
                    elif len(sb_DM_delCols) == 1:
                        st.write("-",len(sb_DM_delCols), " column was manually deleted:", str(sb_DM_delCols[0]))
                    elif len(sb_DM_delCols) == 0:
                        st.write("- No column was manually deleted!")
                    # Filter
                    if filter_var != "-":
                        st.write("-", " Data filtered by:", str(filter_var) , " > " , str(filter_vals))
                        
                    #--------------------------------------------------------------------------------------
                    # DATA IMPUTATION

                    if sb_DM_delRows_wNA == "No" and n_rows_wNAs > 0:
                        st.write("- Data imputation method for numeric variables:", sb_DM_dImp_num)
                        st.write("- Data imputation method for other variable types:", sb_DM_dImp_other)

                    #--------------------------------------------------------------------------------------
                    # DATA TRANSFORMATION

                    # log
                    if len(sb_DM_dTrans_log) > 1:
                        st.write("-", len(sb_DM_dTrans_log), " columns were log-transformed:", ', '.join(sb_DM_dTrans_log))
                    elif len(sb_DM_dTrans_log) == 1:
                        st.write("-",len(sb_DM_dTrans_log), " column was log-transformed:", sb_DM_dTrans_log[0])
                    elif len(sb_DM_dTrans_log) == 0:
                        st.write("- No column was log-transformed!")
                    # sqrt
                    if len(sb_DM_dTrans_sqrt) > 1:
                        st.write("-", len(sb_DM_dTrans_sqrt), " columns were sqrt-transformed:", ', '.join(sb_DM_dTrans_sqrt))
                    elif len(sb_DM_dTrans_sqrt) == 1:
                        st.write("-",len(sb_DM_dTrans_sqrt), " column was sqrt-transformed:", sb_DM_dTrans_sqrt[0])
                    elif len(sb_DM_dTrans_sqrt) == 0:
                        st.write("- No column was sqrt-transformed!")
                    # square
                    if len(sb_DM_dTrans_square) > 1:
                        st.write("-", len(sb_DM_dTrans_square), " columns were squared:", ', '.join(sb_DM_dTrans_square))
                    elif len(sb_DM_dTrans_square) == 1:
                        st.write("-",len(sb_DM_dTrans_square), " column was squared:", sb_DM_dTrans_square[0])
                    elif len(sb_DM_dTrans_square) == 0:
                        st.write("- No column was squared!")
                    # standardize
                    if len(sb_DM_dTrans_stand) > 1:
                        st.write("-", len(sb_DM_dTrans_stand), " columns were standardized:", ', '.join(sb_DM_dTrans_stand))
                    elif len(sb_DM_dTrans_stand) == 1:
                        st.write("-",len(sb_DM_dTrans_stand), " column was standardized:", sb_DM_dTrans_stand[0])
                    elif len(sb_DM_dTrans_stand) == 0:
                        st.write("- No column was standardized!")
                    # normalize
                    if len(sb_DM_dTrans_norm) > 1:
                        st.write("-", len(sb_DM_dTrans_norm), " columns were normalized:", ', '.join(sb_DM_dTrans_norm))
                    elif len(sb_DM_dTrans_norm) == 1:
                        st.write("-",len(sb_DM_dTrans_norm), " column was normalized:", sb_DM_dTrans_norm[0])
                    elif len(sb_DM_dTrans_norm) == 0:
                        st.write("- No column was normalized!")
                    # numeric category
                    if df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0] == 0:
                        if len(sb_DM_dTrans_numCat) > 1:
                            st.write("-", len(sb_DM_dTrans_numCat), " columns were transformed to numeric categories:", ', '.join(sb_DM_dTrans_numCat))
                        elif len(sb_DM_dTrans_numCat) == 1:
                            st.write("-",len(sb_DM_dTrans_numCat), " column was transformed to numeric categories:", sb_DM_dTrans_numCat[0])
                        elif len(sb_DM_dTrans_numCat) == 0:
                            st.write("- No column was transformed to numeric categories!")
            
        #------------------------------------------------------------------------------------------
        
        #++++++++++++++++++++++
        # UPDATED DATA SUMMARY   

        # Show only if changes were made
        if any(v for v in [sb_DM_delRows, sb_DM_delCols, sb_DM_dImp_num, sb_DM_dImp_other, sb_DM_dTrans_log, sb_DM_dTrans_sqrt, sb_DM_dTrans_square, sb_DM_dTrans_stand, sb_DM_dTrans_norm, sb_DM_dTrans_numCat ] if v is not None) or sb_DM_delDup == "Yes" or sb_DM_delRows_wNA == "Yes":
            dev_expander_dsPost = st.beta_expander("Explore cleaned and transformed data ", expanded = False)
            with dev_expander_dsPost:
                if df.shape[1] > 0 and df.shape[0] > 0:

                    # Show cleaned and transformed data & data info
                    df_summary_post = fc.data_summary(df)
                    if st.checkbox("Show cleaned and transformed data ", value = False, key = session_state.id):  
                        n_rows_post = df.shape[0]
                        n_cols_post = df.shape[1]
                        st.write(df)
                        st.write("Data shape: ", n_rows_post, "rows and ", n_cols_post, "columns")
                    if df[df.duplicated()].shape[0] > 0 or df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0] > 0:
                        check_nasAnddupl2 = st.checkbox("Show duplicates and NAs info (processed) ", value = False, key = session_state.id) 
                        if check_nasAnddupl2:
                            index_c = []
                            for c in df.columns:
                                for r in df.index:
                                    if pd.isnull(df[c][r]):
                                        index_c.append(r)      
                            if df[df.duplicated()].shape[0] > 0:
                                st.write("Number of duplicates: ", df[df.duplicated()].shape[0])
                                st.write("Duplicate row index: ", ', '.join(map(str,list(df.index[df.duplicated()]))))
                            if df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0] > 0:
                                st.write("Number of rows with NAs: ", len(pd.unique(sorted(index_c))))
                                st.write("Rows with NAs: ", ', '.join(map(str,list(pd.unique(sorted(index_c))))))

                    # Show cleaned and transformed variable info
                    if st.checkbox("Show cleaned and transformed variable info ", value = False, key = session_state.id): 
                        st.write(df_summary_post["Variable types"])

                    # Show summary statistics (cleaned and transformed data)
                    if st.checkbox('Show summary statistics (cleaned and transformed data) ', value = False, key = session_state.id):
                        st.write(df_summary_post["ALL"])
                        if fc.get_mode(df).loc["n_unique"].any():
                            st.caption("** Mode is not unique.") 
                        if sett_hints:
                            st.info(str(fc.learning_hints("de_summary_statistics")))     
                else: st.error("ERROR: No data available for Data Exploration!") 
                    
    #------------------------------------------------------------------------------------------
    
    data_visualization_container = st.beta_container()
    with data_visualization_container:
        #---------------------------------
        # DATA VISUALIZATION
        #---------------------------------
        st.write("")
        st.write("")
        st.header("**Data visualization**")
        
        #st.subheader("Graphical exploration")
        dev_expander_datavis = st.beta_expander("Check some data charts", expanded = False)
        with dev_expander_datavis:
            
            a4, a5= st.beta_columns(2)   
            with a4:
                # Scatterplot
                st.subheader("Scatterplot") 
                x_var = st.selectbox('Select x variable for your scatterplot', df.columns, key = session_state.id)    
                y_var = st.selectbox('Select y variable for your scatterplot', df.columns, key = session_state.id)
                
                fig = px.scatter(x=df[x_var], y=df[y_var], color_discrete_sequence=['rgba(55, 126, 184, 0.7)'])
                fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
                fig.update_layout(xaxis=dict(title=y_var, titlefont_size=12, tickfont_size=14,),)
                fig.update_layout(yaxis=dict(title=x_var, titlefont_size=12, tickfont_size=14,),)
                st.plotly_chart(fig,use_container_width=True) 
                
                if sett_hints:
                    st.info(str(fc.learning_hints("dv_scatterplot")))
            # st.altair_chart(s_chart, use_container_width=True)
                
            with a5:
                #Boxplot
                st.subheader("Boxplot")  
                bx_var = st.selectbox('Draw a boxplot for...?', df.columns, key = session_state.id)    
                st.markdown("") 
                st.markdown("") 
                st.markdown("")  
                st.markdown("") 
                st.markdown("") 
                st.markdown("") 
                    
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=df[bx_var],
                    name=str(bx_var),boxpoints='all', jitter=0.2,whiskerwidth=0.2,
                    marker_color = 'indianred', marker_size=2, line_width=1)
                )
                #fillcolor='rgba(31, 119, 180, 0.7)',
                fig.update_layout(font=dict(size=12,),)
                fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',})  
                st.plotly_chart(fig, use_container_width=True)

                if sett_hints:
                    st.info(str(fc.learning_hints("dv_boxplot")))


    #---------------------------------
    # METHODS
    #---------------------------------
    
    data_analyses_container = st.beta_container()
    with data_analyses_container:
        
        #---------------------------------
        # Frequency analysis
        #---------------------------------
        st.write("")
        st.write("")
        st.header('**Data analyses**')
        dev_expander_fa = st.beta_expander("Univariate frequency analysis", expanded = False)
        #initialisation
        fa_uniqueLim=30
        fa_low=None
        fa_up=None

        with dev_expander_fa:
                
            feature = st.selectbox('Which variable would you like to analyse?', df.columns, key = session_state.id)
            user_order=[] 

            #-----------------------------------------------
            # Identify the plot type (bar/hist) & key chart properties
            if ((df[feature]).unique()).size>fa_uniqueLim:
                default_BinNo=min(10,math.ceil(np.sqrt(df.size)))
                default_bins=default_BinNo
                plot_type="hist" # i.e. count freq. for value ranges
            elif df[feature].dtypes=="int64" or df[feature].dtypes=="float64" or df[feature].dtypes=="object" or df[feature].dtypes=="bool" or df[feature].dtypes=="category": 
                default_BinNo=((df[feature]).unique()).size
                default_bins=sorted(pd.Series(df[feature]).unique())
                plot_type="bars" # i.e. count freq. for every unique sample values
            else:
                st.error("ERROR: The variable type is not supported, please select another variable!")    
                return
            
            show_add_options_hist=st.checkbox("Show additional frequency analysis settings", value = False)
            if show_add_options_hist:  

                # for int or float the hist with custom bins can be used:    
                if plot_type=="hist": 
                    st.info('Sample value range for the variable "' +str(feature) + '":  \n min=' + str(min(df[feature]))+ '     max=' + str(max(df[feature])))
                    fa_low=st.number_input('Start frequency analysis from the "' + str(feature) + '" value?')
                    fa_up=st.number_input('End frequency analysis at the "' + str(feature) + '" value?')
                    default_bins = st.number_input('Specify the number of histogram classes',value=default_BinNo)
            
                unique_vals=pd.Series(df[feature]).unique()
                
                if plot_type=="bars": 
                    user_order=st.multiselect("In case you want to change the order of x-labels, select labels in order you prefer: the first one you select will be the first label and so on..",unique_vals)
                else:
                    user_order=[]
                #--------------------------------------------------------

            st.write("")  
            run_freq_anal = st.button("Run frequency analysis...")   
            st.write("")
            st.write("")
            fa_data=df[feature]
            

            if run_freq_anal:        
                
            
                #Absolute frequency
                if plot_type=="hist":
                    fig, ax = plt.subplots()
                    if(fa_low !=None and fa_up !=None):
                        fa_data=fa_data[fa_data.between(fa_low,fa_up, inclusive=True)] 
                        n, bins = np.histogram(np.array(fa_data),range=(fa_low,fa_up), bins=default_bins)
                    else:          
                        n, bins, patches=ax.hist(df[feature], bins=default_bins, rwidth=0.95, density=False)
                    
                    #write frequency analysis table
                    lower_limit=bins[0:(len(bins)-1)]
                    upper_limit=bins[1:(len(bins))]
                    class_mean=(bins[0:(len(bins)-1)]+bins[1:(len(bins))])/2
                    df_freqanal = pd.DataFrame(lower_limit,columns=['Lower limit'])
                    df_freqanal['Upper limit']= upper_limit
                    df_freqanal['Class mean']=class_mean
                    df_freqanal['Abs. freq.']= n
                    fa_val=n/sum(n)
                    df_freqanal['Rel. freq.']=fa_val
                    df_freqanal['Rel. freq.']=pd.Series(["{0:.2f}%".format(fa_val * 100) for fa_val in df_freqanal['Rel. freq.']], index = df_freqanal.index)
                    pd_s=pd.Series(n)
                    #calculate cumulative frequencies
                    cum_freq=pd_s.cumsum()/sum(pd_s)
                    df_freqanal['Cum. freq.']=cum_freq
                                
                    names=round(df_freqanal['Class mean'],2)
                                                
                else:
                    #count unique sample values
                    abs_freq=df[feature].value_counts().sort_index() 
                    if len(user_order)==0:
                        names=abs_freq.index
                    else:
                        abs_freq=abs_freq[user_order] 
                        names=abs_freq.index
                    fig, ax = plt.subplots()
                    ax.bar(names, abs_freq,tick_label=names)
                    
                    #write frequency analysis table
                    df_freqanal = pd.DataFrame(names,columns=['Class'])
                    df_freqanal.index =names
                    df_freqanal['Abs. freq.']= abs_freq
                    
                
                #Relative frequency
                if plot_type=="hist":
                    fig = go.Figure()
                
                    fig.add_trace(go.Bar(x=class_mean, y=fa_val, name='Relative frequency',marker_color = 'indianred',opacity=0.5))
                    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',})  
                    fig.update_layout(yaxis=dict(title='Relative frequency', titlefont_size=12, tickfont_size=14,),)
                    fig.update_layout(xaxis=dict(title=feature, titlefont_size=12, tickfont_size=14,),)
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    if len(user_order)==0:
                        rel_freq=df[feature].value_counts(normalize=True).sort_index() 
                        names=rel_freq.index
                    else:
                        rel_freq=df[feature].value_counts(normalize=True).sort_index() 
                        rel_freq=rel_freq[user_order] 
                        names=rel_freq.index
                    
                    fig = go.Figure()
                
                    fig.add_trace(go.Bar(x=names, y=rel_freq, name='Relative frequency',marker_color = 'indianred',opacity=0.5))
                    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',})  
                    fig.update_layout(yaxis=dict(title='Relative frequency', titlefont_size=12, tickfont_size=14,),)
                    fig.update_layout(xaxis=dict(title=feature, titlefont_size=12, tickfont_size=14,),)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    #write frequency analysis table
                    df_freqanal['Rel. freq.']= rel_freq
            #------------------------------------------------------------------------------------------
                
                #Cumulative frequency
                if plot_type=="bars":
                
                    #calculate frequency and sort it by labels
                    abs_freq=df[feature].value_counts().sort_index() 
                    pd_s=pd.Series(abs_freq)
                    #calculate cumulative frequencies
                    cum_freq=pd_s.cumsum()/sum(pd_s)
                    if len(user_order)==0:
                        names=cum_freq.index
                    else:
                        abs_freq=df[feature].value_counts()
                        pd_s=pd.Series(abs_freq[user_order])
                        cum_freq=pd_s.cumsum()/sum(abs_freq) 
                        names=cum_freq.index
                    
                    #write frequency analysis table
                    df_freqanal['Cum. freq.']= cum_freq

                #show frequency table
                st.subheader("Frequency table")
                st.table(df_freqanal)

        # -------------------
        # Anova
        #----------------------

        dev_expander_anova = st.beta_expander("ANOVA", expanded = False)
        with dev_expander_anova:
                
                
            # Target variable
            target_var = st.selectbox('Select the target variable', df.columns, key = session_state.id)
            if df[target_var].dtypes=="int64" or df[target_var].dtypes=="float64": 
                class_var_options = df.columns
                class_var_options = class_var_options[class_var_options.isin(df.drop(target_var, axis = 1).columns)]
                clas_var=st.selectbox('Select the classifier variable', class_var_options, key = session_state.id) 
            
            
                if len((df[clas_var]).unique())>20:
                    st.error("ERROR: The variable you selected is not suitable as a classifier!")
                else:
                    
                    run_anova = st.button("Press to perform one-way ANOVA")  
                    if run_anova:         
                        
                        # Boxplot
                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            x=df[clas_var],y=df[target_var],
                            name='', boxpoints='all', jitter=0.2,
                            whiskerwidth=0.2, marker_color = 'indianred',
                            marker_size=2, line_width=1))
                        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',})  
                        
                        
                        # Histogram
                        fig1 = px.histogram(df, height=400, x=target_var, color=clas_var)     
                        fig1.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
                        fig1.update_traces(opacity=0.35) 

                        a4,a5=st.beta_columns(2)
                        with a4:
                            st.subheader("Boxplots")
                            st.plotly_chart(fig, use_container_width=True)
                        with a5:
                            st.subheader("Histograms")
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        # ANOVA calculation & plots
                        df_grouped=df[[target_var,clas_var]].groupby(clas_var)
                        overal_mean=(df_grouped.mean()*df_grouped.count()).sum()/df_grouped.count().sum()
                        dof1=len(df_grouped.count())-1
                        dof2=df_grouped.count().sum()-len(df_grouped.count())
                        dof_tot=dof1+dof2

                        sqe_stat=(((df_grouped.mean()-overal_mean)**2)*df_grouped.count()).sum()
                        sqr_stat=(df_grouped.var()*(df_grouped.count()-1)).sum()
                        sq_tot=sqe_stat+sqr_stat
                        mqe_stat=sqe_stat/dof1
                        mqr_stat=sqr_stat/dof2
                        F_stat=mqe_stat/mqr_stat
                        p_value=scipy.stats.f.sf(F_stat, dof1, dof2)
                        

                        anova_summary=pd.concat([df_grouped.count(),df_grouped.mean(),df_grouped.var()], axis=1)
                        anova_summary.columns={"count", "mean" , "variance"}
                        st.subheader("Groups summary:")
                        st.table(anova_summary)

                        anova_table=pd.DataFrame({
                            "Deviance": [sqe_stat.values[0], sqr_stat.values[0], sq_tot.values[0]],
                            "DOF": [dof1, dof2.values[0], dof_tot.values[0]],
                            "Mean squared error": [mqe_stat.values[0], mqe_stat.values[0], ""],
                            "F": [F_stat.values[0], "", ""],
                            "p-value": [p_value[0], "", ""]},
                            index=["Between groups", "Within groups", "Total"],)
                        
                        st.subheader("ANOVA")
                        st.table(anova_table)

                        #Anova (OLS)
                        codes, uniques = pd.factorize(df[clas_var])
                        ano_ols = sm.OLS(df[target_var], sm.add_constant(codes))
                        ano_ols_output= ano_ols.fit()
                        residuals=df[target_var]-ano_ols_output.fittedvalues

                        # Q-Q plot residuals
                        qq_plot_data = pd.DataFrame()
                        qq_plot_data["Theoretical quantiles"] = stats.probplot(residuals, dist="norm")[0][0]
                        qq_plot_data["StandResiduals"] = sorted((residuals - residuals.mean())/residuals.std())
                        qq_plot = alt.Chart(qq_plot_data,height=400).mark_circle(size=20).encode(
                            x = alt.X("Theoretical quantiles", title = "theoretical quantiles", scale = alt.Scale(domain = [min(qq_plot_data["Theoretical quantiles"]), max(qq_plot_data["Theoretical quantiles"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            y = alt.Y("StandResiduals", title = "stand. residuals", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            tooltip = ["StandResiduals", "Theoretical quantiles"]
                        )
                        line = alt.Chart(
                            pd.DataFrame({"Theoretical quantiles": [min(qq_plot_data["Theoretical quantiles"]), max(qq_plot_data["Theoretical quantiles"])], "StandResiduals": [min(qq_plot_data["Theoretical quantiles"]), max(qq_plot_data["Theoretical quantiles"])]})).mark_line(size = 2, color = "darkred").encode(
                                    alt.X("Theoretical quantiles"),
                                    alt.Y("StandResiduals"),
                        )
                        
                        # histogram - residuals
                        res_hist = px.histogram(residuals, histnorm='probability density',opacity=0.7,color_discrete_sequence=['indianred'] ,height=400)                    
                        res_hist.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
                        res_hist.layout.showlegend = False                

                        a4,a5=st.beta_columns(2)
                        with a4:
                            st.subheader("Q-Q plot")
                            st.altair_chart(qq_plot + line, use_container_width=True)
                        with a5:
                            st.subheader("Histogram")
                            st.plotly_chart(res_hist, use_container_width=True)
                    
                        st.write("")
                        st.write("")
                        st.write("")
                
            else:
                st.error("ERROR: The target variable must be a numerical one!")
                
        # --------------------
        # Fit theoretical dist
        #----------------------

        dev_expander_ft = st.beta_expander("Distribution fitting", expanded = False)
        with dev_expander_ft:
            
            # initialisation:
            dist_names = ['Alpha','Anglit','Arcsine','Argus','Beta','Beta prime','Bradford','Burr (Type III)','Burr (Type XII)','Cauchy','Chi','Chi-squared','Cosine','Crystalball','Double gamma','Double Weibull','Erlang','Exponential','Exponentially modified Normal','Exponentiated Weibull','Exponential power','F','Fatigue-life (Birnbaum-Saunders)','Fisk','Folded Cauchy','Folded normal','Generalized logistic','Generalized normal','Generalized Pareto','Generalized exponential','Generalized extreme value','Gauss hypergeometric','Gamma','Generalized gamma','Generalized half-logistic','Generalized Inverse Gaussian','Gilbrat','Gompertz (or truncated Gumbel)','Right-skewed Gumbel','Left-skewed Gumbel','Half-Cauchy','Half-logistic','Half-normal','The upper half of a generalized normal','Hyperbolic secant','Inverted gamma','Inverse Gaussian','Inverted Weibull','Johnson SB','Johnson SU','Kappa 4 parameter','Kappa 3 parameter','Laplace','Asymmetric Laplace','Levy','Left-skewed Levy','Logistic (or Sech-squared)','Log gamma','Log-Laplace','Lognormal','Loguniform or reciprocal','Lomax (Pareto of the second kind)','Maxwell','Mielke Beta-Kappa / Dagum','Moyal','Nakagami','Non-central chi-squared','Non-central F distribution','Non-central Student’s t','Normal','Normal Inverse Gaussian','Pareto','Pearson type III','Power-function','Power log-normal','Power normal','R-distributed (symmetric beta)','Rayleigh','Rice','Reciprocal inverse Gaussian','Semicircular','Skew-normal','Student’s t','Trapezoidal','Triangular','Truncated exponential','Truncated normal','Tukey-Lamdba','Uniform','Von Mises','Von Mises','Wald','Weibull minimum','Weibull maximum','Wrapped Cauchy'] 
            num_columns=df.columns
            df_ft = df.copy()
            df_ft=df_ft.dropna() # just to make sure that NAs are removed in case the user hasn't done it before
            iniBins=30
            ft_low=None
            ft_up=None
            
            # check variable type
            for column in df_ft:            
                if not df_ft[column].dtypes in ('float', 'float64', 'int','int64'): 
                    num_columns=num_columns.drop(column) 
                    
            if len(num_columns)==0:
                st.error("ERROR: None of your data is suitable for distribution fitting! Please try another dataset!")    
            else:
                # Variable selection:
                ft_var=st.selectbox("Select a variabe for dist. fitting",num_columns, key = session_state.id)
                ft_data=df[ft_var]

                ft_selection=st.radio("Please choose if you would like to fit all distributions or a selection of distributions:", ["all (Note, this may take a while!)", "selection"], index=1, key = session_state.id)
                
                if ft_selection=='selection':
                    # Theoretical distribution:
                    ft_dist=st.multiselect("Select theoretical distributions for distribution fitting", dist_names, ['Normal','Lognormal','Weibull minimum'], key = session_state.id)
                else:
                    ft_dist=dist_names
                if len(ft_data)>4 and len(ft_data)<500:
                    iniBins=10*math.ceil(((len(ft_data))**0.5)/10)
                
                # Number of classes in the empirical distribution
                Nobins =iniBins
                ft_showOptions=st.checkbox("Additional settings?", value = False, key = session_state.id)
                if ft_showOptions:                         
                    st.info('Sample value range for the variable "' +str(ft_var) + '":  \n min=' + str(min(df[ft_var]))+ '  \n max=' + str(max(df[ft_var])))
                    ft_low=st.number_input('Start fitting from the "' + str(ft_var) + '" value?')
                    ft_up=st.number_input('End fitting at the "' + str(ft_var) + '" value?')
                    Nobins = st.number_input('Number of classes for your histogram?',value=iniBins)
                                    
        
                st.write("")
                run_fitting = st.button("Press to start the distribution fitting...")  
                st.write("") 
                st.write("")
                if run_fitting:

                    # status bar
                    st.info("Distribution fitting progress")

                    # Fit theoretical distribution to empirical data 
                    if(ft_low !=None and ft_up !=None):
                        ft_data=ft_data[ft_data.between(ft_low,ft_up, inclusive=True)] 
                    results, x_mid,xlower,xupper, min_ssd, max_p, clas_prob_res, best_name, best_dist, best_params = fc.fit_scipy_dist(ft_data, Nobins, ft_dist, ft_low, ft_up)
                    y, x = np.histogram(np.array(ft_data), bins=Nobins)
                    rel_freq=y/len(ft_data)
                    best_prob=clas_prob_res[min_ssd]
                    
                    #table output
                    st.success('Distribution fitting done!')
                    st.subheader("Goodness-of-fit results")
                    st.table(results[['SSD', 'Chi-squared','DOF', 'p-value','Distribution']])
                    
                    
                    if sum(results['p-value']>0.05)>0:
                        st.info('The sum of squared diferences (SSD) is smallest for the "' +best_name + ' distribution"  \n The p-value of the $\chi^2$ statistics is largest for the "'+ max_p +' distribution"')
                    else:
                        st.info('The sum of squared diferences (SSD) is smallest for the "' +best_name + ' distribution"  \n The p-value of the $\chi^2$ statistics is for none of the distributions above 5%')
                    st.subheader("A comparision of relative frequencies")
                    rel_freq_comp=pd.DataFrame(xlower,columns=['Lower limit'])
                    rel_freq_comp['Upper limit']= xupper
                    rel_freq_comp['Class mean']=x_mid
                    rel_freq_comp['Rel. freq. (emp.)']= rel_freq
                    rel_freq_comp['Rel. freq. (emp.)'] = pd.Series(["{0:.2f}%".format(val * 100) for val in rel_freq_comp['Rel. freq. (emp.)']], index = rel_freq_comp.index)
                    
                    rel_freq_comp['Rel. freq. ('+best_name +')']= best_prob
                    rel_freq_comp['Rel. freq. ('+best_name +')'] = pd.Series(["{0:.2f}%".format(val * 100) for val in rel_freq_comp['Rel. freq. ('+best_name +')']], index = rel_freq_comp.index)
                
                    st.table(rel_freq_comp)
                            
                    # Plot results:
                    st.subheader("A comparision of relative frequencies (empirical vs. theoretical)")
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(x=x_mid, y=rel_freq, name='empirical',marker_color = 'indianred',opacity=0.5))
                    fig.add_trace(go.Bar(x=x_mid, y=best_prob, name=best_name,marker_color = 'rgb(26, 118, 255)',opacity=0.5))
                    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',})  
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("")
                    if sett_hints:
                        st.info(str(fc.learning_hints("fit_hints")))
                        st.write("")
        # ---------------------
        # Corr. analysis
        #----------------------

        dev_expander_qa = st.beta_expander("Correlation analysis", expanded = False)
        with dev_expander_qa:
            
            df_cor=df.copy()
            non_trans_cols=df_cor.columns
            # check variable type
            transf_cols=[]
            for column in df_cor:            
                if not df_cor[column].dtypes in ('float', 'float64', 'int','int64'): 
                    non_trans_cols=non_trans_cols.drop(column) 
                    df_cor[column]=pd.factorize(df_cor[column])[0]
                    transf_cols.append([column])

            
            if len(non_trans_cols)<len(df_cor.columns):
                #st.info("All columns except " + str(non_trans_cols.values) + " are factorized to enable regression analysis!")    
                if transf_cols !=None:
                    st.info('Please note, non-numerical variables are factorized to enable regression analysis (i.e., automatically transformed to numerical variables)! The factorized variables are: ' + str(transf_cols)) 
            listOfAllColumns = df_cor.columns.to_list()   
            cor_sel_var=st.multiselect("Select variabes for correlation analysis",listOfAllColumns, listOfAllColumns, key = session_state.id)
            cor_methods=['Pearson', 'Kendall', 'Spearman']
            cor_method=st.selectbox("Select the method",cor_methods, key = session_state.id)
            if cor_method=='Pearson':
                sel_method='pearson'
            elif cor_method=='Kendall':
                sel_method='kendall'
            else:
                sel_method='spearman'    

            if st.checkbox("Show data for correlation analysis", value = False, key = session_state.id):        
                st.write(df_cor)
            st.write("")
            st.write("")    

            run_corr = st.button("Press to start the correlation analysis...")           
            st.write("")
            st.write("")

            if run_corr:
                if sett_hints:
                    st.info(str(fc.learning_hints("correlation_hints")))
                    st.write("")
                
                st.subheader("Correlation matrix")
            # Define variable selector
                var_sel_cor = alt.selection_single(fields=['variable', 'variable2'], clear=False, 
                                    init={'variable': cor_sel_var[0], 'variable2': cor_sel_var[0]})
                # Calculate correlation data
                corr_data = df_cor[cor_sel_var].corr(method=sel_method).stack().reset_index().rename(columns={0: "correlation", 'level_0': "variable", 'level_1': "variable2"})
                corr_data["correlation_label"] = corr_data["correlation"].map('{:.2f}'.format)
                # Basic plot
                base = alt.Chart(corr_data).encode(
                    x = alt.X('variable2:O', sort = None, axis = alt.Axis(title = None, labelFontSize = 12)),
                    y = alt.Y('variable:O',  sort = None, axis = alt.Axis(title = None, labelFontSize = 12))
                )
                # Correlation values to insert
                text = base.mark_text().encode(
                    text='correlation_label',
                    color = alt.condition(
                        alt.datum.correlation > 0.5, 
                        alt.value('white'),
                        alt.value('black')
                    )
                )
                # Correlation plot
                corr_plot = base.mark_rect().encode(
                    color = alt.condition(var_sel_cor, alt.value('#86c29c'), 'correlation:Q', legend = alt.Legend(title = "Bravais-Pearson correlation coefficient", orient = "top", gradientLength = 350), scale = alt.Scale(scheme='redblue', reverse = True, domain = [-1,1]))
                ).add_selection(var_sel_cor)
                # Calculate values for 2d histogram
                value_columns = df_cor[cor_sel_var]
                df_2dbinned = pd.concat([fc.compute_2d_histogram(var1, var2, df_cor) for var1 in value_columns for var2 in value_columns])
                # 2d binned histogram plot
                scat_plot = alt.Chart(df_2dbinned).transform_filter(
                    var_sel_cor
                ).mark_rect().encode(
                    alt.X('value2:N', sort = alt.EncodingSortField(field='raw_left_value2'), axis = alt.Axis(title = "Horizontal variable", labelFontSize = 12)), 
                    alt.Y('value:N', axis = alt.Axis(title = "Vertical variable", labelFontSize = 12), sort = alt.EncodingSortField(field='raw_left_value', order = 'descending')),
                    alt.Color('count:Q', scale = alt.Scale(scheme='reds'),  legend = alt.Legend(title = "Count", orient = "top", gradientLength = 350))
                )
                # Combine all plots
                correlation_plot = alt.vconcat((corr_plot + text).properties(width = 400, height = 400), scat_plot.properties(width = 400, height = 400)).resolve_scale(color = 'independent')
                correlation_plot = correlation_plot.properties(padding = {"left": 50, "top": 5, "right": 5, "bottom": 50})
                st.altair_chart(correlation_plot, use_container_width = True)

                st.subheader("Correlation table")
                st.table(df_cor.corr(method=sel_method))

        #--------------------------------------
        # Regression analysis
        #--------------------------------------

        dev_expander_reg = st.beta_expander("Regression techniques", expanded = False) 
        with dev_expander_reg:
            
            # initialisation:
            ra_techniques_names=['Simple Linear Regression', 'Linear-Log Regression', 'Log-Linear Regression', 'Log-Log Regression','Polynomial Regression']
            poly_order=2 # default for the polynomial regression
            num_columns=df.columns
            
            df_ra = df.copy()
            df_ra=df_ra.dropna() # just to make sure that NAs are removed in case the user hasn't done it before
            
            
            # check variable type
            for column in df_ra:            
                if not df_ra[column].dtypes in ('float', 'float64', 'int','int64'): 
                    num_columns=num_columns.drop(column) 
                    
            if len(num_columns)<2:
                st.error("ERROR: Your data is not suitable for regression analysis! Please try another dataset!")    
            else:
                # Variable selection:
                ra_Xvar=st.selectbox("Select the X variabe for regression analysis", num_columns, key = session_state.id)
                ra_Yvar=st.selectbox("Select the Y variabe for regression analysis",num_columns,index=1, key = session_state.id)

                if ra_Xvar==ra_Yvar:
                    st.error("ERROR: Regressing a variable against itself doesn't make much sense!")             
                else:      
                    # regression techniques selection:
                    ra_selection=st.radio("Please choose if you would like to apply all regression techniques or a selection of techniques:", ["all", "selection"], index=1, key = session_state.id)
                    if ra_selection=='selection':
                        ra_tech=st.multiselect("Select regression techniques ", ra_techniques_names, ['Simple Linear Regression'], key = session_state.id)
                    else:
                        ra_tech=ra_techniques_names

                    
                    # Additional settings
                    #ra_showOptions=st.checkbox("Additional regression analysis settings?", value = False)
                    
                    if 'Polynomial Regression' in ra_tech:
                        poly_order=st.number_input('Specify the polynomial order for the polynomial regression',value=2, step=1)    
                    
                    ra_detailed_output=st.checkbox("Show detailed output per technique?", value = False, key = session_state.id)


                    # specify the dataset for the regression analysis:
                    X_ini=df[ra_Xvar]
                    Y_ini=df[ra_Yvar]
                    expl_var=ra_Xvar

                    st.write("")
                    if len(ra_tech)==0:
                        st.error('ERROR: Please select at least one regression technique!')
                    else:
                        st.write("") 
                        run_regression = st.button("Press to start the regression analysis...") 
                        
                        X_label_prefix={'Simple Linear Regression':'',
                            'Linear-Log Regression': 'log_',
                            'Log-Linear Regression': '',
                            'Log-Log Regression': 'log_',
                            'Polynomial Regression':''}
                        Y_label_prefix={'Simple Linear Regression':'',
                            'Linear-Log Regression': '',
                            'Log-Linear Regression': 'log_',
                            'Log-Log Regression': 'log_',
                            'Polynomial Regression':''}    
                        st.write("") 
                        if run_regression:

                            st.info("Regression progress")
                            ra_bar = st.progress(0.0)
                            progress = 0
                            n_techniques=len(ra_tech)
                            
                            if sett_hints:
                                st.write("")
                                st.info(str(fc.learning_hints("reg_hints")))
                                st.write("")

                            model_comparison = pd.DataFrame(index = ["R²","Adj. R²","Log-likelihood", "MSE", "RMSE", "MAE", "MaxErr", "EVRS", "SSR"], columns = ra_techniques_names)
                            
                            for reg_technique in ra_tech:                            
                                mlr_reg_inf, mlr_reg_stats, mlr_reg_anova, mlr_reg_coef,X_data, Y_data, Y_pred = fc.regression_models(X_ini, Y_ini, expl_var,reg_technique,poly_order)
                                # Model comparison
                                model_comparison.loc["R²"][reg_technique] = (mlr_reg_stats.loc["R²"]).Value
                                model_comparison.loc["Adj. R²"][reg_technique] = (mlr_reg_stats.loc["Adj. R²"]).Value 
                                model_comparison.loc["Log-likelihood"][reg_technique] = (mlr_reg_stats.loc["Log-likelihood"]).Value 
                                #model_comparison.loc["% VE"][reg_technique] =  r2_score(Y_data, Y_pred)
                                model_comparison.loc["MSE"][reg_technique] = mean_squared_error(Y_data, Y_pred, squared = True)
                                model_comparison.loc["RMSE"][reg_technique] = mean_squared_error(Y_data, Y_pred, squared = False)
                                model_comparison.loc["MAE"][reg_technique] = mean_absolute_error(Y_data, Y_pred)
                                model_comparison.loc["MaxErr"][reg_technique] = max_error(Y_data, Y_pred)
                                model_comparison.loc["EVRS"][reg_technique] = explained_variance_score(Y_data, Y_pred)
                                model_comparison.loc["SSR"][reg_technique] = ((Y_data-Y_pred)**2).sum()
                                                        
                                # scatterplot with the initial data:
                                x_label=str(X_label_prefix[reg_technique])+str(ra_Xvar)
                                y_label=str(Y_label_prefix[reg_technique])+str(ra_Yvar)
                                reg_plot_data = pd.DataFrame()
                                
                                if reg_technique=='Polynomial Regression':
                                    reg_plot_data[ra_Xvar] = X_ini 
                                else:                                         
                                    reg_plot_data[ra_Xvar] = X_data

                                reg_plot_data[ra_Yvar] = Y_data
                                reg_plot = alt.Chart(reg_plot_data,height=400).mark_circle(size=20).encode(
                                    x = alt.X(ra_Xvar, scale = alt.Scale(domain = [min(reg_plot_data[ra_Xvar]), max(reg_plot_data[ra_Xvar])]), axis = alt.Axis(title=x_label, titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y(ra_Yvar, scale = alt.Scale(domain = [min(min(reg_plot_data[ra_Yvar]),min(Y_pred)), max(max(reg_plot_data[ra_Yvar]),max(Y_pred))]), axis = alt.Axis(title=y_label,titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = [alt.Tooltip(shorthand=ra_Xvar,title=x_label), alt.Tooltip(shorthand=ra_Yvar,title=y_label)]
                                )
                                
                                # model fit plot 
                                line_plot_data = pd.DataFrame()
                                if reg_technique=='Polynomial Regression':
                                    line_plot_data[ra_Xvar] = X_ini 
                                else:       
                                    line_plot_data[ra_Xvar] = X_data
                                line_plot_data[ra_Yvar] = Y_pred
                                
                                line_plot = alt.Chart(line_plot_data,height=400).mark_circle(size=20).mark_point(opacity=0.2, color='darkred').encode(
                                    x = alt.X(ra_Xvar, scale = alt.Scale(domain = [min(reg_plot_data[ra_Xvar]), max(reg_plot_data[ra_Xvar])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y(ra_Yvar, scale = alt.Scale(domain = [min(min(reg_plot_data[ra_Yvar]),min(Y_pred)), max(max(reg_plot_data[ra_Yvar]),max(Y_pred))]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = [alt.Tooltip(shorthand=ra_Xvar,title=x_label), alt.Tooltip(shorthand=ra_Yvar,title=y_label)]
                                )
                                
                                
                                # Q-Q plot residuals
                                qq_plot_data = pd.DataFrame()
                                residuals=Y_data-Y_pred
                                qq_plot_data["Theoretical quantiles"] = stats.probplot(residuals, dist="norm")[0][0]
                                qq_plot_data["StandResiduals"] = sorted((residuals - residuals.mean())/residuals.std())
                                qq_plot = alt.Chart(qq_plot_data,height=400).mark_circle(size=20).encode(
                                    x = alt.X("Theoretical quantiles", title = "theoretical quantiles", scale = alt.Scale(domain = [min(qq_plot_data["Theoretical quantiles"]), max(qq_plot_data["Theoretical quantiles"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("StandResiduals", title = "stand. residuals", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["StandResiduals", "Theoretical quantiles"]
                                )
                                line = alt.Chart(
                                    pd.DataFrame({"Theoretical quantiles": [min(qq_plot_data["Theoretical quantiles"]), max(qq_plot_data["Theoretical quantiles"])], "StandResiduals": [min(qq_plot_data["Theoretical quantiles"]), max(qq_plot_data["Theoretical quantiles"])]})).mark_line(size = 2, color = "darkred").encode(
                                            alt.X("Theoretical quantiles"),
                                            alt.Y("StandResiduals"),
                                )
                                    
                                st.subheader(reg_technique)

                                if ra_detailed_output:
                                    #st.table(mlr_reg_stats)
                                    st.table(mlr_reg_anova)
                                    st.table(mlr_reg_coef)                           
                                    a4,a5=st.beta_columns(2)
                                    with a4:
                                        st.subheader("Regression plot")
                                        st.altair_chart(reg_plot +line_plot, use_container_width=True) 
                                    with a5:
                                        st.subheader("Q-Q plot")
                                        st.altair_chart(qq_plot + line, use_container_width=True)
                                else: 
                                    a4,a5=st.beta_columns(2)
                                    with a4:
                                        st.subheader("Regression plot")
                                        st.altair_chart(reg_plot +line_plot, use_container_width=True) 
                                    with a5:
                                        st.subheader("Q-Q plot")
                                        st.altair_chart(qq_plot + line, use_container_width=True)   

                                progress += 1
                                ra_bar.progress(progress/n_techniques)
                            
                            st.subheader('Regression techniques summary') 
                            model_output=(model_comparison[ra_tech]).transpose()   
                            st.write(model_output)   

        #------------------------------------------------------
        #-------------------------------------------------------


        #-------------------------------------
        # Contingency analysis
        #--------------------------------------
        
        dev_expander_conting = st.beta_expander("Contingency tables and association measures", expanded = False)
        cont_check=False
        cont_unique_limit=30
        data_reclas=0
        with dev_expander_conting:
            if len(df.columns)>2:
                dfnew=df[df.columns[0:2]] 
                listOfColumns = dfnew.columns.to_list()
                listOfAllColumns = df.columns.to_list()
                st.info("Your dataset has more than 2 variables! In case you want to analyse multivariate data, please select 'Multivariate data' on the sidebar. In case you want to analyse uni- and bivariate data please select up to two variables to analyse right below...")
                
                ub_sel_var=st.multiselect("Please select up to 2 variables to analyse",listOfAllColumns, listOfColumns, key = session_state.id)
                #dataframe with selected data:
                df=df[ub_sel_var]
                if (len(df[ub_sel_var].columns)>2):
                    extra_cols=(len(df[ub_sel_var].columns)-2)
                    st.error("ERROR: Please remove " + str(extra_cols) + " variables from your selection!" )
                    return
                elif (len(df[ub_sel_var].columns)<2):  
                    add_cols=(2-len(df[ub_sel_var].columns))
                    st.error("ERROR: Please add " + str(add_cols) + " variables to the analysis set!" )
                    return 
            # check the number of unique values - if tooooo many, then it is not suitable for contingency analysis
            if (len(df.iloc[:,0].unique()))>cont_unique_limit or (len(df.iloc[:,1].unique())) >cont_unique_limit:
                st.error('ERROR: Your dataset has too many unique values and is not suitable for contingency analysis!')
                cont_group=st.selectbox("You can try some of these options:", ['-','Reclassify my data','Use my data anyway' ], key = session_state.id) 
                if cont_group=='-':
                    return
                                            
                elif cont_group=='Use my data anyway':
                    st.info('I will work with your data, but the results might be nonsense due to too many unique values')    
                    cont_check=True
                else: 
                    # check if data-reclassification is managebeable:
                    cont_numerical=[]   # no of numerical columns in df
                    cont_categs=[]      # number of categorical columns in df
                    if df.iloc[:,0].dtypes in ('float', 'float64', 'int','int64'):
                        cont_numerical.append(df.columns[0])
                    elif len(df.iloc[:,0].unique())<cont_unique_limit:
                        cont_categs.append(df.columns[0])

                    if df.iloc[:,1].dtypes in ('float', 'float64', 'int','int64'):
                        cont_numerical.append(df.columns[1])
                    elif len(df.iloc[:,1].unique())<cont_unique_limit:
                        cont_categs.append(df.columns[1])

                    
                    if len(cont_numerical)==2:
                        data_reclas=2
                        cont_check=True 
                        a4,a5=st.beta_columns(2) 
                        with a4:
                            st.subheader(str(cont_numerical[0]))
                            st.table(pd.DataFrame(data={'min': [min(df[cont_numerical[0]])], 'max': [max(df[cont_numerical[0]])]},index=['range']))
                            low0=st.number_input(str(cont_numerical[0]) + ': 1st class should start at?')
                            up0=st.number_input(str(cont_numerical[0]) + ': Max limit for your classes?')
                            noclass0= st.number_input(str(cont_numerical[0]) + ': Number of classes?')
                        
                        with a5:
                            st.subheader(str(cont_numerical[1]))
                            st.table(pd.DataFrame(data={'min': [min(df[cont_numerical[1]])], 'max': [max(df[cont_numerical[1]])]},index=['range']))
                            low1=st.number_input(str(cont_numerical[1]) + ': 1st class should start at?')
                            up1=st.number_input(str(cont_numerical[1]) + ': Max limit for your classes?')
                            noclass1= st.number_input(str(cont_numerical[1]) + ': Number of classes?')
                        
                    elif len(cont_numerical)==1 and len(cont_categs)==1:
                        data_reclas=1
                        cont_check=True 
                        a4,a5=st.beta_columns(2) 
                        with a4:
                            st.subheader(str(cont_numerical[0]))
                            st.table( {'min': [min(df[cont_numerical[0]])], 'max': [max(df[cont_numerical[0]])]})
                            low0=st.number_input(str(cont_numerical[0]) + ': 1st class should start at?')
                            up0=st.number_input(str(cont_numerical[0]) + ': Max limit for your classes?')
                            noclass0= st.number_input(str(cont_numerical[0]) + ': Number of classes?')
                        
                    
                    else:           
                        st.info("Please try data reclassification outside of Staty as the sort of classification you might need is not yet implemented!")    
                
                
            else:
                cont_check=True

            # central part - contigency analysis
            if cont_check==True:
                cont_extra=st.checkbox("Show marginal frequencies", value = False, key = session_state.id)        
                    

                if st.checkbox("Show data for contingency analysis", value = False, key = session_state.id):        
                    st.write(df)

                st.write("")            
                run_cont = st.button("Press to start the data processing...")           
                st.write("")
                st.write("")    
             
                if run_cont:

                    if sett_hints:
                        st.info(str(fc.learning_hints("contingency_hints")))
                        st.write("")

                    if data_reclas==2:
                        step0=(up0-low0)/noclass0
                        lim_ser = pd.Series(np.arange(low0, up0, step0)) 
                        for k in range(len(lim_ser)-1):
                            df.loc[(df[cont_numerical[0]]>lim_ser[k]) & (df[cont_numerical[0]] <= lim_ser[k+1]), cont_numerical[0]] = lim_ser[k]
                        df.loc[df[cont_numerical[0]]>max(lim_ser), cont_numerical[0]] = '>'+ str(max(lim_ser))#+step0

                        step1=(up1-low1)/noclass1
                        lim_ser1 = pd.Series(np.arange(low1, up1, step1))
                        for k in range(len(lim_ser1)-1):
                            df.loc[(df[cont_numerical[1]]>lim_ser1[k]) & (df[cont_numerical[1]] <= lim_ser1[k+1]), cont_numerical[1]] = lim_ser1[k]
                        df.loc[df[cont_numerical[1]]>max(lim_ser1), cont_numerical[1]] = '>'+ str(max(lim_ser1))
                    elif data_reclas==1:
                        step0=(up0-low0)/noclass0
                        lim_ser = pd.Series(np.arange(low0, up0, step0)) 
                        for k in range(len(lim_ser)-1):
                            
                            df.loc[(df[cont_numerical[0]]>lim_ser[k]) & (df[cont_numerical[0]] <= lim_ser[k+1]), cont_numerical[0]] = lim_ser[k]
                        df.loc[df[cont_numerical[0]]>max(lim_ser), cont_numerical[0]] = '>'+ str(max(lim_ser))
    
                
                    bivarite_table = pd.crosstab(index= df.iloc[:,0], columns= df.iloc[:,1] , margins=True, margins_name="Total")
                    stat, p, dof, expected = chi2_contingency(pd.crosstab(index= df.iloc[:,0], columns= df.iloc[:,1])) 
                        
                    st.subheader('Contingency table with absolute frequencies')    
                    st.table(bivarite_table)
                    
                    no_vals=bivarite_table.iloc[len(bivarite_table)-1,len(bivarite_table.columns)-1]
                    st.subheader('Contingency table with relative frequencies')
                    cont_rel=bivarite_table/no_vals
                    st.table(cont_rel)

                    if cont_extra:
                        st.subheader('Contingency table with marginal frequencies ('+ str(ub_sel_var[0])+')')
                        bivarite_table_marg0 =bivarite_table.iloc[:,:].div(bivarite_table.Total, axis=0)
                        st.table(bivarite_table_marg0)

                        st.subheader('Contingency table with marginal frequencies ('+ str(ub_sel_var[1])+')')
                        bivarite_table_marg1 =bivarite_table.div(bivarite_table.iloc[-1])
                        st.table(bivarite_table_marg1)

                        # draw a bar plot with the results
                        colors = px.colors.qualitative.Pastel2
                        data0=bivarite_table_marg0.loc[bivarite_table_marg0.index !='Total', bivarite_table_marg0.columns != 'Total']
                    
                        fig = px.bar(data0, 
                            #x = data0.index,
                        # y = [c for c in data0.columns],
                            color_discrete_sequence = colors,
                            )
                        fig.update_layout(barmode='stack')
                        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
                        
                        
                        data1=bivarite_table_marg1.loc[bivarite_table_marg1.index !='Total', bivarite_table_marg1.columns != 'Total']
                        data1=data1.transpose()
                        
                        fig1 = px.bar(data1, 
                        # x = data1.index,
                        # y = [c for c in data1.columns],
                            color_discrete_sequence = colors,
                            )
                        fig1.update_layout(barmode='stack')
                        fig1.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
                        
                        st.subheader("Bar plots of marginal frequencies")
                        a4, a5= st.beta_columns(2) 
                        with a4:
                            st.plotly_chart(fig, use_container_width=True) 
                        with a5:
                            st.plotly_chart(fig1, use_container_width=True)

                    #--------------------------------------------
                    st.subheader('Contingency table with expected frequencies')
                    dfexp=pd.DataFrame(expected)
                    dfexp.index =bivarite_table.index[0:len(bivarite_table.index)-1]
                    col_names=list(bivarite_table.columns[0:len(bivarite_table.columns)-1])
                    dfexp.columns= col_names
                    st.table(dfexp)
                            
                    st.subheader('Contingency stats') 
                    st.write('')
                    st.write('$\chi^2$ = ' + str(stat))
                    st.write('p-value = ' + str(p))

                    pearson_coef=np.sqrt(stat/(stat+len(df)))  
                    st.write('Pearson contingency coefficient $K$ = ' + str(pearson_coef))

                    min_kl=min(len(bivarite_table)-1,len(bivarite_table.columns)-1)
                    K_max=np.sqrt((min_kl-1)/min_kl)
                    st.write('$K_{max}$ = ' + str(K_max))
                        
                    pearson_cor=pearson_coef/K_max
                    st.write('Corrected Pearson contingency coefficient $K_{cor}$ = ' + str(pearson_cor))

                    #st.latex(r'''\chi^2=\sum_{i=1}^{k}\sum_{j=1}^{l}\frac{\left( n_{ij}-\tilde{n}_{ij} \right) ^3}{n_{ij}}''')

    

    


# Pie chart
        #st.subheader('Pie chart')
        #if plot_type=="hist":
        #    labels=round(df_freqanal['Upper limit'],1)
        #else:
        #    labels=df_freqanal['Class']    
        #values = df_freqanal['Rel. freq.']
        
        # Use hole to create a donut-like pie chart
        #fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        #fig.update_traces(hoverinfo='label+value', marker=dict(colors=px.colors.sequential.Blues,line=dict(color='#FFFFFF', width=1.0)))
       # st.plotly_chart(fig, use_container_width=True)


