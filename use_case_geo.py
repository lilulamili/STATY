import streamlit as st
import pandas as pd
import numpy as np
import functions as fc
import os
import pydeck as pdk
import altair as alt
import datetime
import time
import plotly.express as px 
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit import caching
import SessionState
import sys
import platform


def app():

    # Clear cache
    caching.clear_cache()

    # Hide traceback in error messages (comment out for de-bugging)
    sys.tracebacklimit = 0

   
    # File upload section
    df_dec = st.sidebar.radio("Get data", ["Use example dataset", "Upload your data"])

    if df_dec == "Upload your data":
        #st.subheader("Upload your data")
        uploaded_data = st.sidebar.file_uploader("Upload data:", type=["csv", "txt"])
        if uploaded_data is not None:
            df = pd.read_csv(uploaded_data, sep = ";|,|\t",engine='python')
            st.sidebar.success('Loading data... done!')
            if len(df)<4:
                small_dataset_error ="The sample is so small that you should better use a pocket calculator as the learning effect will be larger!"
        elif uploaded_data is None:            
            df = pd.read_csv("\\default data\\WHR_2021.csv", sep = ";|,|\t",engine='python')
           
    else:        
        df = pd.read_csv("\\default data\\WHR_2021.csv", sep = ";|,|\t",engine='python')
        
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
    # DATA EXPLORATION & VISUALISATION

    st.header("**Geospatial data/Interactive dashboards**")
    st.markdown("Let STATY do the data cleaning, variable transformations, visualisations and deliver you the stats you need. Specify your data processing preferences and start exploring your data stories right below... ")

    if len(df)<4:
        st.error(small_dataset_error)
        return

    st.header("**Data exploration**")
    #------------------------------------------------------------------------------------------

    #++++++++++++++++++++++
    # DATA SUMMARY

    # Main panel for data summary (pre)
    #----------------------------------
    data_exploration_container = st.beta_container()
    with data_exploration_container:
        dev_expander_raw = st.beta_expander("Explore raw data")
        with dev_expander_raw:

            # Show raw data & data info
            df_summary = fc.data_summary(df) 
            if st.checkbox("Show raw data", value = False, key = session_state.id):      
                #st.dataframe(df.style.apply(lambda x: ["background-color: #ffe5e5" if (not pd.isna(df_summary_mq_full.loc["1%-Q"][i]) and df_summary_vt_cat[i] == "numeric" and (v <= df_summary_mq_full.loc["1%-Q"][i] or v >= df_summary_mq_full.loc["99%-Q"][i]) or pd.isna(v)) else "" for i, v in enumerate(x)], axis = 1))
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
                a7, a8 = st.beta_columns(2)
                with a7:
                    st.table(df_summary["Variable types"])
            # Show summary statistics (raw data)
            if st.checkbox('Show summary statistics (raw data)', value = False, key = session_state.id): 
                #st.write(df_summary["ALL"])
                df_datasumstat=df_summary["ALL"]
                #dfStyler = df_datasumstat.style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector = 'th', props=[('text-align', 'left')])]) 
                a7, a8 = st.beta_columns(2)
                with a7:
                    st.table(df_datasumstat)
                    if fc.get_mode(df).loc["n_unique"].any():
                        st.caption("** Mode is not unique.")

        #++++++++++++++++++++++
        # DATA PROCESSING

        # Settings for data processing
        #-------------------------------------
        
        dev_expander_dm_sb = st.beta_expander("Specify data processing preferences", expanded = False)
        with dev_expander_dm_sb:
            
            n_rows_wNAs = df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0]
            if n_rows_wNAs > 0:
                a1, a2, a3 = st.beta_columns(3)
            else: a1, a2 = st.beta_columns(2)

            sb_DM_dImp_num = None 
            sb_DM_dImp_other = None
            group_by_num = None
            group_by_other = None
            if n_rows_wNAs > 0:
                with a1:
                    #--------------------------------------------------------------------------------------
                    # DATA CLEANING

                    st.markdown("**Data cleaning**")

                    # Delete duplicates if any exist
                    if df[df.duplicated()].shape[0] > 0:
                        sb_DM_delDup = st.selectbox("Delete duplicate rows", ["No", "Yes"], key = session_state.id)
                        if sb_DM_delDup == "Yes":
                            n_rows_dup = df[df.duplicated()].shape[0]
                            df = df.drop_duplicates()
                    elif df[df.duplicated()].shape[0] == 0:   
                        sb_DM_delDup = "No"    
                        
                    # Delete rows with NA if any exist
                    n_rows_wNAs = df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0]
                    if n_rows_wNAs > 0:
                        sb_DM_delRows_wNA = st.selectbox("Delete rows with NAs", ["No", "Yes"], key = session_state.id)
                        if sb_DM_delRows_wNA == "Yes": 
                            df = df.dropna()
                    elif n_rows_wNAs == 0: 
                        sb_DM_delRows_wNA = "No"   

                    # Delete rows
                    sb_DM_delRows = st.multiselect("Select rows to delete", df.index, key = session_state.id)
                    df = df.loc[~df.index.isin(sb_DM_delRows)]

                    # Delete columns
                    sb_DM_delCols = st.multiselect("Select columns to delete", df.columns, key = session_state.id)
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
                    sb_DM_dImp_num = None 
                    sb_DM_dImp_other = None
                    group_by_num = None
                    group_by_other = None
                    if sb_DM_delRows_wNA == "No" and n_rows_wNAs > 0:
                        st.markdown("**Data imputation**")
                        sb_DM_dImp_choice = st.selectbox("Replace entries with NA", ["No", "Yes"], key = session_state.id)
                        if sb_DM_dImp_choice == "Yes":
                            # Numeric variables
                            sb_DM_dImp_num = st.selectbox("Imputation method for numeric variables", ["Mean", "Median", "Random value"], key = session_state.id)
                            # Other variables
                            sb_DM_dImp_other = st.selectbox("Imputation method for other variables", ["Mode", "Random value"], key = session_state.id)
                            group_by_num = st.selectbox("Group imputation by", ["None"] + list(df.columns), key = session_state.id)
                            group_by_other = group_by_num
                            df = fc.data_impute_grouped(df, sb_DM_dImp_num, sb_DM_dImp_other, group_by_num, group_by_other)
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
                    sb_DM_dTrans_log = st.multiselect("Select columns to transform with log", transform_options, key = session_state.id)
                    if sb_DM_dTrans_log is not None: 
                        df = fc.var_transform_log(df, sb_DM_dTrans_log)
                    sb_DM_dTrans_sqrt = st.multiselect("Select columns to transform with sqrt", transform_options, key = session_state.id)
                    if sb_DM_dTrans_sqrt is not None: 
                        df = fc.var_transform_sqrt(df, sb_DM_dTrans_sqrt)
                    sb_DM_dTrans_square = st.multiselect("Select columns for squaring", transform_options, key = session_state.id)
                    if sb_DM_dTrans_square is not None: 
                        df = fc.var_transform_square(df, sb_DM_dTrans_square)
                    sb_DM_dTrans_stand = st.multiselect("Select columns for standardization", transform_options, key = session_state.id)
                    if sb_DM_dTrans_stand is not None: 
                        df = fc.var_transform_stand(df, sb_DM_dTrans_stand)
                    sb_DM_dTrans_norm = st.multiselect("Select columns for normalization", transform_options, key = session_state.id)
                    if sb_DM_dTrans_norm is not None: 
                        df = fc.var_transform_norm(df, sb_DM_dTrans_norm)
                    if df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0] == 0:
                        sb_DM_dTrans_numCat = st.multiselect("Select columns for numeric categorization", numCat_options, key = session_state.id)
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
                        sb_DM_delDup = st.selectbox("Delete duplicate rows", ["No", "Yes"], key = session_state.id)
                        if sb_DM_delDup == "Yes":
                            n_rows_dup = df[df.duplicated()].shape[0]
                            df = df.drop_duplicates()
                    elif df[df.duplicated()].shape[0] == 0:   
                        sb_DM_delDup = "No"    
                        
                    # Delete rows with NA if any exist
                    n_rows_wNAs = df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0]
                    if n_rows_wNAs > 0:
                        sb_DM_delRows_wNA = st.selectbox("Delete rows with NAs", ["No", "Yes"], key = session_state.id)
                        if sb_DM_delRows_wNA == "Yes": 
                            df = df.dropna()
                    elif n_rows_wNAs == 0: 
                        sb_DM_delRows_wNA = "No"   

                    # Delete rows
                    sb_DM_delRows = st.multiselect("Select rows to delete", df.index, key = session_state.id)
                    df = df.loc[~df.index.isin(sb_DM_delRows)]

                    # Delete columns
                    sb_DM_delCols = st.multiselect("Select columns to delete", df.columns, key = session_state.id)
                    df = df.loc[:,~df.columns.isin(sb_DM_delCols)]

                    # Filter data
                    st.markdown("**Data filtering**")
                    filter_var = st.selectbox('Filter your data by a variable...', list('-')+ list(df.columns), key = session_state.id)
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
                    sb_DM_dTrans_log = st.multiselect("Select columns to transform with log", transform_options, key = session_state.id)
                    if sb_DM_dTrans_log is not None: 
                        df = fc.var_transform_log(df, sb_DM_dTrans_log)
                    sb_DM_dTrans_sqrt = st.multiselect("Select columns to transform with sqrt", transform_options, key = session_state.id)
                    if sb_DM_dTrans_sqrt is not None: 
                        df = fc.var_transform_sqrt(df, sb_DM_dTrans_sqrt)
                    sb_DM_dTrans_square = st.multiselect("Select columns for squaring", transform_options, key = session_state.id)
                    if sb_DM_dTrans_square is not None: 
                        df = fc.var_transform_square(df, sb_DM_dTrans_square)
                    sb_DM_dTrans_stand = st.multiselect("Select columns for standardization", transform_options, key = session_state.id)
                    if sb_DM_dTrans_stand is not None: 
                        df = fc.var_transform_stand(df, sb_DM_dTrans_stand)
                    sb_DM_dTrans_norm = st.multiselect("Select columns for normalization", transform_options, key = session_state.id)
                    if sb_DM_dTrans_norm is not None: 
                        df = fc.var_transform_norm(df, sb_DM_dTrans_norm)
                    if df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0] == 0:
                        sb_DM_dTrans_numCat = st.multiselect("Select columns for numeric categorization", numCat_options, key = session_state.id)
                        if sb_DM_dTrans_numCat is not None: 
                            df = fc.var_transform_numCat(df, sb_DM_dTrans_numCat)
                    else:
                        sb_DM_dTrans_numCat = None

            #--------------------------------------------------------------------------------------
            # PROCESSING SUMMARY

            if st.checkbox('Show a summary of my data processing preferences', value = False, key = session_state.id): 
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
                    st.write("- Imputation grouped by:", group_by_num)

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
        if  any(v for v in [sb_DM_delRows, sb_DM_delCols, sb_DM_dImp_num, sb_DM_dImp_other, sb_DM_dTrans_log, sb_DM_dTrans_sqrt, sb_DM_dTrans_square, sb_DM_dTrans_stand, sb_DM_dTrans_norm, sb_DM_dTrans_numCat ] if v is not None) or sb_DM_delDup == "Yes" or sb_DM_delRows_wNA == "Yes":
            dev_expander_dsPost = st.beta_expander("Explore cleaned and transformed panel data", expanded = False)
            with dev_expander_dsPost:
                if df.shape[1] > 2 and df.shape[0] > 0:

                    # Show cleaned and transformed data & data info
                    df_summary_post = fc.data_summary(df)
                    if st.checkbox("Show cleaned and transformed data", value = False):  
                        n_rows_post = df.shape[0]
                        n_cols_post = df.shape[1]
                        st.dataframe(df)
                        st.write("Data shape: ", n_rows_post, "rows and ", n_cols_post, "columns")
                    if df[df.duplicated()].shape[0] > 0 or df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0] > 0:
                        check_nasAnddupl2 = st.checkbox("Show duplicates and NAs info (processed)", value = False) 
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
                    if st.checkbox("Show cleaned and transformed variable info", value = False): 
                        st.write(df_summary_post["Variable types"])

                    # Show summary statistics (cleaned and transformed data)
                    if st.checkbox('Show summary statistics (cleaned and transformed data)', value = False):
                        st.write(df_summary_post["ALL"])
                        if fc.get_mode(df).loc["n_unique"].any():
                            st.caption("** Mode is not unique.") 
                else: st.error("ERROR: No data available for Data Exploration!") 

    #--------------------------------------------------
    #--------------------------------------------------
    # Geodata processing
    #---------------------------------------------------
    
    data_geodata_processing_container = st.beta_container()
    with data_geodata_processing_container:
        #initialisation:
        anim_show=False
        #check what variables are numerical ones
        st.write("")
        st.write("")
        st.header('**Geospatial data visualisation**')
        df=df.dropna() # just to make sure that NAs are removed in case the user hasn't done it before
        num_cols=df.columns  # no of numerical columns in df
        objcat_cols=df.columns
        for column in df:              
            if not df[column].dtypes in ('float', 'float64', 'int','int64','datetime64'): 
                num_cols=num_cols.drop(column) 
            elif df[column].dtypes !='object':
                objcat_cols=objcat_cols.drop(column) 

        if len(num_cols)==0 or len(objcat_cols)==0:
            st.error("ERROR: Your dataset is not suitable for the geospatial analysis!")  
        else:
            map_code=[]
            a4,a5=st.beta_columns(2)
            with a4:
                map_code=st.selectbox('What kind of country info do you have?',['country name','country code'], key = session_state.id)
                map_loc=st.selectbox('Select the data column with the country info',objcat_cols, key = session_state.id)
                
            with a5:
                if not 'Ladder' in list(df.columns):
                    map_var=st.selectbox('Select the variable to plot',num_cols, key = session_state.id)
                else:
                    map_var=st.selectbox('Select the variable to plot',num_cols, index=1, key = session_state.id) 
                map_time_filter=st.selectbox('Select time variable (if avaiable)',list('-')+ list(num_cols) + list(objcat_cols), key = session_state.id)
                    
            if map_time_filter !='-':
                anim_show=st.checkbox('Show animation of temporal development?',value=False, key = session_state.id)

            miss_val_show=st.checkbox('Show contours of countries with missing values?', value =False, key = session_state.id)
            #set mapping key for the geojson data:
            if map_code=='country name':
                fid_key='properties.name'
            else:
                fid_key='properties.adm0_a3'

            # read geojson ne_110m_admin_0_countries.geojson: 110m resolution geojson from http://geojson.xyz/ based on Natural Earth Data            
            
            geojson_file =("/default data/ne_110m_admin_0_countries.geojson")
            geojson_handle = open(geojson_file, )
            geojson = json.load(geojson_handle)
            geojson_handle.close()
            # fine resolution json
            #with urlopen('https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson') as response:
                #geojson = json.load(response)

            #animated chart
            if anim_show:
                i = 0
                st.markdown(' --- ')
                ani_text = st.empty()
                ani_map = st.empty()
                time_range=sorted(pd.Series(df[map_time_filter]).unique())
                
                for k in range(len(time_range)-1):
                    map_time_step=time_range[k]
                    ani_text.subheader(str(map_time_step))
                    anim_filtered_data = df[df[map_time_filter]==map_time_step]
                    fig = px.choropleth(anim_filtered_data, geojson=geojson, locations=map_loc, featureidkey=fid_key, color=map_var,
                                        color_continuous_scale="rdbu_r",
                                        range_color=(min(df[map_var]), max(df[map_var])),
                                        scope="world",
                                        labels={'Ladder':'Ladder'}
                                        )
                    if miss_val_show:
                        fig.update_geos(visible=False, resolution=50,showcountries=True, countrycolor="lightgrey")
                    else:
                        fig.update_geos(visible=False)
                    fig.update_geos( center=dict(lon=0, lat=0),lataxis_range=[-90,90], lonaxis_range=[-180, 180])
                    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                    ani_map.plotly_chart(fig, use_container_width=True)
                    time.sleep(0.5)




            # place time-bar out of the beta_column:        
            if map_time_filter !='-':
                try:
                    if int(df[map_time_filter][0])<3000:
                        time_to_filter = st.slider('time', min(df[map_time_filter]), max(df[map_time_filter]),min(df[map_time_filter]))
                except ValueError:                    
                    time_to_filter =st.selectbox('Specify the time',list(df[map_time_filter]), key = session_state.id)
                
                filtered_data = df[df[map_time_filter]==time_to_filter]
            else:
                filtered_data=df
                
                    
            #choropleth map:
            fig = px.choropleth(filtered_data, geojson=geojson, locations=map_loc, featureidkey=fid_key, color=map_var,
                                color_continuous_scale="rdbu_r",
                                range_color=(min(df[map_var]), max(df[map_var])),
                                scope="world",
                                labels={'Ladder':'Ladder'}
                                )
            if miss_val_show:
                fig.update_geos(visible=False, resolution=50,showcountries=True, countrycolor="lightgrey")
            else:
                fig.update_geos(visible=False)
            fig.update_geos( center=dict(lon=0, lat=0),lataxis_range=[-90,90], lonaxis_range=[-180, 180])
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)

            # identify the unique vals of map entities
            map_entities=(df[map_loc]).unique()
            
            if map_time_filter !='-':
                
                # compare countries over time
                mp_comp=st.multiselect('Select entities for a comparison', list(map_entities),[map_entities[0]])
                df_sel=df[df[map_loc].isin(mp_comp)]
                fig = px.line(df_sel, x=map_time_filter, y=map_var, color=map_loc,color_discrete_sequence=px.colors.qualitative.Pastel2)
                fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',})  
                fig.update_layout(yaxis=dict(title=map_var, titlefont_size=12, tickfont_size=14,),)
                fig.update_layout(xaxis=dict(title=map_time_filter, titlefont_size=12, tickfont_size=14,),)
                fig.update_yaxes(range=[min(df[map_var]), max(df[map_var])])
                st.plotly_chart(fig, use_container_width=True)
            
            a4,a5=st.beta_columns(2)
            with a4:
                #plot relative frequency
                all_list=list(map_entities)
                all_list[1:len(all_list)-1]=all_list
                all_list[0]= 'all data'
                        
                rf_map_var = st.selectbox('Draw relative frequency for...?',all_list, key = session_state.id)  
                            
                fig, ax = plt.subplots()  
                if rf_map_var=='all data':                     
                    n, bins, patches=ax.hist(df[map_var], bins=10, rwidth=0.95, density=False)
                else:
                    rf_filtered_data=df[df[map_loc]==rf_map_var]  
                    n, bins, patches=ax.hist(rf_filtered_data[map_var], bins=10, rwidth=0.95, density=False)    
                
                class_mean=(bins[0:(len(bins)-1)]+bins[1:(len(bins))])/2
                fa_val=n/sum(n) #relative frequency
                
                fig = go.Figure()            
                fig.add_trace(go.Bar(x=class_mean, y=fa_val, name='Relative frequency',marker_color = 'indianred',opacity=0.5))
                fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',})  
                fig.update_layout(yaxis=dict(title='Relative frequency', titlefont_size=12, tickfont_size=14,),)
                fig.update_layout(xaxis=dict(title=map_var, titlefont_size=12, tickfont_size=14,),)
                #fig.update_yaxes(range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

            with a5:
                #boxlot               
                bx_map_var = st.selectbox('Draw a boxplot for ' + map_var + ' for...?', map_entities, key = session_state.id)  
                bx_filtered_data=df[df[map_loc]==bx_map_var] 
                
                fig = go.Figure()
                fig.add_trace(go.Box(
                        y=bx_filtered_data[map_var],
                        name=str(bx_map_var),boxpoints='all',jitter=0.2, whiskerwidth=0.2,
                        marker_color = 'indianred',marker_size=2, line_width=1)
                    )
                fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',})  
                fig.update_yaxes(range=[min(df[map_var]), max(df[map_var])])
                st.plotly_chart(fig, use_container_width=True)          
        

