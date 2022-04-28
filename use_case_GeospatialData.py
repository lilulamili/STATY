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
import base64
from io import BytesIO


def app():

    # Clear cache
    st.legacy_caching.clear_cache()

    # Hide traceback in error messages (comment out for de-bugging)
    sys.tracebacklimit = 0

    # workaround for Firefox bug- hide the scrollbar while keeping the scrolling functionality
    st.markdown("""
        <style>
        .ReactVirtualized__Grid::-webkit-scrollbar {
        display: none;
        }

        .ReactVirtualized__Grid {
        -ms-overflow-style: none;  /* IE and Edge */
        scrollbar-width: none;  /* Firefox */
        }
        </style>
        """, unsafe_allow_html=True)
    #++++++++++++++++++++++++++++++++++++++++++++
    # RESET INPUT
    
    #Session state
    if 'key' not in st.session_state:
        st.session_state['key'] = 0
    reset_clicked = st.sidebar.button("Reset all your input")
    if reset_clicked:
        st.session_state['key'] = st.session_state['key'] + 1
    st.sidebar.markdown("")
   
    # File upload section
    df_dec = st.sidebar.radio("Get data", ["Use example dataset", "Upload data"], key = st.session_state['key'])
    uploaded_data=None
    if df_dec == "Upload data":
        separator_expander=st.sidebar.expander('Upload settings')
        with separator_expander:                      
            a4,a5=st.columns(2)
            with a4:
                dec_sep=a4.selectbox("Decimal sep.",['.',','], key = st.session_state['key'])
            with a5:
                col_sep=a5.selectbox("Column sep.",[';',  ','  , '|', '\s+','\t','other'], key = st.session_state['key'])
                if col_sep=='other':
                    col_sep=st.text_input('Specify your column separator', key = st.session_state['key'])     

            a4,a5=st.columns(2)  
            with a4:    
                thousands_sep=a4.selectbox("Thousands x sep.",[None,'.', ' ','\s+', 'other'], key = st.session_state['key'])
                if thousands_sep=='other':
                    thousands_sep=st.text_input('Specify your thousands separator', key = st.session_state['key'])  
            with a5:    
                encoding_val=a5.selectbox("Encoding",[None,'utf_8','utf_8_sig','utf_16_le','cp1140','cp1250','cp1251','cp1252','cp1253','cp1254','other'], key = st.session_state['key'])
                if encoding_val=='other':
                    encoding_val=st.text_input('Specify your encoding', key = st.session_state['key'])  
        
        # Error handling for separator selection:
        if dec_sep==col_sep: 
            st.sidebar.error("Decimal and column separators cannot be identical!") 
        elif dec_sep==thousands_sep:
            st.sidebar.error("Decimal and thousands separators cannot be identical!") 
        elif  col_sep==thousands_sep:
            st.sidebar.error("Column and thousands separators cannot be identical!")    
        
        uploaded_data = st.sidebar.file_uploader("Default separators: decimal '.'    |     column  ';'", type=["csv", "txt"])
        
        if uploaded_data is not None:
            df = pd.read_csv(uploaded_data, decimal=dec_sep, sep = col_sep,thousands=thousands_sep,encoding=encoding_val, engine='python')
            df_name=os.path.splitext(uploaded_data.name)[0]
            st.sidebar.success('Loading data... done!')
            if len(df)<4:
                small_dataset_error ="The sample is so small that you should better use a pocket calculator as the learning effect will be larger!"
        elif uploaded_data is None:            
            df = pd.read_csv("default data/WHR_2021.csv", sep = ";|,|\t",engine='python')
            df_name="WHR_2021"
    else:        
        df = pd.read_csv("default data/WHR_2021.csv", sep = ";|,|\t",engine='python')
        df_name="WHR_2021"
    st.sidebar.markdown("")

    #Basic data info      
    n_rows = df.shape[0]
    n_cols = df.shape[1] 
    
    #------------------------------------------------------------------------------------------
    # SETTINGS

    settings_expander=st.sidebar.expander('Settings')
    with settings_expander:
        st.caption("**Help**")
        sett_hints = st.checkbox('Show learning hints', value=False, key = st.session_state['key'])
        st.caption("**Appearance**")
        sett_wide_mode = st.checkbox('Wide mode', value=False, key = st.session_state['key'])
        sett_theme = st.selectbox('Theme', ["Light", "Dark"], key = st.session_state['key'])
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
    fc.theme_func_dl_button()

    #------------------------------------------------------------------------------------------

    #++++++++++++++++++++++++++++++++++++++++++++
    # DATA PREPROCESSING & VISUALISATION

    st.header("**Geospatial data/Interactive dashboards**")
    st.markdown("Let STATY do the data cleaning, variable transformations, visualisations and deliver you the stats you need. Specify your data processing preferences and start exploring your data stories right below... ")

    if len(df)<4:
        st.error(small_dataset_error)
        return

    st.header("**Data screening and processing**")
    #------------------------------------------------------------------------------------------

    #++++++++++++++++++++++
    # DATA SUMMARY

    # Main panel for data summary (pre)
    #----------------------------------
    data_exploration_container = st.container()
    with data_exploration_container:
        dev_expander_raw = st.expander("Explore raw data info and stats")
        with dev_expander_raw:
            
            # Default data description:
            if uploaded_data == None:
                if st.checkbox("Show data description", value = False, key = st.session_state['key']):          
                    st.markdown("**Data source:**")
                    st.markdown("The data come from the Gallup World Poll surveys from 2018 to 2020. For more details see the [World Happiness Report 2021] (https://worldhappiness.report/).")
                    st.markdown("**Citation:**")
                    st.markdown("Helliwell, John F., Richard Layard, Jeffrey Sachs, and Jan-Emmanuel De Neve, eds. 2021. World Happiness Report 2021. New York: Sustainable Development Solutions Network.")
                    st.markdown("**Variables in the dataset:**")

                    col1,col2=st.columns(2) 
                    col1.write("Country")
                    col2.write("country name")
                    
                    col1,col2=st.columns(2)
                    col1.write("Year ")
                    col2.write("year ranging from 2005 to 2020")
                    
                    col1,col2=st.columns(2) 
                    col1.write("Ladder")
                    col2.write("happiness  score  or  subjective  well-being with the best possible life being a 10, and the worst possible life being a 0")
                    
                    col1,col2=st.columns(2) 
                    col1.write("Log GDP per capita")
                    col2.write("in purchasing power parity at  constant  2017  international  dollar  prices")
                    
                    col1,col2=st.columns(2) 
                    col1.write("Social support")
                    col2.write("the national average of the binary responses (either 0 or 1) to the question regarding relatives or friends to count on")
                    
                    col1,col2=st.columns(2) 
                    col1.write("Healthy life expectancy at birth")
                    col2.write("based on  the  data  extracted  from  the  World  Health  Organization’s  Global Health Observatory data repository")
                   
                    col1,col2=st.columns(2) 
                    col1.write("Freedom to make life choices")
                    col2.write("national average of responses to the corresponding question")

                    col1,col2=st.columns(2) 
                    col1.write("Generosity")
                    col2.write("residual of regressing national average of response to the question rerading money donations in the past month on GDPper capita")

                    col1,col2=st.columns(2) 
                    col1.write("Perceptions of corruption")
                    col2.write("the national average of the survey responses to the coresponding question")
                    
                    col1,col2=st.columns(2) 
                    col1.write("Positive affect")
                    col2.write("the  average  of  three  positive  affect  measures (happiness,  laugh  and  enjoyment)")
                    
                    col1,col2=st.columns(2)
                    col1.write("Negative affectt (worry, sadness and anger)")
                    col2.write("the  average  of  three  negative  affect  measures  (worry, sadness and anger)")

                    st.markdown("")
            # Show raw data & data info
            df_summary = fc.data_summary(df) 
            if st.checkbox("Show raw data", value = False, key = st.session_state['key']):      
                #st.dataframe(df.style.apply(lambda x: ["background-color: #ffe5e5" if (not pd.isna(df_summary_mq_full.loc["1%-Q"][i]) and df_summary_vt_cat[i] == "numeric" and (v <= df_summary_mq_full.loc["1%-Q"][i] or v >= df_summary_mq_full.loc["99%-Q"][i]) or pd.isna(v)) else "" for i, v in enumerate(x)], axis = 1))
                st.write(df)
                st.write("Data shape: ", n_rows,  " rows and ", n_cols, " columns")
                #st.info("** Note that NAs and numerical values below/ above the 1%/ 99% quantile are highlighted.") 
            if df[df.duplicated()].shape[0] > 0 or df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0] > 0:
                check_nasAnddupl=st.checkbox("Show duplicates and NAs info", value = False, key = st.session_state['key']) 
                if check_nasAnddupl:      
                    if df[df.duplicated()].shape[0] > 0:
                        st.write("Number of duplicates: ", df[df.duplicated()].shape[0])
                        st.write("Duplicate row index: ", ', '.join(map(str,list(df.index[df.duplicated()]))))
                    if df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0] > 0:
                        st.write("Number of rows with NAs: ", df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0])
                        st.write("Rows with NAs: ", ', '.join(map(str,list(pd.unique(np.where(df.isnull())[0])))))
                
            # Show variable info 
            if st.checkbox('Show variable info', value = False, key = st.session_state['key']): 
                #st.write(df_summary["Variable types"])
                a7, a8 = st.columns(2)
                with a7:
                    st.table(df_summary["Variable types"])
            # Show summary statistics (raw data)
            if st.checkbox('Show summary statistics (raw data)', value = False, key = st.session_state['key']): 
                #st.write(df_summary["ALL"])
                df_datasumstat=df_summary["ALL"]
                #dfStyler = df_datasumstat.style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector = 'th', props=[('text-align', 'left')])]) 
                a7, a8 = st.columns(2)
                with a7:
                    st.table(df_datasumstat)
                    if fc.get_mode(df).loc["n_unique"].any():
                        st.caption("** Mode is not unique.")

                # Download link for summary statistics
                output = BytesIO()
                excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                df_summary["Variable types"].to_excel(excel_file, sheet_name="variable_info")
                df_summary["ALL"].to_excel(excel_file, sheet_name="summary_statistics")
                excel_file.save()
                excel_file = output.getvalue()
                b64 = base64.b64encode(excel_file)
                dl_file_name = "Summary statistics__" + df_name + ".xlsx"
                st.markdown(
                    f"""
                <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download summary statistics</a>
                """,
                unsafe_allow_html=True)
                st.write("")


        #++++++++++++++++++++++
        # DATA PROCESSING

        # Settings for data processing
        #-------------------------------------
        
        dev_expander_dm_sb = st.expander("Specify data processing preferences", expanded = False)
        with dev_expander_dm_sb:
            
            n_rows_wNAs = df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0]
            n_rows_wNAs_pre_processing = "No"
            if n_rows_wNAs > 0:
                n_rows_wNAs_pre_processing = "Yes"
                a1, a2, a3 = st.columns(3)
            else: a1, a3 = st.columns(2)
            
            sb_DM_dImp_num = None 
            sb_DM_dImp_other = None
            sb_DM_delRows=None
            sb_DM_keepRows=None
            group_by_num = None
            group_by_other = None
            
            with a1:
                #--------------------------------------------------------------------------------------
                # DATA CLEANING

                st.markdown("**Data cleaning**")

                # Delete rows
                delRows =st.selectbox('Delete rows with index ...', options=['-', 'greater', 'greater or equal', 'smaller', 'smaller or equal', 'equal', 'between'], key = st.session_state['key'])
                if delRows!='-':                                
                    if delRows=='between':
                        row_1=st.number_input('Lower limit is', value=0, step=1, min_value= 0, max_value=len(df)-1, key = st.session_state['key'])
                        row_2=st.number_input('Upper limit is', value=2, step=1, min_value= 0, max_value=len(df)-1, key = st.session_state['key'])
                        if (row_1 + 1) < row_2 :
                            sb_DM_delRows=df.index[(df.index > row_1) & (df.index < row_2)]
                        elif (row_1 + 1) == row_2 : 
                            st.warning("WARNING: No row is deleted!")
                        elif row_1 == row_2 : 
                            st.warning("WARNING: No row is deleted!")
                        elif row_1 > row_2 :
                            st.error("ERROR: Lower limit must be smaller than upper limit!")  
                            return                   
                    elif delRows=='equal':
                        sb_DM_delRows = st.multiselect("to...", df.index, key = st.session_state['key'])
                    else:
                        row_1=st.number_input('than...', step=1, value=1, min_value = 0, max_value=len(df)-1, key = st.session_state['key'])                    
                        if delRows=='greater':
                            sb_DM_delRows=df.index[df.index > row_1]
                            if row_1 == len(df)-1:
                                st.warning("WARNING: No row is deleted!") 
                        elif delRows=='greater or equal':
                            sb_DM_delRows=df.index[df.index >= row_1]
                            if row_1 == 0:
                                st.error("ERROR: All rows are deleted!")
                                return
                        elif delRows=='smaller':
                            sb_DM_delRows=df.index[df.index < row_1]
                            if row_1 == 0:
                                st.warning("WARNING: No row is deleted!") 
                        elif delRows=='smaller or equal':
                            sb_DM_delRows=df.index[df.index <= row_1]
                            if row_1 == len(df)-1:
                                st.error("ERROR: All rows are deleted!")
                                return
                    if sb_DM_delRows is not None:
                        df = df.loc[~df.index.isin(sb_DM_delRows)]
                        no_delRows=n_rows-df.shape[0]

                # Keep rows
                keepRows =st.selectbox('Keep rows with index ...', options=['-', 'greater', 'greater or equal', 'smaller', 'smaller or equal', 'equal', 'between'], key = st.session_state['key'])
                if keepRows!='-':                                
                    if keepRows=='between':
                        row_1=st.number_input('Lower limit is', value=0, step=1, min_value= 0, max_value=len(df)-1, key = st.session_state['key'])
                        row_2=st.number_input('Upper limit is', value=2, step=1, min_value= 0, max_value=len(df)-1, key = st.session_state['key'])
                        if (row_1 + 1) < row_2 :
                            sb_DM_keepRows=df.index[(df.index > row_1) & (df.index < row_2)]
                        elif (row_1 + 1) == row_2 : 
                            st.error("ERROR: No row is kept!")
                            return
                        elif row_1 == row_2 : 
                            st.error("ERROR: No row is kept!")
                            return
                        elif row_1 > row_2 :
                            st.error("ERROR: Lower limit must be smaller than upper limit!")  
                            return                   
                    elif keepRows=='equal':
                        sb_DM_keepRows = st.multiselect("to...", df.index, key = st.session_state['key'])
                    else:
                        row_1=st.number_input('than...', step=1, value=1, min_value = 0, max_value=len(df)-1, key = st.session_state['key'])                    
                        if keepRows=='greater':
                            sb_DM_keepRows=df.index[df.index > row_1]
                            if row_1 == len(df)-1:
                                st.error("ERROR: No row is kept!") 
                                return
                        elif keepRows=='greater or equal':
                            sb_DM_keepRows=df.index[df.index >= row_1]
                            if row_1 == 0:
                                st.warning("WARNING: All rows are kept!")
                        elif keepRows=='smaller':
                            sb_DM_keepRows=df.index[df.index < row_1]
                            if row_1 == 0:
                                st.error("ERROR: No row is kept!") 
                                return
                        elif keepRows=='smaller or equal':
                            sb_DM_keepRows=df.index[df.index <= row_1]
                    if sb_DM_keepRows is not None:
                        df = df.loc[df.index.isin(sb_DM_keepRows)]
                        no_keptRows=df.shape[0]

                # Delete columns
                sb_DM_delCols = st.multiselect("Select columns to delete", df.columns, key = st.session_state['key'])
                df = df.loc[:,~df.columns.isin(sb_DM_delCols)]

                # Keep columns
                sb_DM_keepCols = st.multiselect("Select columns to keep", df.columns, key = st.session_state['key'])
                if len(sb_DM_keepCols) > 0:
                    df = df.loc[:,df.columns.isin(sb_DM_keepCols)]

                # Delete duplicates if any exist
                if df[df.duplicated()].shape[0] > 0:
                    sb_DM_delDup = st.selectbox("Delete duplicate rows", ["No", "Yes"], key = st.session_state['key'])
                    if sb_DM_delDup == "Yes":
                        n_rows_dup = df[df.duplicated()].shape[0]
                        df = df.drop_duplicates()
                elif df[df.duplicated()].shape[0] == 0:   
                    sb_DM_delDup = "No"    
                    
                # Delete rows with NA if any exist
                n_rows_wNAs = df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0]
                if n_rows_wNAs > 0:
                    sb_DM_delRows_wNA = st.selectbox("Delete rows with NAs", ["No", "Yes"], key = st.session_state['key'])
                    if sb_DM_delRows_wNA == "Yes": 
                        df = df.dropna()
                elif n_rows_wNAs == 0: 
                    sb_DM_delRows_wNA = "No"   

                # Filter data
                st.markdown("**Data filtering**")
                filter_var = st.selectbox('Filter your data by a variable...', list('-')+ list(df.columns), key = st.session_state['key'])
                if filter_var !='-':
                    
                    if df[filter_var].dtypes=="int64" or df[filter_var].dtypes=="float64": 
                        if df[filter_var].dtypes=="float64":
                            filter_format="%.8f"
                        else:
                            filter_format=None

                        user_filter=st.selectbox('Select values that are ...', options=['greater','greater or equal','smaller','smaller or equal', 'equal','between'], key = st.session_state['key'])
                                                
                        if user_filter=='between':
                            filter_1=st.number_input('Lower limit is', format=filter_format, value=df[filter_var].min(), min_value=df[filter_var].min(), max_value=df[filter_var].max(), key = st.session_state['key'])
                            filter_2=st.number_input('Upper limit is', format=filter_format, value=df[filter_var].max(), min_value=df[filter_var].min(), max_value=df[filter_var].max(), key = st.session_state['key'])
                            #reclassify values:
                            if filter_1 < filter_2 :
                                df = df[(df[filter_var] > filter_1) & (df[filter_var] < filter_2)] 
                                if len(df) == 0:
                                   st.error("ERROR: No data available for the selected limits!")  
                                   return        
                            elif filter_1 >= filter_2 :
                                st.error("ERROR: Lower limit must be smaller than upper limit!")  
                                return                    
                        elif user_filter=='equal':                            
                            filter_1=st.multiselect('to... ', options=df[filter_var].values, key = st.session_state['key'])
                            if len(filter_1)>0:
                                df = df.loc[df[filter_var].isin(filter_1)]

                        else:
                            filter_1=st.number_input('than... ',format=filter_format, value=df[filter_var].min(), min_value=df[filter_var].min(), max_value=df[filter_var].max(), key = st.session_state['key'])
                            #reclassify values:
                            if user_filter=='greater':
                                df = df[df[filter_var] > filter_1]
                            elif user_filter=='greater or equal':
                                df = df[df[filter_var] >= filter_1]        
                            elif user_filter=='smaller':
                                df= df[df[filter_var]< filter_1] 
                            elif user_filter=='smaller or equal':
                                df = df[df[filter_var] <= filter_1]
                
                            if len(df) == 0:
                                st.error("ERROR: No data available for the selected value!")
                                return 
                            elif len(df) == n_rows:
                                st.warning("WARNING: Data are not filtered for this value!")         
                    else:                  
                        filter_1=st.multiselect('Filter your data by a value...', (df[filter_var]).unique(), key = st.session_state['key'])
                        if len(filter_1)>0:
                            df = df.loc[df[filter_var].isin(filter_1)]

            if n_rows_wNAs_pre_processing == "Yes":
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
                        sb_DM_dImp_choice = st.selectbox("Replace entries with NA", ["No", "Yes"], key = st.session_state['key'])
                        if sb_DM_dImp_choice == "Yes":
                            # Numeric variables
                            sb_DM_dImp_num = st.selectbox("Imputation method for numeric variables", ["Mean", "Median", "Random value"], key = st.session_state['key'])
                            # Other variables
                            sb_DM_dImp_other = st.selectbox("Imputation method for other variables", ["Mode", "Random value"], key = st.session_state['key'])
                            group_by_num = st.selectbox("Group imputation by", ["None"] + list(df.columns), key = st.session_state['key'])
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
                sb_DM_dTrans_log = st.multiselect("Select columns to transform with log", transform_options, key = st.session_state['key'])
                if sb_DM_dTrans_log is not None: 
                    df = fc.var_transform_log(df, sb_DM_dTrans_log)
                sb_DM_dTrans_sqrt = st.multiselect("Select columns to transform with sqrt", transform_options, key = st.session_state['key'])
                if sb_DM_dTrans_sqrt is not None: 
                    df = fc.var_transform_sqrt(df, sb_DM_dTrans_sqrt)
                sb_DM_dTrans_square = st.multiselect("Select columns for squaring", transform_options, key = st.session_state['key'])
                if sb_DM_dTrans_square is not None: 
                    df = fc.var_transform_square(df, sb_DM_dTrans_square)
                sb_DM_dTrans_cent = st.multiselect("Select columns for centering ", transform_options, key = st.session_state['key'])
                if sb_DM_dTrans_cent is not None: 
                    df = fc.var_transform_cent(df, sb_DM_dTrans_cent)
                sb_DM_dTrans_stand = st.multiselect("Select columns for standardization", transform_options, key = st.session_state['key'])
                if sb_DM_dTrans_stand is not None: 
                    df = fc.var_transform_stand(df, sb_DM_dTrans_stand)
                sb_DM_dTrans_norm = st.multiselect("Select columns for normalization", transform_options, key = st.session_state['key'])
                if sb_DM_dTrans_norm is not None: 
                    df = fc.var_transform_norm(df, sb_DM_dTrans_norm)
                sb_DM_dTrans_numCat = st.multiselect("Select columns for numeric categorization ", numCat_options, key = st.session_state['key'])
                if sb_DM_dTrans_numCat:
                    if not df[sb_DM_dTrans_numCat].columns[df[sb_DM_dTrans_numCat].isna().any()].tolist(): 
                        sb_DM_dTrans_numCat_sel = st.multiselect("Select variables for manual categorization ", sb_DM_dTrans_numCat, key = st.session_state['key'])
                        if sb_DM_dTrans_numCat_sel:
                            for var in sb_DM_dTrans_numCat_sel:
                                if df[var].unique().size > 5: 
                                    st.error("ERROR: Selected variable has too many categories (>5): " + str(var))
                                    return
                                else:
                                    manual_cats = pd.DataFrame(index = range(0, df[var].unique().size), columns=["Value", "Cat"])
                                    text = "Category for "
                                    # Save manually selected categories
                                    for i in range(0, df[var].unique().size):
                                        text1 = text + str(var) + ": " + str(sorted(df[var].unique())[i])
                                        man_cat = st.number_input(text1, value = 0, min_value=0, key = st.session_state['key'])
                                        manual_cats.loc[i]["Value"] = sorted(df[var].unique())[i]
                                        manual_cats.loc[i]["Cat"] = man_cat
                                    
                                    new_var_name = "numCat_" + var
                                    new_var = pd.DataFrame(index = df.index, columns = [new_var_name])
                                    for c in df[var].index:
                                        if pd.isnull(df[var][c]) == True:
                                            new_var.loc[c, new_var_name] = np.nan
                                        elif pd.isnull(df[var][c]) == False:
                                            new_var.loc[c, new_var_name] = int(manual_cats[manual_cats["Value"] == df[var][c]]["Cat"])
                                    df[new_var_name] = new_var.astype('int64')
                                # Exclude columns with manual categorization from standard categorization
                                numCat_wo_manCat = [var for var in sb_DM_dTrans_numCat if var not in sb_DM_dTrans_numCat_sel]
                                df = fc.var_transform_numCat(df, numCat_wo_manCat)
                        else:
                            df = fc.var_transform_numCat(df, sb_DM_dTrans_numCat)
                    else:
                        col_with_na = df[sb_DM_dTrans_numCat].columns[df[sb_DM_dTrans_numCat].isna().any()].tolist()
                        st.error("ERROR: Please select columns without NAs: " + ', '.join(map(str,col_with_na)))
                        return
                else:
                    sb_DM_dTrans_numCat = None
                sb_DM_dTrans_mult = st.number_input("Number of variable multiplications ", value = 0, min_value=0, key = st.session_state['key'])
                if sb_DM_dTrans_mult != 0: 
                    multiplication_pairs = pd.DataFrame(index = range(0, sb_DM_dTrans_mult), columns=["Var1", "Var2"])
                    text = "Multiplication pair"
                    for i in range(0, sb_DM_dTrans_mult):
                        text1 = text + " " + str(i+1)
                        text2 = text + " " + str(i+1) + " "
                        mult_var1 = st.selectbox(text1, transform_options, key = st.session_state['key'])
                        mult_var2 = st.selectbox(text2, transform_options, key = st.session_state['key'])
                        multiplication_pairs.loc[i]["Var1"] = mult_var1
                        multiplication_pairs.loc[i]["Var2"] = mult_var2
                        fc.var_transform_mult(df, mult_var1, mult_var2)
                sb_DM_dTrans_div = st.number_input("Number of variable divisions ", value = 0, key = st.session_state['key'])
                if sb_DM_dTrans_div != 0:
                    division_pairs = pd.DataFrame(index = range(0, sb_DM_dTrans_div), columns=["Var1", "Var2"]) 
                    text = "Division pair"
                    for i in range(0, sb_DM_dTrans_div):
                        text1 = text + " " + str(i+1) + " (numerator)"
                        text2 = text + " " + str(i+1) + " (denominator)"
                        div_var1 = st.selectbox(text1, transform_options, key = st.session_state['key'])
                        div_var2 = st.selectbox(text2, transform_options, key = st.session_state['key'])
                        division_pairs.loc[i]["Var1"] = div_var1
                        division_pairs.loc[i]["Var2"] = div_var2
                        fc.var_transform_div(df, div_var1, div_var2)

                data_transfrom=st.checkbox("Transfrom data in Excel?", value=False)
                if data_transfrom==True:
                    st.info("Press the button to open your data in Excel. Don't forget to save your result as a csv or a txt file!")
                    # Download link
                    output = BytesIO()
                    excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                    df.to_excel(excel_file, sheet_name="data",index=False)    
                    excel_file.save()
                    excel_file = output.getvalue()
                    b64 = base64.b64encode(excel_file)
                    dl_file_name = "Data_transformation__" + df_name + ".xlsx"
                    st.markdown(
                        f"""
                    <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Transfrom your data in Excel</a>
                    """,
                    unsafe_allow_html=True)
                st.write("")  

            #--------------------------------------------------------------------------------------
            # PROCESSING SUMMARY

            if st.checkbox('Show a summary of my data processing preferences', value = False, key = st.session_state['key']): 
                st.markdown("Summary of data changes:")
                
                #--------------------------------------------------------------------------------------
                # DATA CLEANING

                # Rows
                if sb_DM_delRows is not None and delRows!='-' :
                    if no_delRows > 1:
                        st.write("-", no_delRows, " rows were deleted!")
                    elif no_delRows == 1:
                        st.write("-",no_delRows, " row was deleted!")
                    elif no_delRows == 0:
                        st.write("- No row was deleted!")
                else:
                    st.write("- No row was deleted!")
                if sb_DM_keepRows is not None and keepRows!='-' :
                    if no_keptRows > 1:
                        st.write("-", no_keptRows, " rows are kept!")
                    elif no_keptRows == 1:
                        st.write("-",no_keptRows, " row is kept!")
                    elif no_keptRows == 0:
                        st.write("- All rows are kept!")
                else:
                    st.write("- All rows are kept!") 
                # Columns
                if len(sb_DM_delCols) > 1:
                    st.write("-", len(sb_DM_delCols), " columns were manually deleted:", ', '.join(sb_DM_delCols))
                elif len(sb_DM_delCols) == 1:
                    st.write("-",len(sb_DM_delCols), " column was manually deleted:", str(sb_DM_delCols[0]))
                elif len(sb_DM_delCols) == 0:
                    st.write("- No column was manually deleted!")
                if len(sb_DM_keepCols) > 1:
                    st.write("-", len(sb_DM_keepCols), " columns are kept:", ', '.join(sb_DM_keepCols))
                elif len(sb_DM_keepCols) == 1:
                    st.write("-",len(sb_DM_keepCols), " column is kept:", str(sb_DM_keepCols[0]))
                elif len(sb_DM_keepCols) == 0:
                    st.write("- All columns are kept!")
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
                # Filter
                if filter_var != "-":
                    if df[filter_var].dtypes=="int64" or df[filter_var].dtypes=="float64":
                        if isinstance(filter_1, list):
                            if len(filter_1) == 0:
                                st.write("-", " Data was not filtered!")
                            elif len(filter_1) > 0:
                                st.write("-", " Data filtered by:", str(filter_var))
                        elif filter_1 is not None:
                            st.write("-", " Data filtered by:", str(filter_var))
                        else:
                            st.write("-", " Data was not filtered!")
                    elif len(filter_1)>0:
                        st.write("-", " Data filtered by:", str(filter_var))
                    elif len(filter_1) == 0:
                        st.write("-", " Data was not filtered!")
                else:
                    st.write("-", " Data was not filtered!")
                
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
                # centering
                if len(sb_DM_dTrans_cent) > 1:
                    st.write("-", len(sb_DM_dTrans_cent), " columns were centered:", ', '.join(sb_DM_dTrans_cent))
                elif len(sb_DM_dTrans_cent) == 1:
                    st.write("-",len(sb_DM_dTrans_cent), " column was centered:", sb_DM_dTrans_cent[0])
                elif len(sb_DM_dTrans_cent) == 0:
                    st.write("- No column was centered!")
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
                if sb_DM_dTrans_numCat is not None:
                    if len(sb_DM_dTrans_numCat) > 1:
                        st.write("-", len(sb_DM_dTrans_numCat), " columns were transformed to numeric categories:", ', '.join(sb_DM_dTrans_numCat))
                    elif len(sb_DM_dTrans_numCat) == 1:
                        st.write("-",len(sb_DM_dTrans_numCat), " column was transformed to numeric categories:", sb_DM_dTrans_numCat[0])
                elif sb_DM_dTrans_numCat is None:
                    st.write("- No column was transformed to numeric categories!")
                # multiplication
                if sb_DM_dTrans_mult != 0:
                    st.write("-", "Number of variable multiplications: ", sb_DM_dTrans_mult)
                elif sb_DM_dTrans_mult == 0:
                    st.write("- No variables were multiplied!")
                # division
                if sb_DM_dTrans_div != 0:
                    st.write("-", "Number of variable divisions: ", sb_DM_dTrans_div)
                elif sb_DM_dTrans_div == 0:
                    st.write("- No variables were divided!")
                st.write("")
                st.write("")
        
        #------------------------------------------------------------------------------------------
        
        #++++++++++++++++++++++
        # UPDATED DATA SUMMARY   

        # Show only if changes were made
        if  any(v for v in [sb_DM_delCols, sb_DM_dImp_num, sb_DM_dImp_other, sb_DM_dTrans_log, sb_DM_dTrans_sqrt, sb_DM_dTrans_square, sb_DM_dTrans_cent, sb_DM_dTrans_stand, sb_DM_dTrans_norm, sb_DM_dTrans_numCat ] if v is not None) or sb_DM_delDup == "Yes" or sb_DM_delRows_wNA == "Yes" or sb_DM_dTrans_mult != 0 or sb_DM_dTrans_div != 0 or filter_var != "-" or delRows!='-' or keepRows!='-' or len(sb_DM_keepCols) > 0:
            dev_expander_dsPost = st.expander("Explore cleaned and transformed data info and stats", expanded = False)
            with dev_expander_dsPost:
                if df.shape[1] > 0 and df.shape[0] > 0:

                    # Show cleaned and transformed data & data info
                    df_summary_post = fc.data_summary(df)
                    if st.checkbox("Show cleaned and transformed data", value = False):  
                        n_rows_post = df.shape[0]
                        n_cols_post = df.shape[1]
                        st.dataframe(df)
                        st.write("Data shape: ", n_rows_post, "rows and ", n_cols_post, "columns")
                         # Download transformed data:
                        output = BytesIO()
                        excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                        df.to_excel(excel_file, sheet_name="Clean. and transf. data")
                        excel_file.save()
                        excel_file = output.getvalue()
                        b64 = base64.b64encode(excel_file)
                        dl_file_name = "CleanedTransfData__" + df_name + ".xlsx"
                        st.markdown(
                            f"""
                        <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download cleaned and transformed data</a>
                        """,
                        unsafe_allow_html=True)
                        st.write("")

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

                        # Download link for cleaned summary statistics
                        output = BytesIO()
                        excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                        df.to_excel(excel_file, sheet_name="cleaned_data")
                        df_summary_post["Variable types"].to_excel(excel_file, sheet_name="cleaned_variable_info")
                        df_summary_post["ALL"].to_excel(excel_file, sheet_name="cleaned_summary_statistics")
                        excel_file.save()
                        excel_file = output.getvalue()
                        b64 = base64.b64encode(excel_file)
                        dl_file_name = "Cleaned data summary statistics_geo_" + df_name + ".xlsx"
                        st.markdown(
                            f"""
                        <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download cleaned data summary statistics</a>
                        """,
                        unsafe_allow_html=True)
                        st.write("") 

                        if fc.get_mode(df).loc["n_unique"].any():
                            st.caption("** Mode is not unique.") 
                else: st.error("ERROR: No data available for data preprocessing!") 

    #--------------------------------------------------
    #--------------------------------------------------
    # Geodata processing
    #---------------------------------------------------
    
    data_geodata_processing_container = st.container()
    with data_geodata_processing_container:
        #initialisation:
        anim_show=False
        #check what variables are numerical ones
        st.write("")
        st.write("")
        st.header('**Geospatial data visualisation**')
        
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
            a4,a5=st.columns(2)
            with a4:
                map_code=st.selectbox('What kind of country info do you have?',['country name','country code'], key = st.session_state['key'])
                map_loc=st.selectbox('Select the data column with the country info',objcat_cols, key = st.session_state['key'])
                
            with a5:
                if not 'Ladder' in list(df.columns):
                    map_var=st.selectbox('Select the variable to plot',num_cols, key = st.session_state['key'])
                else:
                    map_var=st.selectbox('Select the variable to plot',num_cols, index=1, key = st.session_state['key']) 
                map_time_filter=st.selectbox('Select time variable (if avaiable)',list('-')+ list(num_cols) + list(objcat_cols), key = st.session_state['key'])
                    
            if map_time_filter !='-':
                anim_show=st.checkbox('Show animation of temporal development?',value=False, key = st.session_state['key'])

            miss_val_show=st.checkbox('Show contours of countries with missing values?', value =False, key = st.session_state['key'])
            #set mapping key for the geojson data:
            if map_code=='country name':
                fid_key='properties.name'
            else:
                fid_key='properties.adm0_a3'

            # read geojson ne_110m_admin_0_countries.geojson: 110m resolution geojson from http://geojson.xyz/ based on Natural Earth Data            
            
            geojson_file =("default data/ne_110m_admin_0_countries.geojson")
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




            # place time-bar out of the column:        
            if map_time_filter !='-':
                try:
                    if int(df[map_time_filter][0])<3000:
                        time_to_filter = st.slider('time', min(df[map_time_filter]), max(df[map_time_filter]),min(df[map_time_filter]))
                except ValueError:                    
                    time_to_filter =st.selectbox('Specify the time',list(df[map_time_filter]), key = st.session_state['key'])
                
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
            
            a4,a5=st.columns(2)
            with a4:
                #plot relative frequency
                all_list=list(map_entities)
                all_list[1:len(all_list)-1]=all_list
                all_list[0]= 'all data'
                        
                rf_map_var = st.selectbox('Draw relative frequency for...?',all_list, key = st.session_state['key'])  
                            
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
                bx_map_var = st.selectbox('Draw a boxplot for ' + map_var + ' for...?', map_entities, key = st.session_state['key'])  
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
        

