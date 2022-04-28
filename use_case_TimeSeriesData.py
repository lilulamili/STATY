import streamlit as st
import pandas as pd
import numpy as np
import functions as fc
import modelling as ml
import os
import datetime
import time
import plotly.express as px 
import plotly.graph_objects as go
import sys
import platform
import base64
from io import BytesIO
from io import StringIO
import csv

from scipy import stats
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima.datasets import load_wineind


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

    #++++++++++++++++++++++++++++++++++++++++++++
    # DATA IMPORT

    # File upload section
    df_dec = st.sidebar.radio("Get data", ["Use example dataset", "Upload data"], key = st.session_state['key'])
    uploaded_data=None
    if df_dec == "Upload data":
        #st.subheader("Upload your data")
        #uploaded_data = st.sidebar.file_uploader("Make sure that dot (.) is a decimal separator!", type=["csv", "txt"])
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
            df = pd.read_csv(uploaded_data, sep = ";|,|\t",engine='python')
            df_name=os.path.splitext(uploaded_data.name)[0]
            st.sidebar.success('Loading data... done!')
        elif uploaded_data is None:
           df = pd.read_csv("default data/Air_passengers.csv", sep = ";|,|\t",engine='python')
           df_name='Air passangers'
    else:
        df = pd.read_csv("default data/Air_passengers.csv", sep = ";|,|\t",engine='python')
        df_name='Air passangers'
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

    st.header("**Time series data**")
    st.markdown("Let STATY do the data cleaning, variable transformations, visualisations and deliver you the stats you need. Specify your data processing preferences and start exploring your data stories right below... ")

    # Check if enough data is available
    if n_cols > 0 and n_rows > 0:
        st.empty()
    else:
        st.error("ERROR: Not enough data!")
        return

    data_exploration_container = st.container()
    with data_exploration_container:
        st.header("**Data screening and processing**")
        #------------------------------------------------------------------------------------------

        #++++++++++++++++++++++
        # DATA SUMMARY

        # Main panel for data summary (pre)
        #----------------------------------

        ts_expander_raw = st.expander("Explore raw data info and stats")
        with ts_expander_raw:
            # Default data description:
            if uploaded_data == None:
                if st.checkbox("Show data description", value = False, key = st.session_state['key']):          
                    st.markdown("**Data source:**")
                    st.markdown("The data come from Box & Jenkins (1970), but we use the version that is integrated in the R package ['astsa'] (https://www.stat.pitt.edu/stoffer/tsa4/ ) which is a companion to the book ['Time Series Analysis and Its Applications'] (https://www.springer.com/de/book/9783319524511) by Shumway & Stoffer's (2017)  .")
                                       
                    st.markdown("**Citation:**")
                    st.markdown("Box, G.E.P. and G.M. Jenkins (1970).Time Series Analysis, Forecasting, and Control. Oakland,CA: Holden-Day")
                    st.markdown("Shumway, R.H, and D.S. Stoffer (2017) Time Series Analysis and Its Applications: With R Examples. New York: Springer")
                    
                    st.markdown("**Variables in the dataset:**")

                    col1,col2=st.columns(2) 
                    col1.write("Air passengers")
                    col2.write("The monthly totals of international airline passengers")
                    
                    col1,col2=st.columns(2)
                    col1.write("Date ")
                    col2.write("Month ranging from January 1949 to December 1960")
                    
                    st.markdown("")
            # Show raw data & data info
            df_summary = fc.data_summary(df) 
            if st.checkbox("Show raw time series data", value = False, key = st.session_state['key']):      
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
            if st.checkbox('Show summary statistics (raw data)', value = False, key = st.session_state['key'] ): 
                #st.write(df_summary["ALL"])
                df_datasumstat=df_summary["ALL"]

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

                #dfStyler = df_datasumstat.style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector = 'th', props=[('text-align', 'left')])]) 
                a7, a8 = st.columns(2)
                with a7:
                    st.table(df_datasumstat)
                    if fc.get_mode(df).loc["n_unique"].any():
                        st.caption("** Mode is not unique.")

        #++++++++++++++++++++++
        # DATA PROCESSING

        # Settings for data processing
        #-------------------------------------
        
        #st.write("")
        #st.subheader("Data processing")

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
                sb_DM_delCols = st.multiselect("Select columns to delete ", df.columns, key = st.session_state['key'])
                df = df.loc[:,~df.columns.isin(sb_DM_delCols)]

                # Keep columns
                sb_DM_keepCols = st.multiselect("Select columns to keep", df.columns, key = st.session_state['key'])
                if len(sb_DM_keepCols) > 0:
                    df = df.loc[:,df.columns.isin(sb_DM_keepCols)]

                # Delete duplicates if any exist
                if df[df.duplicated()].shape[0] > 0:
                    sb_DM_delDup = st.selectbox("Delete duplicate rows ", ["No", "Yes"], key = st.session_state['key'])
                    if sb_DM_delDup == "Yes":
                        n_rows_dup = df[df.duplicated()].shape[0]
                        df = df.drop_duplicates()
                elif df[df.duplicated()].shape[0] == 0:   
                    sb_DM_delDup = "No"    
                    
                # Delete rows with NA if any exist
                n_rows_wNAs = df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0]
                if n_rows_wNAs > 0:
                    sb_DM_delRows_wNA = st.selectbox("Delete rows with NAs ", ["No", "Yes"], key = st.session_state['key'])
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
                    if sb_DM_delRows_wNA == "No" and n_rows_wNAs > 0:
                        st.markdown("**Data imputation**")
                        sb_DM_dImp_choice = st.selectbox("Replace entries with NA ", ["No", "Yes"], key = st.session_state['key'])
                        if sb_DM_dImp_choice == "Yes":
                            # Numeric variables
                            sb_DM_dImp_num = st.selectbox("Imputation method for numeric variables ", ["Mean", "Median", "Random value"], key = st.session_state['key'])
                            # Other variables
                            sb_DM_dImp_other = st.selectbox("Imputation method for other variables ", ["Mode", "Random value"], key = st.session_state['key'])
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
                sb_DM_dTrans_log = st.multiselect("Select columns to transform with log ", transform_options, key = st.session_state['key'])
                if sb_DM_dTrans_log is not None: 
                    df = fc.var_transform_log(df, sb_DM_dTrans_log)
                sb_DM_dTrans_sqrt = st.multiselect("Select columns to transform with sqrt ", transform_options, key = st.session_state['key'])
                if sb_DM_dTrans_sqrt is not None: 
                    df = fc.var_transform_sqrt(df, sb_DM_dTrans_sqrt)
                sb_DM_dTrans_square = st.multiselect("Select columns for squaring ", transform_options, key = st.session_state['key'])
                if sb_DM_dTrans_square is not None: 
                    df = fc.var_transform_square(df, sb_DM_dTrans_square)
                sb_DM_dTrans_cent = st.multiselect("Select columns for centering ", transform_options, key = st.session_state['key'])
                if sb_DM_dTrans_cent is not None: 
                    df = fc.var_transform_cent(df, sb_DM_dTrans_cent)
                sb_DM_dTrans_stand = st.multiselect("Select columns for standardization ", transform_options, key = st.session_state['key'])
                if sb_DM_dTrans_stand is not None: 
                    df = fc.var_transform_stand(df, sb_DM_dTrans_stand)
                sb_DM_dTrans_norm = st.multiselect("Select columns for normalization ", transform_options, key = st.session_state['key'])
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
                sb_DM_dTrans_div = st.number_input("Number of variable divisions ", value = 0, min_value=0, key = st.session_state['key'])
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
            
            if st.checkbox('Show a summary of my data processing preferences ', value = False, key = st.session_state['key']): 
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
        if any(v for v in [sb_DM_delCols, sb_DM_dImp_num, sb_DM_dImp_other, sb_DM_dTrans_log, sb_DM_dTrans_sqrt, sb_DM_dTrans_square, sb_DM_dTrans_cent, sb_DM_dTrans_stand, sb_DM_dTrans_norm, sb_DM_dTrans_numCat ] if v is not None) or sb_DM_delDup == "Yes" or sb_DM_delRows_wNA == "Yes" or filter_var != "-" or delRows!='-' or keepRows!='-' or len(sb_DM_keepCols) > 0:
            dev_expander_dsPost = st.expander("Explore cleaned and transformed data info and stats", expanded = False)
            with dev_expander_dsPost:
                if df.shape[1] > 0 and df.shape[0] > 0:

                    # Show cleaned and transformed data & data info
                    df_summary_post = fc.data_summary(df)
                    if st.checkbox("Show cleaned and transformed data ", value = False, key = st.session_state['key']):  
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
                        check_nasAnddupl2 = st.checkbox("Show duplicates and NAs info (processed) ", value = False, key = st.session_state['key']) 
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
                    if st.checkbox("Show cleaned and transformed variable info ", value = False, key = st.session_state['key']): 
                        st.write(df_summary_post["Variable types"])

                    # Show summary statistics (cleaned and transformed data)
                    if st.checkbox('Show summary statistics (cleaned and transformed data) ', value = False, key = st.session_state['key']):
                        st.write(df_summary_post["ALL"])

                        # Download link for cleaned data statistics
                        output = BytesIO()
                        excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                        df.to_excel(excel_file, sheet_name="cleaned_data")
                        df_summary_post["Variable types"].to_excel(excel_file, sheet_name="cleaned_variable_info")
                        df_summary_post["ALL"].to_excel(excel_file, sheet_name="cleaned_summary_statistics")
                        excel_file.save()
                        excel_file = output.getvalue()
                        b64 = base64.b64encode(excel_file)
                        dl_file_name = "Cleaned data summary statistics_ts_" + df_name + ".xlsx"
                        st.markdown(
                            f"""
                        <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download cleaned data summary statistics</a>
                        """,
                        unsafe_allow_html=True)
                        st.write("")

                        if fc.get_mode(df).loc["n_unique"].any():
                            st.caption("** Mode is not unique.")  
                else: st.error("ERROR: No data available for preprocessing!") 
            
    
    #--------------------------------------------------
    #--------------------------------------------------
    # Time-series data
    #---------------------------------------------------
    data_predictiv_analysis_container = st.container()
    with data_predictiv_analysis_container:

        st.write("")
        st.write("")
        st.header('**Predictive data analysis and modelling**')
        st.write('Go for creating predictive models of your data using classical univariate time-series techniques! Staty will take care of the modelling for you, so you can put your focus on results interpretation and communication!')
            
        # check if dataset contains numerical data
        num_cols=df.columns  
        date_cols=df.columns
        for column in df:              
            if not df[column].dtypes in ('float', 'float64', 'int','int64','datetime64'): 
                num_cols=num_cols.drop(column) 
            elif df[column].dtypes !='datetime64':
                date_cols=date_cols.drop(column) 

        if len(num_cols)==0 or (len(num_cols)==1 and len(date_cols)==0):
            st.error("ERROR: Your dataset is not suitable for the time series analysis!")       
        else:
            a4,a5=st.columns(2)
            with a4:
                ts_var=st.selectbox('Select the variable for time-series analysis and modelling', list(num_cols), key = st.session_state['key'])
                #ts_exo=st.selectbox('Select exogenous variables for your model', list(num_cols), key = st.session_state['key'])
    
            with a5:
                ts_time=st.selectbox('Select the time info for your data',list(date_cols)+list(num_cols), key = st.session_state['key'])
            
            #time series:
            ts=df[[ts_var,ts_time]]
            ts_show_ts=st.checkbox('Show time series data',value=False, key = st.session_state['key'])

            if ts_show_ts:
                st.write(ts)
            

            # check NA's:
            n_NAs = df.iloc[list(pd.unique(np.where(ts.isnull())[0]))].shape[0]
            if n_NAs>0:
                st.error("ERROR: Your data has missing values! Find a solution for the missing values or let STATY do that for you - in the latter case, please check the menu above 'Specify your data processing preferences' (i.e., select the method that fits well your data in the 'Data imputation' section, or select 'Delete rows with NAs' in the 'Data cleaning' section)!")
            else:
                #---------------------------------
                #convert time column to datetime  
                #---------------------------------          
                try:
                    if 35<= ts[ts_time][0] <=3000:# check if the date is year only
                        ts=ts.set_index(pd.to_datetime(ts[ts_time], format='%Y')) 
                                    
                except:
                    try:
                        ts=ts.set_index(pd.to_datetime(ts[ts_time],dayfirst=True,errors='raise'))     
                        
                                    
                    except:
                        if ts_var==ts_time:
                            st.error("ERROR: Variable for time series analysis and the time info should not be the same!")
                            return
                        else:    
                            st.error("ERROR: Please change the format of the time info of your data -     \n  check 'pandas.to_datetime' requirements!")          
                            return
    
                #---------------------------------
                # Diagnosis plots and stats
                #---------------------------------
                # initialisation
                st_dif_order=1

                st.write("")
                ts_expander_datavis = st.expander("Diagnosis plots and tests")
                with ts_expander_datavis:
                
                    st.write('**Time series pattern**')
                    
                    ts_pattern_sel=st.selectbox('Select the analysis type',['Fixed window statistics check','Simple moving window', 'Zoom in data'], key = st.session_state['key'])
                
                    if ts_pattern_sel=='Fixed window statistics check':
                        
                        a4,a5=st.columns(2)  
                        time_list=list(ts.index) 
                        with a4: 
                            start_time=st.selectbox('Specify the window start',list(ts.index),index=0, key = st.session_state['key'])
                        with a5:
                            end_time=st.selectbox('Specify the window end',list(ts.index),index=len(list(ts.index))-1, key = st.session_state['key'])
                            if end_time<start_time:
                                st.error('ERROR: End time cannot be before start time!')
                                return
                        
                        #filter out time series within a given range  
                        ts_selection=np.logical_and(ts.index>=start_time ,ts.index<=end_time)
                        filtered_data = ts.loc[ts_selection]   
                        
                        filt_stats= pd.DataFrame(index = ["mean", "std", "variance"], columns = ["Window statistics"])
                        filt_stats["Window statistics"][0]=filtered_data[ts_var].mean()
                        filt_stats["Window statistics"][1]=filtered_data[ts_var].std()
                        filt_stats["Window statistics"][2]=filtered_data[ts_var].var()
                        st.table(filt_stats)

                        fig = px.area(x=df[ts_time], y=df[ts_var], color_discrete_sequence=['rgba(55, 126, 184, 0.7)'])
                        fig.add_trace(go.Scatter(x=filtered_data[ts_time], y=filtered_data[ts_var], fill='tonexty',mode='lines',line_color='rgba(198,224,180, 0.6)')) # fill to trace0 y
                        fig.update_layout(showlegend=False)
                        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
                        fig.update_layout(yaxis=dict(title=ts_var, titlefont_size=12, tickfont_size=14,),)
                        fig.update_layout(xaxis=dict(title="", titlefont_size=12, tickfont_size=14,),)
                        
                        #fig.update_xaxes(rangeslider_visible=True)
                        st.plotly_chart(fig,use_container_width=True) 
                        
                    elif ts_pattern_sel=='Simple moving window':
                        #calculate the moving average    
                        ts_window_size=st.number_input('Specify the window size',min_value=1,max_value=len(ts))
                        ts_mw_mean=ts[ts_var].rolling(window=ts_window_size).mean()
                        ts_mw_std=ts[ts_var].rolling(window=ts_window_size).std()
                        st.write("**Moving window mean and standard deviation**")
                    # st.write('If the window has a size of 3, the moving average will start from the 3rd sample value')

                        fig = px.area(x=df[ts_time], y=df[ts_var], color_discrete_sequence=['rgba(55, 126, 184, 0.7)'])
                        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
                        fig.update_layout(yaxis=dict(title=ts_var, titlefont_size=12, tickfont_size=14,),)
                        fig.update_layout(xaxis=dict(title="", titlefont_size=12, tickfont_size=14,),)
                        fig.add_trace(go.Scatter(x=ts[ts_time], y=ts_mw_mean, fill='tonexty',mode='lines',line_color='rgba(198,224,180, 0.6)')) # fill to trace0 y
                        fig.add_trace(go.Scatter(x=ts[ts_time], y=ts_mw_std, fill='tozeroy',mode='lines',line_color='rgba(233,183,123, 1)')) # fill to trace0 y
                        fig.update_layout(showlegend=False)
                        
                        st.plotly_chart(fig,use_container_width=True) 
                
                    elif ts_pattern_sel=='Zoom in data':
                        st.info('You can inspect the series using a slider below the chart')
                        
                        fig = px.area(x=df[ts_time], y=df[ts_var], color_discrete_sequence=['rgba(55, 126, 184, 0.7)'])
                        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
                        fig.update_layout(yaxis=dict(title=ts_var, titlefont_size=12, tickfont_size=14,),)
                        fig.update_layout(xaxis=dict(title="", titlefont_size=12, tickfont_size=14,),)
                        
                        fig.update_xaxes(rangeslider_visible=True)
                        st.plotly_chart(fig,use_container_width=True) 



                    st.write('**Autocorrelation and partial autocorrelation plots**')
                    # diagnosis plots for raw data                  
                    fig, (ax1, ax2, ax3)  = plt.subplots(1, 3, sharex=False)
                    fig.set_figheight(2)
                    plt.rcParams.update({'font.size': 8})
                    plt.rcParams['lines.linewidth'] = 1
                    plt.rcParams['lines.markersize']= 1
                    ax1.plot(ts[ts_var])
                    ax1.set_title('Time series')
                    #nlab=len(ax1.get_xticklabels())
                    #for i, label in enumerate(ax1.get_xticklabels()):
                    #    if i > 1 and i < (nlab-1):
                     #       label.set_visible(False)
                    fig.autofmt_xdate() 
                    ax1.set_ylabel('raw series')
                    
                    plot_acf(df[ts_var], ax=ax2)
                    ax2.set_title('ACF')
                    plot_pacf(df[ts_var], ax=ax3,lags=df.shape[0] // 2 - 2 )
                    ax3.set_title('PACF')

                    for k in [ax1,ax2,ax3]:
                        k.spines['top'].set_visible(False)
                        k.spines['right'].set_visible(False)
                        k.spines['bottom'].set_visible(False)
                        k.spines['left'].set_visible(False)              
                    st.pyplot(fig)


                    #Augmented Dickey Fuller Test (ADF Test) Ho:not stationary H1:stationary
                    st.markdown('')
                    st.write('**Augmented Dickey Fuller Test**')
                    adf_test = adfuller(ts[ts_var])
                    st.write('ADF:     %f' % adf_test[0])
                    st.write('p-value: %f' % adf_test[1])
                    st.markdown("")
                    
                    if sett_hints:
                        st.info(str(fc.learning_hints("ts_time_series_pattern")))
                        st.write("")
                    


            #-----------------------------------------------------------
            # Detrending and seasonal adjustment
            #-----------------------------------------------------------
            ts_expander_decomp = st.expander("Differencing, detrending and seasonal adjustment")
            with ts_expander_decomp:
                ts_decomp = st.selectbox("Specify your time series differencing and decomposition preferences:", 
                    ["n-order differences", "detrending", "seasonal adjustment", "detrending & seasonal adjustment"], key = st.session_state['key'])

                #----------------------------------------------------------
                # n-order differences
                #----------------------------------------------------------
                if ts_decomp=="n-order differences":   
                                    
                    st_dif_order=st.number_input('Specify the highest differencing order',min_value=1, key = st.session_state['key'])
                    st.write("")  
                   
                    # initialize table for the ADF test results:                      
                    adf_list=['raw series']  
                    for k in range(st_dif_order):
                        adf_list.append(str(k+1)+ '. order differences')
                    adf_test_ndiff=pd.DataFrame(index= adf_list,columns=['ADF', 'p-value'])
                    adf_test_ndiff['ADF'][0] = adfuller(ts[ts_var])[0]
                    adf_test_ndiff['p-value'][0] = adfuller(ts[ts_var])[1]
                    
                    # figure initialisation
                    fig, ax  = plt.subplots(st_dif_order+1, 3, sharex=False)
                    fig.subplots_adjust(hspace=.5)
                    fig.set_figheight(st_dif_order*3)
                    plt.rcParams.update({'font.size': 8})
                    plt.rcParams['lines.linewidth'] = 1
                    plt.rcParams['lines.markersize']= 1
                    
                    # raw data & ACF
                    ax[0, 0].plot(ts[ts_var])
                    ax[0, 0].set_title('Time series')
                    fig.autofmt_xdate() 
                    #nlab=len(ax[0,0].get_xticklabels())
                    #for i, label in enumerate(ax[0,0].get_xticklabels()):
                    #    if i > 1 and i < (nlab-1):
                    #        label.set_visible(False)
                    ax[0, 0].set_ylabel('raw series')
                    plot_acf(df[ts_var], ax=ax[0,1])
                    ax[0, 1].set_title('ACF')
                    plot_pacf(df[ts_var], ax=ax[0,2],lags=df.shape[0] // 2 - 2)
                    ax[0, 2].set_title('PACF')
                    for k in range(3):
                        ax[0, k].spines['top'].set_visible(False)
                        ax[0, k].spines['right'].set_visible(False)
                        ax[0, k].spines['bottom'].set_visible(False)
                        ax[0, k].spines['left'].set_visible(False)  
                    
                    # n-order differences & ACF
                    ts_difs=ts[ts_var]
                    for j in range(1,st_dif_order+1):
                        ts_difs=ts_difs.diff()
                            
                        #ADF test
                        ts[adf_list[j]]=ts_difs
                        adf_test_ndiff['ADF'][j] = adfuller(ts_difs.dropna())[0]
                        adf_test_ndiff['p-value'][j] = adfuller(ts_difs.dropna())[1]

                        # ACF & PACF chart for n-diffs
                        ax[j, 0].plot(ts_difs)
                        ax[j, 0].set_ylabel(str(j) +'. order diff.')
                        nlab_j=len(ax[j,0].get_xticklabels())
                        for i, label in enumerate(ax[j,0].get_xticklabels()):
                            if i > 1 and i < (nlab_j-1):
                                label.set_visible(False)
                        fig.autofmt_xdate() 
                        plot_acf(ts_difs.dropna(), ax=ax[j,1])
                        plot_pacf(ts_difs.dropna(), ax=ax[j,2],lags=ts_difs.dropna().shape[0] // 2 - 2)
                        
                        ax[j, 1].set_title('')
                        ax[j, 2].set_title('')

                        for k in range(3):
                            ax[j, k].spines['top'].set_visible(False)
                            ax[j, k].spines['right'].set_visible(False)
                            ax[j, k].spines['bottom'].set_visible(False)
                            ax[j, k].spines['left'].set_visible(False)              
                    st.pyplot(fig)   
                    
                    st.write("")
                    st.write('**Augmented Dickey Fuller Test**')
                    st.table(adf_test_ndiff)

                    st.write("")
                    if sett_hints:
                        st.info(str(fc.learning_hints("ts_n_order_differences")))
                        st.write("")
                    
                
                    # data selection for further modelling
                    st.write("")                    
                    st_order_selection=st.selectbox('Select data for further modelling',adf_list, key = st.session_state['key'])
                    if st_order_selection=='raw series':
                        ts_sel_data=ts[ts_var] 
                    else:       
                        ts_sel_data=ts[st_order_selection]

                    ts_show_ndifData=st.checkbox('Show selected data?', value=False, key = st.session_state['key'])    
                    if ts_show_ndifData:
                        st.write(ts_sel_data)
                    st.markdown('')  

                #----------------------------------------------------------
                # Detrending
                #----------------------------------------------------------
                elif ts_decomp=="detrending": 
                    st.write("")
                    st.write('**Time series and the trend component**')  
                    
                    # decompose time series:
                    ts_decom_name='detrended'
                    ml.decompose_plots(ts_decomp,ts_decom_name,df,ts,ts_var,ts_time)
                    ts_detrended_show=st.checkbox('Show ACF and PACF of detrended series?', value=False)
                    if ts_detrended_show:
                        ml.series_acf_pacf_plot(ts_decom_name,ts[ts_decom_name])

                    #Augmented Dickey Fuller Test
                    st.write("")
                    st.write('**Augmented Dickey Fuller Test**')
                    ml.adf_test(ts_decom_name,ts,ts_var)

                    st.write("")
                    if sett_hints:
                        st.info(str(fc.learning_hints("ts_detrending_hints")))
                        st.write("")
    
                    # data selection for further modelling
                    st.write("")                   
                    st_order_selection=st.selectbox('Select data for further modelling',['raw series', 'detrended data'])
                    if st_order_selection=='raw series':
                        ts_sel_data=ts[ts_var] 
                    else:                     
                        ts_sel_data=ts[ts_decom_name]
                    
                    ts_show_ndifData=st.checkbox('Show selected data?', value=False, key = st.session_state['key'])   
                    if ts_show_ndifData:
                        st.write(ts_sel_data)
                    st.markdown('')  
                    


                #----------------------------------------------------------
                # Seasonal adjustement
                #----------------------------------------------------------

                elif ts_decomp=="seasonal adjustment":
                    st.write("")
                    st.write('**Time series and the seasonal component**') 
                    # decompose time series:
                    ts_decom_name='seasonally adjusted'
                    ml.decompose_plots(ts_decomp,ts_decom_name,df,ts,ts_var,ts_time)
                    
                    ts_season_show=st.checkbox('Show ACF and PACF of seasonally adjusted series?', value=False)
                    if ts_season_show:
                        ml.series_acf_pacf_plot(ts_decom_name,ts[ts_decom_name])

                    #Augmented Dickey Fuller Test
                    st.write("")
                    st.write('**Augmented Dickey Fuller Test**')
                    ml.adf_test(ts_decom_name,ts,ts_var)

                    st.write("")
                    if sett_hints:
                        st.info(str(fc.learning_hints("ts_seasonal_hints")))
                        st.write("")

                    # data selection for further modelling
                    st.write("")
                    st_order_selection=st.selectbox('Select data for further modelling',['raw series', 'seasonally adjusted data'])
                    if st_order_selection=='raw series':
                        ts_sel_data=ts[ts_var] 
                    else:                     
                        ts_sel_data=ts[ts_decom_name]
 
                    ts_show_ndifData=st.checkbox('Show selected data', value=False, key = st.session_state['key']) 
                    if ts_show_ndifData:
                        st.write(ts_sel_data)
                    st.markdown('')  
                    

                #----------------------------------------------------------
                # Detrending & seasonal adjustement
                #----------------------------------------------------------    
                elif ts_decomp=="detrending & seasonal adjustment":
                    st.write('**Time series, trend and the seasonal component**')
                # decompose time series:
                    ts_decom_name='detrended and seasonally adjusted'
                    ml.decompose_plots(ts_decomp,ts_decom_name,df,ts,ts_var,ts_time)
                    
                    ts_ds_show=st.checkbox('Show ACF and PACF of detrended and seasonally adjusted series?', value=False)
                    if ts_ds_show:
                        ml.series_acf_pacf_plot(ts_decom_name,ts[ts_decom_name])

                    #Augmented Dickey Fuller Test
                    st.write("")
                    st.write('**Augmented Dickey Fuller Test**')
                    ml.adf_test(ts_decom_name,ts,ts_var)

                    st.write("")
                    if sett_hints:
                        st.info(str(fc.learning_hints("ts_detrend_seasonal_hints")))
                        st.write("")

                    # data selection for further modelling
                    st.write("")
                    st.write('**Select data for further modelling:**')
                    st_order_selection=st.selectbox('',['raw series', ts_decom_name])
                    if st_order_selection=='raw series':
                        ts_sel_data=ts[ts_var] 
                    else:                     
                        ts_sel_data=ts[ts_decom_name]
                    
                    ts_show_ndifData=st.checkbox('Show selected data',  value=False, key = st.session_state['key']) 
                    if ts_show_ndifData:
                        st.write(ts_sel_data)
                    st.markdown('')  

                


            #-----------------------------------------------------------
            # TIME SERIES MODELLING
            #-----------------------------------------------------------
            
            st.write("")                 
                
            ts_sel_data=ts_sel_data.dropna()                       
            #initialisation                                   
            trend_key={'No':None,'constant term (intercept)':'c','linear trend':'t', 'second order polinomial':[1,1,1]}
            ts_ic_key={'AIC':'aic', 'BIC':'bic', 'HQIC':'hqic', 'OOB':'oob'}
            ts_train,ts_ic,ts_trend_spec=1,'AIC','constant term (intercept)'
            d=0     
            ts_expander_mod = st.expander("Model specification")
            with ts_expander_mod:
                ts_algorithms = ["MA", "AR", "ARMA", "non-seasonal ARIMA", "seasonal ARIMA"]
                ts_alg_list = list(ts_algorithms)
                ts_alg = st.selectbox("Select modelling technique", ts_alg_list, key = st.session_state['key'])
                
                st.write("")
                if sett_hints:
                    st.info(str(fc.learning_hints("ts_models_hints")))
                    st.write("")
                
                # Validation Settings                
                ts_modval= st.checkbox("Use model validation?", value=False, key = st.session_state['key'])
                if ts_modval:
                    a4,a5=st.columns(2)
                    with a4:
                        # Select training/ test ratio 
                        ts_train = st.slider("Select training data size", 0.5, 0.95, 0.8)
                        
                ts_forecast= st.checkbox("Use model for forecast?", value=False, key = st.session_state['key'])
                if ts_forecast:
                    a4,a5=st.columns(2)
                    with a4:
                        ts_forecast_no=st.number_input('Specify the number of forecast steps',min_value=1,value=2)
                
                
                ts_parametrisation= st.checkbox('Automatic parameterization of models?',value=True, key = st.session_state['key'])
                
                st.write("")
                if ts_parametrisation==False:  
                    #initialisation:              
                    p,q,d,pp,dd,qq,s=0,0,0,0,0,0,0
                    ts_trend_spec='constant term (intercept)' 

                    a4,a5=st.columns(2)
                    if ts_alg=='AR':
                        with a4:
                            p = st.slider("Select order of the AR model (p)", 1, 30, 2, key = st.session_state['key'])
                    elif ts_alg=='MA':
                        with a4:
                            q = st.slider("Select the MA 'window' size over your data (q)", 1, 15, 2, key = st.session_state['key'])
                    elif ts_alg=='ARMA':
                        with a4:
                            p = st.slider("Select order of the AR model (p)", 0, 15, 2, key = st.session_state['key'])
                            q = st.slider("Select the MA 'window' size over your data (q)", 0, 15, 2, key = st.session_state['key'])   
                    elif ts_alg =='non-seasonal ARIMA':
                        with a4:
                            p = st.slider("Select order of the AR model (p)", 0, 15, 2, key = st.session_state['key'])
                            d= st.slider("Select the degree of differencing (d)", 0, 15, 2, key = st.session_state['key'])
                            q = st.slider("Select the MA 'window' size over your data (q)", 0, 15, 2, key = st.session_state['key'])   
                    elif ts_alg=='seasonal ARIMA':
                        with a4:
                            p = st.slider("Select order of the AR model (p)", 0, 15, 0, key = st.session_state['key'])
                            d= st.slider("Select the degree of differencing (d)", 0, 15, 2, key = st.session_state['key'])
                            q = st.slider("Select the MA 'window' size over your data (q)", 0, 15, 0, key = st.session_state['key'])   

                        with a5:
                            pp = st.slider("Select the AR order of the seasonal component (P)", 0, 15, 1, key = st.session_state['key'])
                            dd= st.slider("Select the integration order (D)", 0, 30, 0, key = st.session_state['key'])
                            qq = st.slider("Select the MA order of the seasonal component (Q)", 0, 15, 1, key = st.session_state['key']) 
                            s = st.slider("Specify the periodicity (number of periods in season)", 0, 52, 2, key = st.session_state['key']) 
                        
                    #additional settings for the model calibartion:
                    ts_man_para_add=st.checkbox('Show additional settings for manual model calibration?', value=False, key = st.session_state['key']) 
                    if ts_man_para_add:                    
                        # trend specification                           
                        a4,a5=st.columns(2)
                        with a4:
                            ts_trend_spec=st.selectbox('Include a trend component in the model specification', ['No', 'constant term (intercept)', 'linear trend', 'second order polinomial'], key = st.session_state['key'])
                            
                    
                    st.write("")
                       
                
                # automatic paramater selection    
                else:                                       
                    if ts_alg=='AR':
                        p,q,pp,qq=1,0,0,0 
                        maxp,maxq,maxd,maxpp,maxdd,maxqq,s=5,0,1,0,0,0,1 
                    elif ts_alg=='MA':
                        p,q,pp,qq,s=0,1,0,0,1  
                        maxp,maxq,maxd,maxpp,maxdd,maxqq,s=0,5,1,0,0,0,1
                    elif ts_alg=='ARMA':
                        p,q,pp,qq=1,1,0,0  
                        maxp,maxq,maxd,maxpp,maxdd,maxqq,s=5,5,1,0,0,0,1 
                    elif ts_alg =='non-seasonal ARIMA':
                        p,q,pp,qq=2,2,0,0  
                        maxp,maxq,maxd,maxpp,maxdd,maxqq,s=5,5,2,0,0,0,1          
                    elif ts_alg=='seasonal ARIMA':
                        p,q,pp,qq=1,1,1,1 
                        maxp,maxq,maxd, maxpp,maxdd,maxqq,s=5,5,2,2,1,2,2

                    # additional settings for automatic model parametrisation
                    st_para_spec=st.checkbox('Show additional settings for automatic model parametrisation?', value=False, key = st.session_state['key'])              
                    if st_para_spec: 
                        #Information criterion used to select the model
                        a4,a5=st.columns(2)
                        with a4:
                            ts_ic=st.selectbox('Select the information crtiteria to be used for the model selection', ['AIC', 'BIC', 'HQIC', 'OOB'], key = st.session_state['key'])  
                        
                        #specification of the maximum valus for the model paramaters
                        a4,a5=st.columns(2)               
                        if ts_alg=='AR':
                            with a4:                                
                                maxp = st.slider("Maximum order of the AR model (max p)?", 1, 30, 5, key = st.session_state['key'])
                        elif ts_alg=='MA':
                            with a4:                                
                                maxq = st.slider("Maximum 'window' size over your data (max q)?", 1, 15, 5, key = st.session_state['key'])
                        elif ts_alg=='ARMA':
                            with a4:                                
                                maxp = st.slider("Maximum order of the AR model (max p)?", 0, 15, 2, key = st.session_state['key'])
                                maxq = st.slider("Maximum MA 'window' size over your data (max q)?", 0, 15, 2, key = st.session_state['key'])   
                        elif ts_alg =='non-seasonal ARIMA':
                            with a4:                               
                                maxp = st.slider("Maximum order of the AR model (max p)?", 0, 15, 5, key = st.session_state['key'])
                                maxd= st.slider("Maximum degree of differencing (max d)?", 0, 15, 2, key = st.session_state['key'])
                                maxq = st.slider("Maximum MA 'window' size over your data (max q)?", 0, 15, 5, key = st.session_state['key'])   
                        elif ts_alg=='seasonal ARIMA':
                            with a4:                               
                                maxp = st.slider("Maximum order of the AR model (max p)?", 0, 15, 5, key = st.session_state['key'])
                                maxd= st.slider("Maximum degree of differencing (max d)?", 0, 15, 2, key = st.session_state['key'])
                                maxq = st.slider("Maximum MA 'window' size over your data (max q)?", 0, 15, 5, key = st.session_state['key'])   

                            with a5:
                                maxpp = st.slider("Maximum AR order of the seasonal component (max P)", 0, 15, 2, key = st.session_state['key'])
                                maxdd= st.slider("Maximum integration order (max D)", 0, 30, 1, key = st.session_state['key'])
                                maxqq = st.slider("Maximum MA order of the seasonal component (max Q)", 0, 15, 2, key = st.session_state['key']) 
                                s = st.slider("Specify the periodicity (number of periods in season)", 0, 52, 2, key = st.session_state['key']) 
                    st.write("")
                
                #ts_data_output=st.checkbox("Include time series data in the output files", value=False)
                
                
                st.write("")    
                run_ts_model = st.button("Estimate time series model")  
                st.write("")    
                   
                    
                #------------------------------------------------------------------------
                # Model output
                #------------------------------------------------------------------------
                if run_ts_model: 
                                                            
                    #-------------------------------------
                    #model fitting
                    #-------------------------------------

                    


                    n_training=int(len(ts_sel_data)*ts_train)
                    ts_training = ts_sel_data[:n_training]
                    ts_test = ts_sel_data[n_training:]
 
                    if ts_parametrisation==False:
                        # SARIMAX "manual" model fit                            
                        mod = sm.tsa.statespace.SARIMAX(ts_sel_data[:n_training], order=(p,d,q),seasonal_order=(pp,dd,qq,s), trend=trend_key[ts_trend_spec])
                      
                    else:
                        
                        # pm.autoarima automatic model fit  
                        arima = pm.auto_arima(ts_sel_data[:n_training], start_p=p,  d=None, start_q=q, max_p=maxp, 
                            max_d=maxd, max_q=maxq, start_P=pp, D=None, start_Q=qq, max_P=maxpp, 
                            max_D=maxdd, max_Q=maxqq, max_order=5, m=s, seasonal=True,trace=True,
                            information_criterion=ts_ic_key[ts_ic],
                            error_action='ignore',  suppress_warnings=True,  stepwise=True)  

                        get_parametes = arima.get_params()        
                        # use SARIMAX to fit the model with the choosen paramaters (to simplify the model output reading functions)
                        mod = sm.tsa.statespace.SARIMAX(ts_sel_data[:n_training], order=arima.order,seasonal_order=get_parametes["seasonal_order"], trend='c')
                        
                    #---------------------------------------
                    #Model output statistics
                    #---------------------------------------
                    st.write("")
                    st.subheader("Model output statistics")     
                    ts_results = mod.fit()
                    st.text(ts_results.summary())
                    

                    st.write("")
                    if sett_hints:
                        st.info(str(fc.learning_hints("ts_model_results_hints")))
                        st.write("")
                    #---------------------------------------
                    # Residual diagnostics
                    #---------------------------------------
                    st.write("")
                    st.subheader("Diagnostic plots for standardized residuals")
                    ts_res_diagn=ts_results.plot_diagnostics(figsize=(10, 5))
                    st.write(ts_res_diagn)

                    #---------------------------------------
                    # Model validation
                    #--------------------------------------- 
                                       
                    st.subheader("One-step ahead predictions (dots)") 

                    ts_predict = ts_results.get_prediction()
                    ts_predict_ci =ts_predict.conf_int()
                    
                    lower_ci =  ts_predict_ci.iloc[d:, 0]
                    upper_ci =  ts_predict_ci.iloc[d:, 1]
                    ts_predict_mean=ts_predict.predicted_mean[d:,]
                    
                    fig = go.Figure()                                                    
                    fig.add_trace(go.Scatter(x=lower_ci.index,y=lower_ci, fill=None,mode='lines',line_color='rgba(255, 229, 229, 0.8)'))
                    fig.add_trace(go.Scatter(x=upper_ci.index,y=upper_ci, fill='tonexty',mode='lines',line_color='rgba(255, 229, 229, 0.8)')) 
                    fig.add_trace(go.Scatter(x=ts_sel_data.index, y=ts_sel_data, line=dict(color='rgba(31, 119, 180, 1)', width=1)))
                    fig.add_trace(go.Scatter(x=ts_predict_mean.index, y=ts_predict_mean, mode='markers', marker_size=4,
                        marker_color='indianred'))
                    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig,use_container_width=True) 

                    if ts_modval:
                                                   
                        st.subheader("Model validation")   
                        
                        ts_modval_data = ts_results.get_prediction(start=n_training, end=len(ts_sel_data)-1)
                        ts_modval_ci = ts_modval_data.conf_int()
                        ts_modval_lower_ci=ts_modval_ci.iloc[:, 0]
                        ts_modval_upper_ci=ts_modval_ci.iloc[:, 1]
                    
                        fig = go.Figure()                                                    
                        fig.add_trace(go.Scatter(x=ts_modval_lower_ci.index,y=ts_modval_lower_ci, fill=None,mode='lines',line_color='rgba(255, 229, 229, 0.8)'))
                        fig.add_trace(go.Scatter(x=ts_modval_upper_ci.index,y=ts_modval_upper_ci, fill='tonexty',mode='lines',line_color='rgba(255, 229, 229, 0.8)')) 
                        # observed data
                        fig.add_trace(go.Scatter(x=ts_sel_data.index, y=ts_sel_data, line=dict(color='rgba(31, 119, 180, 1)', width=1)))
                        # model validation
                        fig.add_trace(go.Scatter(x=ts_modval_data.predicted_mean.index, y=ts_modval_data.predicted_mean, mode='markers', marker_size=4,
                            marker_color='indianred'))
                        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig,use_container_width=True) 
                        
                        # Model evaluation                                                   
                        ts_model_eval_stats=ml.ts_model_evaluation(ts_test, ts_modval_data.predicted_mean)
                        st.table(ts_model_eval_stats)

                    #---------------------------------------
                    # Use model for forecast
                    #---------------------------------------
                    if ts_forecast:

                        st.write("") 
                        st.subheader(str(ts_forecast_no)+ " steps ahead forecast")

                        #get n-steps forecast (if the model validation was used then len(ts_test)+ts_forecast_no)                        
                        ts_fc = ts_results.get_forecast(steps=len(ts_test)+ts_forecast_no).summary_frame()
                        ts_fc=ts_fc.tail(ts_forecast_no)
                                            
                        # plot the forecast and the confidence intervals
                        fig = go.Figure()                                                    
                        fig.add_trace(go.Scatter(x=ts_fc["mean_ci_lower"].index,y=ts_fc["mean_ci_lower"], fill=None,mode='lines',line_color='rgba(255, 229, 229, 0.8)'))
                        fig.add_trace(go.Scatter(x=ts_fc["mean_ci_upper"].index,y=ts_fc["mean_ci_upper"], fill='tonexty',mode='lines',line_color='rgba(255, 229, 229, 0.8)')) 
                        # observed data
                        fig.add_trace(go.Scatter(x=ts_sel_data.index, y=ts_sel_data, line=dict(color='rgba(31, 119, 180, 1)', width=1)))
                        # model validation
                        fig.add_trace(go.Scatter(x=ts_fc["mean"].index, y=ts_fc["mean"], mode='markers', marker_size=4,
                            marker_color='indianred'))
                        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig,use_container_width=True) 

                        a4,a5=st.columns(2)
                        with a4:                        
                            st.write("")
                            st.table(ts_fc)   

                        st.write("")                        


                    # Download link
                    output = BytesIO()
                    excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                    ts_results_html = ts_results.summary().as_html()
                    ts_results_df_info = pd.read_html(ts_results_html)[0]
                    ts_results_df_coef = pd.read_html(ts_results_html, header = 0, index_col = 0)[1]
                    ts_results_df_tests = pd.read_html(ts_results_html)[2]
                    ts_sel_data.to_excel(excel_file, sheet_name="data")    
                    ts_results_df_info.to_excel(excel_file, sheet_name="ts_results_info")
                    ts_results_df_coef.to_excel(excel_file, sheet_name="ts_results_coef")
                    ts_results_df_tests.to_excel(excel_file, sheet_name="ts_results_tests")
                    excel_file.save()
                    excel_file = output.getvalue()
                    b64 = base64.b64encode(excel_file)
                    dl_file_name = "time series__" + df_name + ts_var + ts_time + ".xlsx"
                    st.markdown(
                        f"""
                    <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download time series results</a>
                    """,
                    unsafe_allow_html=True)
                    st.write("")    
                    st.write("") 
                    st.write("")   
                           
                        
                   


                                    
