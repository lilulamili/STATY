#----------------------------------------------------------------------------------------------

####################
# IMPORT LIBRARIES #
####################

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
import os
import altair as alt
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, roc_auc_score, max_error, log_loss, average_precision_score, precision_recall_curve, auc, roc_curve, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
import scipy
import sys
import platform
import base64
from io import BytesIO
from linearmodels import PanelOLS
from linearmodels import RandomEffects
from linearmodels import PooledOLS

#----------------------------------------------------------------------------------------------

def app(): 

    # Clear cache
    st.legacy_caching.clear_cache()

    # Hide traceback in error messages (comment out for de-bugging)
    sys.tracebacklimit = 0

    # Show altair tooltip when full screen
    st.markdown('<style>#vg-tooltip-element{z-index: 1000051}</style>',unsafe_allow_html=True)

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
    #------------------------------------------------------------------------------------------

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
                col_sep=a5.selectbox("Column sep.",[';',  ','  , '|', '\s+', '\t','other'], key = st.session_state['key'])
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
        elif uploaded_data is None:
           df = pd.read_csv("default data/Grunfeld.csv", sep = ";|,|\t",engine='python')
           df_name="Grunfeld" 
    else:
        df = pd.read_csv("default data/Grunfeld.csv", sep = ";|,|\t",engine='python') 
        df_name="Grunfeld" 
    st.sidebar.markdown("")
    
    #Basic data info
    n_rows = df.shape[0]
    n_cols = df.shape[1]

    #++++++++++++++++++++++++++++++++++++++++++++
    # SETTINGS

    settings_expander=st.sidebar.expander('Settings')
    with settings_expander:
        st.caption("**Precision**")
        user_precision=int(st.number_input('Number of digits after the decimal point',min_value=0,max_value=10,step=1,value=4, key = st.session_state['key']))
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
    # DATA PREPROCESSING & VISUALIZATION
    
    st.header("**Panel data**")
    st.markdown("Get your data ready for powerfull methods! Let STATY do data cleaning, variable transformations, visualizations and deliver you the stats you need. Specify your data processing preferences and start exploring your data stories right below... ")

    # Check if enough data is available
    if n_cols >= 2 and n_rows > 0:
        st.empty()
    else:
        st.error("ERROR: Not enough data!")
        return

    # Specify entity and time
    st.markdown("**Panel data specification**")
    col1, col2 = st.columns(2)
    with col1:
        entity_na_warn = False
        entity_options = df.columns
        entity = st.selectbox("Select variable for entity", entity_options, key = st.session_state['key'])
    with col2:
        time_na_warn = False
        time_options = df.columns 
        time_options = list(time_options[time_options.isin(df.drop(entity, axis = 1).columns)])
        time = st.selectbox("Select variable for time", time_options, key = st.session_state['key'])
        
    if np.where(df[entity].isnull())[0].size > 0:
        entity_na_warn = "ERROR: The variable selected for entity has NAs!"
        st.error(entity_na_warn)
    if np.where(df[time].isnull())[0].size > 0:
        time_na_warn = "ERROR: The variable selected for time has NAs!"
        st.error(time_na_warn)
    if df[time].dtypes != "float64" and df[time].dtypes != "float32" and df[time].dtypes != "int64" and df[time].dtypes != "int32":
        time_na_warn = "ERROR: Time variable must be numeric!"
        st.error(time_na_warn)
    

    run_models = False
    if time_na_warn == False and entity_na_warn == False:

        data_empty_container = st.container()
        with data_empty_container:
            st.empty()
            st.empty()
            st.empty()
            st.empty()
            st.empty()
            st.empty()
            st.empty()
            st.empty()

        # Make sure time is numeric
        df[time] = pd.to_numeric(df[time])

        data_exploration_container2 = st.container()
        with data_exploration_container2:

            st.header("**Data screening and processing**")

            #------------------------------------------------------------------------------------------

            #++++++++++++++++++++++
            # DATA SUMMARY

            # Main panel for data summary (pre)
            #----------------------------------

            dev_expander_dsPre = st.expander("Explore raw panel data info and stats", expanded = False)
            st.empty()
            with dev_expander_dsPre:
                # Default data description:
                if uploaded_data == None:
                    if st.checkbox("Show data description", value = False, key = st.session_state['key']):          
                        st.markdown("**Data source:**")
                        st.markdown("This is the original 11-firm data set from Grunfeld’s Ph.D. thesis (*Grunfeld, 1958, The Determinants of Corporate Investment, Department of Economics, University of Chicago*). For more details see online complements for the article [The Grunfeld Data at 50] (https://www.zeileis.org/grunfeld/).")
                        st.markdown("**Citation:**")
                        st.markdown("Kleiber C, Zeileis A (2010). “The Grunfeld Data at 50,” German Economic Review, 11(4), 404-417. [doi:10.1111/j.1468-0475.2010.00513.x] (https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-0475.2010.00513.x)")
                        st.markdown("**Variables in the dataset:**")

                        col1,col2=st.columns(2) 
                        col1.write("invest")
                        col2.write("Gross  investment,  defined  as  additions  to  plant  and  equipment  plus maintenance and repairs in millions of dollars deflated by the implicit price deflator of producers’ durable equipment (base 1947)")
                        
                        col1,col2=st.columns(2)
                        col1.write("value")
                        col2.write("Market  value  of  the  firm,  defined  as  the  price  of  common  shares  at December 31 (or, for WH, IBM and CH, the average price of December  31  and  January  31  of  the  following  year)  times  the  number  of common shares outstanding plus price of preferred shares at December 31 (or average price of December 31 and January 31 of the following year) times number of preferred shares plus total book value of debt at December 31 in millions of dollars deflated by the implicit GNP price deflator (base 1947)")
                        
                        col1,col2=st.columns(2)
                        col1.write("capital")
                        col2.write("Stock of plant and equipment, defined as the accumulated sum of net additions to plant and equipment deflated by the implicit price deflator for producers’ durable equipment (base 1947) minus depreciation allowance deflated by depreciation expense deflator (10 years moving average  of  wholesale  price  index  of  metals  and  metal  products,  base1947)")

                        col1,col2=st.columns(2)
                        col1.write("firm")
                        col2.write("General Motors (GM), US Steel (US), General Electric (GE), Chrysler (CH),  Atlantic Refining (AR), IBM, Union Oil (UO), Westinghouse (WH), Goodyear (GY), Diamond Match (DM), American Steel (AS)")

                        col1,col2=st.columns(2)
                        col1.write("year")
                        col2.write("Year ranging from 1935 to 1954")
                        st.markdown("")

                # Show raw data & data info
                df_summary = fc.data_summary(df) 
                if st.checkbox("Show raw data", value = False, key = st.session_state['key']):      
                    st.write(df)
                    #st.info("Data shape: "+ str(n_rows) + " rows and " + str(n_cols) + " columns")
                    st.write("Data shape: ", n_rows,  " rows and ", n_cols, " columns")
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
                    st.write(df_summary["Variable types"])
            
                # Show summary statistics (raw data)
                if st.checkbox('Show summary statistics (raw data)', value = False, key = st.session_state['key']): 
                    st.write(df_summary["ALL"].style.set_precision(user_precision))
                    
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

                    if fc.get_mode(df).loc["n_unique"].any():
                        st.caption("** Mode is not unique.")
                    if sett_hints:
                        st.info(str(fc.learning_hints("de_summary_statistics")))

                

            dev_expander_anovPre = st.expander("ANOVA for raw panel data", expanded = False)
            with dev_expander_anovPre:  
                if df.shape[1] > 2:
                    # Target variable
                    target_var = st.selectbox('Select target variable ', df.drop([entity, time], axis = 1).columns, key = st.session_state['key'])
                    
                    if df[target_var].dtypes == "int64" or df[target_var].dtypes == "float64": 
                        class_var_options = df.columns
                        class_var_options = class_var_options[class_var_options.isin(df.drop(target_var, axis = 1).columns)]
                        clas_var = st.selectbox('Select classifier variable ', [entity, time], key = st.session_state['key']) 

                        # Means and sd by entity 
                        col1, col2 = st.columns(2) 
                        with col1:
                            df_anova_woTime = df.drop([time], axis = 1)
                            df_grouped_ent = df_anova_woTime.groupby(entity)
                            st.write("Mean based on entity:")
                            st.write(df_grouped_ent.mean()[target_var])
                            st.write("")
                        with col2:
                            st.write("SD based on entity:")
                            st.write(df_grouped_ent.std()[target_var])
                            st.write("")

                        # Means and sd by time
                        col3, col4 = st.columns(2) 
                        with col3:
                            df_anova_woEnt= df.drop([entity], axis = 1)
                            df_grouped_time = df_anova_woEnt.groupby(time)
                            counts_time = pd.DataFrame(df_grouped_time.count()[target_var])
                            counts_time.columns = ["count"]
                            st.write("Mean based on time:")
                            st.write(df_grouped_time.mean()[target_var])
                            st.write("")
                        with col4:
                            st.write("SD based on time:")
                            st.write(df_grouped_time.std()[target_var])
                            st.write("")

                        col9, col10 = st.columns(2) 
                        with col9:
                            st.write("Boxplot grouped by entity:")
                            box_size1 = st.slider("Select box size", 1, 50, 5, key = st.session_state['key'])
                            # Grouped boxplot by entity
                            grouped_boxplot_data = pd.DataFrame()
                            grouped_boxplot_data[entity] = df[entity]
                            grouped_boxplot_data[time] = df[time]
                            grouped_boxplot_data["Index"] = df.index
                            grouped_boxplot_data[target_var] = df[target_var]
                            grouped_boxchart_ent = alt.Chart(grouped_boxplot_data, height = 300).mark_boxplot(size = box_size1, color = "#1f77b4", median = dict(color = "darkred")).encode(
                                x = alt.X(entity, scale = alt.Scale(zero = False)),
                                y = alt.Y(target_var, scale = alt.Scale(zero = False)), 
                                tooltip = [target_var, entity, time, "Index"]
                            ).configure_axis(
                                labelFontSize = 11,
                                titleFontSize = 12
                            )
                            st.altair_chart(grouped_boxchart_ent, use_container_width=True)
                        with col10:
                            st.write("Boxplot grouped by time:")
                            box_size2 = st.slider("Select box size ", 1, 50, 5, key = st.session_state['key'])
                            # Grouped boxplot by time
                            grouped_boxplot_data = pd.DataFrame()
                            grouped_boxplot_data[entity] = df[entity]
                            grouped_boxplot_data[time] = df[time]
                            grouped_boxplot_data["Index"] = df.index
                            grouped_boxplot_data[target_var] = df[target_var]
                            grouped_boxchart_time = alt.Chart(grouped_boxplot_data, height = 300).mark_boxplot(size = box_size2, color = "#1f77b4", median = dict(color = "darkred")).encode(
                                x = alt.X(time, scale = alt.Scale(domain = [min(df[time]), max(df[time])])),
                                y = alt.Y(target_var, scale = alt.Scale(zero = False)),
                                tooltip = [target_var, entity, time, "Index"]
                            ).configure_axis(
                                labelFontSize = 11,
                                titleFontSize = 12
                            )
                            st.altair_chart(grouped_boxchart_time, use_container_width=True)
                        if sett_hints:
                            st.info(str(fc.learning_hints("de_anova_boxplot")))
                        st.write("")
                        
                        # Count for entity and time
                        col5, col6 = st.columns(2)
                        with col5:
                            st.write("Number of observations per entity:")
                            counts_ent = pd.DataFrame(df_grouped_ent.count()[target_var])
                            counts_ent.columns = ["count"]
                            st.write(counts_ent.transpose())
                        with col6:
                            st.write("Number of observations per time:")
                            counts_time = pd.DataFrame(df_grouped_time.count()[target_var])
                            counts_time.columns = ["count"]
                            st.write(counts_time.transpose())  
                        if sett_hints:
                            st.info(str(fc.learning_hints("de_anova_count")))
                        st.write("")
                        
                        # ANOVA calculation
                        df_grouped = df[[target_var,clas_var]].groupby(clas_var)
                        overall_mean = (df_grouped.mean()*df_grouped.count()).sum()/df_grouped.count().sum()
                        dof_between = len(df_grouped.count())-1
                        dof_within = df_grouped.count().sum()-len(df_grouped.count())
                        dof_tot = dof_between + dof_within

                        SS_between = (((df_grouped.mean()-overall_mean)**2)*df_grouped.count()).sum()
                        SS_within =  (df_grouped.var()*(df_grouped.count()-1)).sum()
                        SS_total = SS_between + SS_within

                        MS_between = SS_between/dof_between
                        MS_within = SS_within/dof_within
                        F_stat = MS_between/MS_within
                        p_value = scipy.stats.f.sf(F_stat, dof_between, dof_within)

                        anova_table=pd.DataFrame({
                            "DF": [dof_between, dof_within.values[0], dof_tot.values[0]],
                            "SS": [SS_between.values[0], SS_within.values[0], SS_total.values[0]],
                            "MS": [MS_between.values[0], MS_within.values[0], ""],
                            "F-statistic": [F_stat.values[0], "", ""],
                            "p-value": [p_value[0], "", ""]},
                            index = ["Between", "Within", "Total"],)
                        
                        st.write("ANOVA:")
                        st.write(anova_table)
                        if sett_hints:
                            st.info(str(fc.learning_hints("de_anova_table")))    
                        st.write("")

                        #Anova (OLS)
                        codes = pd.factorize(df[clas_var])[0]
                        ano_ols = sm.OLS(df[target_var], sm.add_constant(codes))
                        ano_ols_output = ano_ols.fit()
                        residuals = ano_ols_output.resid
                    
                        col7, col8 = st.columns(2)
                        with col7:
                            # QQ-plot
                            st.write("Normal QQ-plot:")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            qq_plot_data = pd.DataFrame()
                            qq_plot_data["StandResiduals"] = (residuals - residuals.mean())/residuals.std()
                            qq_plot_data["Index"] = df.index
                            qq_plot_data[entity] = df[entity]
                            qq_plot_data[time] = df[time]
                            qq_plot_data = qq_plot_data.sort_values(by = ["StandResiduals"])
                            qq_plot_data["Theoretical quantiles"] = stats.probplot(residuals, dist="norm")[0][0]
                            qq_plot = alt.Chart(qq_plot_data, height = 300).mark_circle(size=20).encode(
                                x = alt.X("Theoretical quantiles", title = "theoretical quantiles", scale = alt.Scale(domain = [min(qq_plot_data["Theoretical quantiles"]), max(qq_plot_data["Theoretical quantiles"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                y = alt.Y("StandResiduals", title = "stand. residuals", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                tooltip = ["StandResiduals", "Theoretical quantiles", entity, time, "Index"]
                            )
                            line = alt.Chart(
                                pd.DataFrame({"Theoretical quantiles": [min(qq_plot_data["Theoretical quantiles"]), max(qq_plot_data["Theoretical quantiles"])], "StandResiduals": [min(qq_plot_data["Theoretical quantiles"]), max(qq_plot_data["Theoretical quantiles"])]})).mark_line(size = 2, color = "darkred").encode(
                                        alt.X("Theoretical quantiles"),
                                        alt.Y("StandResiduals"),
                            )
                            st.altair_chart(qq_plot + line, use_container_width = True)
                        with col8:
                            # Residuals histogram
                            st.write("Residuals histogram:")
                            residuals_hist = pd.DataFrame(residuals)
                            residuals_hist.columns = ["residuals"]
                            binNo_res = st.slider("Select maximum number of bins ", 5, 100, 25, key = st.session_state['key'])
                            hist_plot_res = alt.Chart(residuals_hist, height = 300).mark_bar().encode(
                                x = alt.X("residuals", title = "residuals", bin = alt.BinParams(maxbins = binNo_res), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                y = alt.Y("count()", title = "count of records", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                tooltip = ["count()", alt.Tooltip("residuals", bin = alt.BinParams(maxbins = binNo_res))]
                            ) 
                            st.altair_chart(hist_plot_res, use_container_width=True)
                        if sett_hints:
                            st.info(str(fc.learning_hints("de_anova_residuals"))) 
                        
                        # Download link for ANOVA statistics
                        output = BytesIO()
                        excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                        df_grouped_ent.mean()[target_var].to_excel(excel_file, sheet_name="entity_mean")
                        df_grouped_ent.std()[target_var].to_excel(excel_file, sheet_name="entity_sd")
                        df_grouped_time.mean()[target_var].to_excel(excel_file, sheet_name="time_mean")
                        df_grouped_time.std()[target_var].to_excel(excel_file, sheet_name="time_sd")
                        counts_ent.transpose().to_excel(excel_file, sheet_name="entity_obs")
                        counts_time.transpose().to_excel(excel_file, sheet_name="time_obs")
                        anova_table.to_excel(excel_file, sheet_name="ANOVA table")
                        excel_file.save()
                        excel_file = output.getvalue()
                        b64 = base64.b64encode(excel_file)
                        dl_file_name = "ANOVA statistics__" + target_var + "__" + df_name + ".xlsx"
                        st.markdown(
                            f"""
                        <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download ANOVA statistics</a>
                        """,
                        unsafe_allow_html=True)
                        st.write("")

                    else:
                        st.error("ERROR: The target variable must be a numerical one!")
                else: st.error("ERROR: No variables available for ANOVA!")

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
                    sb_DM_delCols = st.multiselect("Select columns to delete", df.drop([entity, time], axis = 1).columns, key = st.session_state['key'])
                    df = df.loc[:,~df.columns.isin(sb_DM_delCols)]

                    # Keep columns
                    sb_DM_keepCols = st.multiselect("Select columns to keep", df.drop([entity, time], axis = 1).columns, key = st.session_state['key'])
                    if len(sb_DM_keepCols) > 0:
                        df = df.loc[:,df.columns.isin([entity, time] + sb_DM_keepCols)]

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
                        if sb_DM_delRows_wNA == "No" and n_rows_wNAs > 0:
                            st.markdown("**Data imputation**")
                            sb_DM_dImp_choice = st.selectbox("Replace entries with NA", ["No", "Yes"], key = st.session_state['key'])
                            if sb_DM_dImp_choice == "Yes":
                                # Numeric variables
                                sb_DM_dImp_num = st.selectbox("Imputation method for numeric variables", ["Mean", "Median", "Random value"], key = st.session_state['key'])
                                # Other variables
                                sb_DM_dImp_other = st.selectbox("Imputation method for other variables", ["Mode", "Random value"], key = st.session_state['key'])
                                group_by_num = st.selectbox("Group imputation by", ["None", "Entity", "Time"], key = st.session_state['key'])
                                group_by_other = group_by_num
                                df = fc.data_impute_panel(df, sb_DM_dImp_num, sb_DM_dImp_other, group_by_num, group_by_other, entity, time)
                        else: 
                            st.markdown("**Data imputation**")
                            st.write("")
                            st.info("No NAs in data set!")
                
                with a3:
                    #--------------------------------------------------------------------------------------
                    # DATA TRANSFORMATION

                    st.markdown("**Data transformation**")
                    # Select columns for different transformation types
                    transform_options = df.drop([entity, time], axis = 1).select_dtypes([np.number]).columns
                    numCat_options = df.drop([entity, time], axis = 1).columns
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

                    data_transform=st.checkbox("Transform data in Excel?", value=False)
                    if data_transform==True:
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
                        <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Transform your data in Excel</a>
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
            if any(v for v in [sb_DM_delCols, sb_DM_dImp_num, sb_DM_dImp_other, sb_DM_dTrans_log, sb_DM_dTrans_sqrt, sb_DM_dTrans_square, sb_DM_dTrans_cent, sb_DM_dTrans_stand, sb_DM_dTrans_norm, sb_DM_dTrans_numCat ] if v is not None) or sb_DM_delDup == "Yes" or sb_DM_delRows_wNA == "Yes" or sb_DM_dTrans_mult != 0 or sb_DM_dTrans_div != 0 or filter_var != "-" or delRows!='-' or keepRows!='-' or len(sb_DM_keepCols) > 0:
                dev_expander_dsPost = st.expander("Explore cleaned and transformed panel data info and stats", expanded = False)
                with dev_expander_dsPost:
                    if df.shape[1] > 2 and df.shape[0] > 0:

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
                            st.write(df_summary_post["ALL"].style.set_precision(user_precision))

                            # Download link for cleaned data statistics
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            df.to_excel(excel_file, sheet_name="cleaned_data")
                            df_summary_post["Variable types"].to_excel(excel_file, sheet_name="cleaned_variable_info")
                            df_summary_post["ALL"].to_excel(excel_file, sheet_name="cleaned_summary_statistics")
                            excel_file.save()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "Cleaned data summary statistics_panel_" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download cleaned data summary statistics</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("")
                            
                            if fc.get_mode(df).loc["n_unique"].any():
                                st.caption("** Mode is not unique.") 
                            if sett_hints:
                                st.info(str(fc.learning_hints("de_summary_statistics")))
                    else: 
                        st.error("ERROR: No data available for preprocessing!") 
                        return

                dev_expander_anovPost = st.expander("ANOVA for cleaned and transformed panel data", expanded = False)
                with dev_expander_anovPost:
                    if df.shape[1] > 2 and df.shape[0] > 0:
                        
                        # Target variable
                        target_var2 = st.selectbox('Select target variable', df.drop([entity, time], axis = 1).columns)
                        
                        if df[target_var2].dtypes == "int64" or df[target_var2].dtypes == "float64": 
                            class_var_options = df.columns
                            class_var_options = class_var_options[class_var_options.isin(df.drop(target_var2, axis = 1).columns)]
                            clas_var2 = st.selectbox('Select classifier variable', [entity, time],) 

                            # Means and sd by entity
                            col1, col2 = st.columns(2) 
                            with col1:
                                df_anova_woTime = df.drop([time], axis = 1)
                                df_grouped_ent = df_anova_woTime.groupby(entity)
                                st.write("Mean based on entity:")
                                st.write(df_grouped_ent.mean()[target_var2])
                                st.write("")
                            with col2:
                                st.write("SD based on entity:")
                                st.write(df_grouped_ent.std()[target_var2])
                                st.write("")

                            # Means and sd by time
                            col3, col4 = st.columns(2) 
                            with col3:
                                df_anova_woEnt= df.drop([entity], axis = 1)
                                df_grouped_time = df_anova_woEnt.groupby(time)
                                counts_time = pd.DataFrame(df_grouped_time.count()[target_var2])
                                counts_time.columns = ["count"]
                                st.write("Mean based on time:")
                                st.write(df_grouped_time.mean()[target_var2])
                                st.write("")
                            with col4:
                                st.write("SD based on time:")
                                st.write(df_grouped_time.std()[target_var2])
                                st.write("")

                            col9, col10 = st.columns(2) 
                            with col9:
                                st.write("Boxplot grouped by entity:")
                                box_size1 = st.slider("Select box size  ", 1, 50, 5)
                                # Grouped boxplot by entity
                                grouped_boxplot_data = pd.DataFrame()
                                grouped_boxplot_data[entity] = df[entity]
                                grouped_boxplot_data[time] = df[time]
                                grouped_boxplot_data["Index"] = df.index
                                grouped_boxplot_data[target_var2] = df[target_var2]
                                grouped_boxchart_ent = alt.Chart(grouped_boxplot_data, height = 300).mark_boxplot(size = box_size1, color = "#1f77b4", median = dict(color = "darkred")).encode(
                                    x = alt.X(entity, scale = alt.Scale(zero = False)),
                                    y = alt.Y(target_var2, scale = alt.Scale(zero = False)),
                                    tooltip = [target_var2, entity, time, "Index"]
                                ).configure_axis(
                                    labelFontSize = 11,
                                    titleFontSize = 12
                                )
                                st.altair_chart(grouped_boxchart_ent, use_container_width=True)
                            with col10:
                                st.write("Boxplot grouped by time:")
                                box_size2 = st.slider("Select box size   ", 1, 50, 5)
                                # Grouped boxplot by time
                                grouped_boxplot_data = pd.DataFrame()
                                grouped_boxplot_data[time] = df[time]
                                grouped_boxplot_data[entity] = df[entity]
                                grouped_boxplot_data["Index"] = df.index
                                grouped_boxplot_data[target_var2] = df[target_var2]
                                grouped_boxchart_time = alt.Chart(grouped_boxplot_data, height = 300).mark_boxplot(size = box_size2, color = "#1f77b4", median = dict(color = "darkred")).encode(
                                    x = alt.X(time, scale = alt.Scale(domain = [min(df[time]), max(df[time])])),
                                    y = alt.Y(target_var2, scale = alt.Scale(zero = False)),
                                    tooltip = [target_var2, entity, time, "Index"]
                                ).configure_axis(
                                    labelFontSize = 11,
                                    titleFontSize = 12
                                )
                                st.altair_chart(grouped_boxchart_time, use_container_width=True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("de_anova_boxplot")))
                            st.write("")
                            
                            # Count for entity and time
                            col5, col6 = st.columns(2)
                            with col5:
                                st.write("Number of observations per entity:")
                                counts_ent = pd.DataFrame(df_grouped_ent.count()[target_var2])
                                counts_ent.columns = ["count"]
                                st.write(counts_ent.transpose())
                            with col6:
                                st.write("Number of observations per time:")
                                counts_time = pd.DataFrame(df_grouped_time.count()[target_var2])
                                counts_time.columns = ["count"]
                                st.write(counts_time.transpose()) 
                            if sett_hints:
                                st.info(str(fc.learning_hints("de_anova_count")))
                            st.write("")
                            
                            # ANOVA calculation
                            df_grouped = df[[target_var2,clas_var2]].groupby(clas_var2)
                            overall_mean = (df_grouped.mean()*df_grouped.count()).sum()/df_grouped.count().sum()
                            dof_between = len(df_grouped.count())-1
                            dof_within = df_grouped.count().sum()-len(df_grouped.count())
                            dof_tot = dof_between + dof_within

                            SS_between = (((df_grouped.mean()-overall_mean)**2)*df_grouped.count()).sum()
                            SS_within =  (df_grouped.var()*(df_grouped.count()-1)).sum()
                            SS_total = SS_between + SS_within

                            MS_between = SS_between/dof_between
                            MS_within = SS_within/dof_within
                            F_stat = MS_between/MS_within
                            p_value = scipy.stats.f.sf(F_stat, dof_between, dof_within)

                            anova_table=pd.DataFrame({
                                "DF": [dof_between, dof_within.values[0], dof_tot.values[0]],
                                "SS": [SS_between.values[0], SS_within.values[0], SS_total.values[0]],
                                "MS": [MS_between.values[0], MS_within.values[0], ""],
                                "F-statistic": [F_stat.values[0], "", ""],
                                "p-value": [p_value[0], "", ""]},
                                index = ["Between", "Within", "Total"],)
                            
                            st.write("ANOVA:")
                            st.write(anova_table)
                            if sett_hints:
                                st.info(str(fc.learning_hints("de_anova_table"))) 
                            st.write("")  

                            #Anova (OLS)
                            codes = pd.factorize(df[clas_var2])[0]
                            ano_ols = sm.OLS(df[target_var2], sm.add_constant(codes))
                            ano_ols_output = ano_ols.fit()
                            residuals = ano_ols_output.resid
                            
                            col7, col8 = st.columns(2)
                            with col7:
                                # QQ-plot
                                st.write("Normal QQ-plot:")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                qq_plot_data = pd.DataFrame()
                                qq_plot_data["StandResiduals"] = (residuals - residuals.mean())/residuals.std()
                                qq_plot_data["Index"] = df.index
                                qq_plot_data[entity] = df[entity]
                                qq_plot_data[time] = df[time]
                                qq_plot_data = qq_plot_data.sort_values(by = ["StandResiduals"])
                                qq_plot_data["Theoretical quantiles"] = stats.probplot(residuals, dist="norm")[0][0]
                                qq_plot = alt.Chart(qq_plot_data, height = 300).mark_circle(size=20).encode(
                                    x = alt.X("Theoretical quantiles", title = "theoretical quantiles", scale = alt.Scale(domain = [min(qq_plot_data["Theoretical quantiles"]), max(qq_plot_data["Theoretical quantiles"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("StandResiduals", title = "stand. residuals", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["StandResiduals", "Theoretical quantiles", entity, time, "Index"]
                                )
                                line = alt.Chart(
                                    pd.DataFrame({"Theoretical quantiles": [min(qq_plot_data["Theoretical quantiles"]), max(qq_plot_data["Theoretical quantiles"])], "StandResiduals": [min(qq_plot_data["Theoretical quantiles"]), max(qq_plot_data["Theoretical quantiles"])]})).mark_line(size = 2, color = "darkred").encode(
                                            alt.X("Theoretical quantiles"),
                                            alt.Y("StandResiduals"),
                                )
                                st.altair_chart(qq_plot + line, use_container_width = True)
                            with col8:
                                # Residuals histogram
                                st.write("Residuals histogram:")
                                residuals_hist = pd.DataFrame(residuals)
                                residuals_hist.columns = ["residuals"]
                                binNo_res2 = st.slider("Select maximum number of bins  ", 5, 100, 25)
                                hist_plot = alt.Chart(residuals_hist, height = 300).mark_bar().encode(
                                    x = alt.X("residuals", title = "residuals", bin = alt.BinParams(maxbins = binNo_res2), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("count()", title = "count of records", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["count()", alt.Tooltip("residuals", bin = alt.BinParams(maxbins = binNo_res2))]
                                ) 
                                st.altair_chart(hist_plot, use_container_width=True)  
                            if sett_hints:
                                st.info(str(fc.learning_hints("de_anova_residuals"))) 
                            
                            # Download link for ANOVA statistics
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            df_grouped_ent.mean()[target_var2].to_excel(excel_file, sheet_name="entity_mean")
                            df_grouped_ent.std()[target_var2].to_excel(excel_file, sheet_name="entity_sd")
                            df_grouped_time.mean()[target_var2].to_excel(excel_file, sheet_name="time_mean")
                            df_grouped_time.std()[target_var2].to_excel(excel_file, sheet_name="time_sd")
                            counts_ent.transpose().to_excel(excel_file, sheet_name="entity_obs")
                            counts_time.transpose().to_excel(excel_file, sheet_name="time_obs")
                            anova_table.to_excel(excel_file, sheet_name="ANOVA table")
                            excel_file.save()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "Cleaned ANOVA statistics__" + target_var2 + "__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download cleaned ANOVA statistics</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("")       
                        else:
                            st.error("ERROR: The target variable must be a numerical one!")
                    else: 
                        st.error("ERROR: No data available for ANOVA!") 
                        return
                
        #------------------------------------------------------------------------------------------
        
        #++++++++++++++++++++++
        # DATA VISUALIZATION

        data_visualization_container = st.container()
        with data_visualization_container:
            #st.write("")
            st.write("")
            st.write("")
            st.header("**Data visualization**")

            dev_expander_dv = st.expander("Explore visualization types", expanded = False)
            
            with dev_expander_dv:
                if df.shape[1] > 2 and df.shape[0] > 0:
                    st.write('**Variable selection**')
                    varl_sel_options = df.columns
                    varl_sel_options = varl_sel_options[varl_sel_options.isin(df.drop([entity, time], axis = 1).columns)]
                    var_sel = st.selectbox('Select variable for visualizations', varl_sel_options, key = st.session_state['key'])

                    if df[var_sel].dtypes == "float64" or df[var_sel].dtypes == "float32" or df[var_sel].dtypes == "int64" or df[var_sel].dtypes == "int32":
                        a4, a5 = st.columns(2)
                        with a4:
                            st.write('**Scatterplot with LOESS line**')
                            yy_options = df.columns
                            yy_options = yy_options[yy_options.isin(df.drop([entity, time], axis = 1).columns)]
                            yy = st.selectbox('Select variable for y-axis', yy_options, key = st.session_state['key'])
                            if df[yy].dtypes == "float64" or df[yy].dtypes == "float32" or df[yy].dtypes == "int64" or df[yy].dtypes == "int32":
                                fig_data = pd.DataFrame()
                                fig_data[yy] = df[yy]
                                fig_data[var_sel] = df[var_sel]
                                fig_data["Index"] = df.index
                                fig_data[entity] = df[entity]
                                fig_data[time] = df[time]
                                fig = alt.Chart(fig_data).mark_circle().encode(
                                    x = alt.X(var_sel, scale = alt.Scale(domain = [min(fig_data[var_sel]), max(fig_data[var_sel])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y(yy, scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = [yy, var_sel, entity, time, "Index"]
                                )
                                st.altair_chart(fig + fig.transform_loess(var_sel, yy).mark_line(size = 2, color = "darkred"), use_container_width=True)
                                if sett_hints:
                                    st.info(str(fc.learning_hints("dv_scatterplot")))
                            else: st.error("ERROR: Please select a numeric variable for the y-axis!")   
                        with a5:
                            st.write('**Histogram**')
                            binNo = st.slider("Select maximum number of bins", 5, 100, 25, key = st.session_state['key'])
                            fig2 = alt.Chart(df).mark_bar().encode(
                                x = alt.X(var_sel, title = var_sel + " (binned)", bin = alt.BinParams(maxbins = binNo), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                y = alt.Y("count()", title = "count of records", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                tooltip = ["count()", alt.Tooltip(var_sel, bin = alt.BinParams(maxbins = binNo))]
                            )
                            st.altair_chart(fig2, use_container_width=True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("dv_histogram")))

                        a6, a7 = st.columns(2)
                        with a6:
                            st.write('**Boxplot**')
                            # Boxplot
                            boxplot_data = pd.DataFrame()
                            boxplot_data[var_sel] = df[var_sel]
                            boxplot_data["Index"] = df.index
                            boxplot_data[entity] = df[entity]
                            boxplot_data[time] = df[time]
                            boxplot = alt.Chart(boxplot_data).mark_boxplot(size = 100, color = "#1f77b4", median = dict(color = "darkred")).encode(
                                y = alt.Y(var_sel, scale = alt.Scale(zero = False)),
                                tooltip = [var_sel, entity, time, "Index"]
                            ).configure_axis(
                                labelFontSize = 11,
                                titleFontSize = 12
                            )
                            st.altair_chart(boxplot, use_container_width=True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("dv_boxplot")))
                        with a7:
                            st.write("**QQ-plot**")
                            var_values = df[var_sel]
                            qqplot_data = pd.DataFrame()
                            qqplot_data[var_sel] = var_values
                            qqplot_data["Index"] = df.index
                            qqplot_data[entity] = df[entity]
                            qqplot_data[time] = df[time]
                            qqplot_data = qqplot_data.sort_values(by = [var_sel])
                            qqplot_data["Theoretical quantiles"] = stats.probplot(var_values, dist="norm")[0][0]
                            qqplot = alt.Chart(qqplot_data).mark_circle(size=20).encode(
                                x = alt.X("Theoretical quantiles", title = "theoretical quantiles", scale = alt.Scale(domain = [min(qqplot_data["Theoretical quantiles"]), max(qqplot_data["Theoretical quantiles"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                y = alt.Y(var_sel, title = str(var_sel), scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                tooltip = [var_sel, "Theoretical quantiles", entity, time, "Index"]
                            )
                            st.altair_chart(qqplot + qqplot.transform_regression('Theoretical quantiles', var_sel).mark_line(size = 2, color = "darkred"), use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("dv_qqplot")))
                    else: st.error("ERROR: Please select a numeric variable!") 
                else: st.error("ERROR: No data available for Data Visualization!")  

            # Check again after processing
            if np.where(df[entity].isnull())[0].size > 0:
                    entity_na_warn = "WARNING: The variable selected for entity has NAs!"
            else:entity_na_warn = False
            if np.where(df[time].isnull())[0].size > 0:
                    time_na_warn = "WARNING: The variable selected for time has NAs!"
            else:time_na_warn = False

        #------------------------------------------------------------------------------------------

        #++++++++++++++++++++++++++++++++++++++++++++
        # PANEL DATA MODELLING

        data_modelling_container = st.container()
        with data_modelling_container:
            #st.write("")
            #st.write("")
            #st.write("")
            st.write("")
            st.write("")
            st.header("**Panel data modelling**")
            st.markdown("Go for creating predictive models of your panel data using panel data modelling!  STATY will take care of the modelling for you, so you can put your focus on results interpretation and communication! ")

            PDM_settings = st.expander("Specify model", expanded = False)
            with PDM_settings:
                
                if time_na_warn == False and entity_na_warn == False:

                    # Initial status for running models
                    model_full_results = None
                    do_modval = "No"
                    model_val_results = None
                    model_full_results = None
                    panel_model_fit = None

                    if df.shape[1] > 2 and df.shape[0] > 0:

                        #--------------------------------------------------------------------------------------
                        # GENERAL SETTINGS
                    
                        st.markdown("**Variable selection**")

                        # Variable categories
                        df_summary_model = fc.data_summary(df)
                        var_cat = df_summary_model["Variable types"].loc["category"]

                        # Response variable
                        response_var_options = df.columns
                        response_var_options = response_var_options[response_var_options.isin(df.drop(entity, axis = 1).columns)]
                        if time != "NA":
                            response_var_options = response_var_options[response_var_options.isin(df.drop(time, axis = 1).columns)]
                        response_var = st.selectbox("Select response variable", response_var_options, key = st.session_state['key'])

                        # Check if response variable is numeric and has no NAs
                        response_var_message_num = False
                        response_var_message_na = False
                        response_var_message_cat = False

                        if var_cat.loc[response_var] == "string/binary" or var_cat.loc[response_var] == "bool/binary":
                            response_var_message_num = "ERROR: Please select a numeric response variable!"
                        elif var_cat.loc[response_var] == "string/categorical" or var_cat.loc[response_var] == "other" or var_cat.loc[response_var] == "string/single":
                            response_var_message_num = "ERROR: Please select a numeric response variable!"
                        elif var_cat.loc[response_var] == "categorical":
                            response_var_message_cat = "WARNING: Categorical variable is treated as continuous variable!"

                        if response_var_message_num != False:
                            st.error(response_var_message_num)
                        if response_var_message_na != False:
                            st.error(response_var_message_na)
                        if response_var_message_cat != False:
                            st.warning(response_var_message_cat)

                        # Continue if everything is clean for response variable
                        if response_var_message_num == False and response_var_message_na == False:
                            # Select explanatory variables
                            expl_var_options = response_var_options[response_var_options.isin(df.drop(response_var, axis = 1).columns)]
                            expl_var = st.multiselect("Select explanatory variables", expl_var_options, key = st.session_state['key'])
                            var_list = list([entity]) + list([time]) + list([response_var]) + list(expl_var)

                            # Check if explanatory variables are numeric
                            expl_var_message_num = False
                            expl_var_message_na = False
                            if any(a for a in df[expl_var].dtypes if a != "float64" and a != "float32" and a != "int64" and a != "int64"): 
                                expl_var_not_num = df[expl_var].select_dtypes(exclude=["int64", "int32", "float64", "float32"]).columns
                                expl_var_message_num = "ERROR: Please exclude non-numeric variables: " + ', '.join(map(str,list(expl_var_not_num)))
                            
                            # Check if NAs are present and delete them automatically (delete before run models button)
                            if np.where(df[var_list].isnull())[0].size > 0:
                                st.warning("WARNING: Your modelling data set includes NAs. Rows with NAs are automatically deleted!")

                            if expl_var_message_num != False:
                                st.error(expl_var_message_num)
                            elif expl_var_message_na != False:
                                st.error(expl_var_message_na)
                                
                            # Continue if everything is clean for explanatory variables and at least one was selected
                            elif expl_var_message_num == False and expl_var_message_na == False and len(expl_var) > 0:
                            
                                #--------------------------------------------------------------------------------------
                                # ALGORITHMS

                                st.markdown("**Specify modelling algorithm**")

                                # Algorithms selection
                                col1, col2 = st.columns(2)
                                algorithms = ["Entity Fixed Effects", "Time Fixed Effects", "Two-ways Fixed Effects", "Random Effects", "Pooled"]
                                with col1:
                                    PDM_alg = st.selectbox("Select modelling technique", algorithms)
                                
                                # Covariance type
                                with col2:
                                    PDM_cov_type = st.selectbox("Select covariance type", ["homoskedastic", "heteroskedastic", "clustered"])
                                    PDM_cov_type2 = None 
                                    if PDM_cov_type == "clustered":
                                        PDM_cov_type2 = st.selectbox("Select cluster type", ["entity", "time", "both"])

                                #--------------------------------------------------------------------------------------
                                # VALIDATION SETTINGS

                                st.markdown("**Validation settings**")

                                do_modval= st.selectbox("Use model validation", ["No", "Yes"])

                                if do_modval == "Yes":
                                    col1, col2 = st.columns(2)
                                    # Select training/ test ratio 
                                    with col1:
                                        train_frac = st.slider("Select training data size", 0.5, 0.95, 0.8)

                                    # Select number for validation runs
                                    with col2:
                                        val_runs = st.slider("Select number for validation runs", 5, 100, 10)

                                #--------------------------------------------------------------------------------------
                                # PREDICTION SETTINGS

                                st.markdown("**Model predictions**")
                                do_modprednew = st.selectbox("Use model prediction for new data", ["No", "Yes"])

                                if do_modprednew == "No":
                                    df_new = pd.DataFrame()
                                if do_modprednew == "Yes":
                                    # Upload new data
                                    new_data_pred = st.file_uploader("  ", type=["csv", "txt"])

                                    if new_data_pred is not None:

                                        # Read data
                                        if uploaded_data is not None:
                                            df_new = pd.read_csv(new_data_pred, decimal=dec_sep, sep = col_sep,thousands=thousands_sep,encoding=encoding_val, engine='python')
                                        else:
                                            df_new = pd.read_csv(new_data_pred, sep = ";|,|\t",engine='python')
                                        st.success('Loading data... done!')

                                        # Transform columns if any were transformed
                                        # Log-transformation
                                        if sb_DM_dTrans_log is not None:

                                            # List of log-transformed variables that are included as explanatory variables
                                            tv_list = []
                                            for tv in sb_DM_dTrans_log:
                                                if "log_"+tv in expl_var:
                                                    tv_list.append(tv)
                                            
                                            # Check if log-transformed explanatory variables are available for transformation in new data columns
                                            tv_list_not_avail = []
                                            if tv_list:
                                                for tv in tv_list:
                                                    if tv not in df_new.columns:
                                                        tv_list_not_avail.append(tv)
                                                if tv_list_not_avail:
                                                    st.error("ERROR: Some variables are not available for log-transformation in new data: "+ ', '.join(tv_list_not_avail))
                                                    return
                                                else:
                                                    # Transform data if variables for transformation are all available in new data   
                                                    df_new = fc.var_transform_log(df_new, tv_list)

                                        # Sqrt-transformation
                                        if sb_DM_dTrans_sqrt is not None:

                                            # List of sqrt-transformed variables that are included as explanatory variables
                                            tv_list = []
                                            for tv in sb_DM_dTrans_sqrt:
                                                if "sqrt_"+tv in expl_var:
                                                    tv_list.append(tv)
                                            
                                            # Check if sqrt-transformed explanatory variables are available for transformation in new data columns
                                            tv_list_not_avail = []
                                            if tv_list:
                                                for tv in tv_list:
                                                    if tv not in df_new.columns:
                                                        tv_list_not_avail.append(tv)
                                                if tv_list_not_avail:
                                                    st.error("ERROR: Some variables are not available for sqrt-transformation in new data: "+ ', '.join(tv_list_not_avail))
                                                    return
                                                else:
                                                    # Transform data if variables for transformation are all available in new data   
                                                    df_new = fc.var_transform_sqrt(df_new, tv_list)
                                        
                                        # Square-transformation
                                        if sb_DM_dTrans_square is not None:

                                            # List of square-transformed variables that are included as explanatory variables
                                            tv_list = []
                                            for tv in sb_DM_dTrans_square:
                                                if "square_"+tv in expl_var:
                                                    tv_list.append(tv)
                                            
                                            # Check if square-transformed explanatory variables are available for transformation in new data columns
                                            tv_list_not_avail = []
                                            if tv_list:
                                                for tv in tv_list:
                                                    if tv not in df_new.columns:
                                                        tv_list_not_avail.append(tv)
                                                if tv_list_not_avail:
                                                    st.error("ERROR: Some variables are not available for square-transformation in new data: "+ ', '.join(tv_list_not_avail))
                                                    return
                                                else:
                                                    # Transform data if variables for transformation are all available in new data   
                                                    df_new = fc.var_transform_square(df_new, tv_list)

                                        # Standardization
                                        if sb_DM_dTrans_stand is not None:

                                            # List of standardized variables that are included as explanatory variables
                                            tv_list = []
                                            for tv in sb_DM_dTrans_stand:
                                                if "stand_"+tv in expl_var:
                                                    tv_list.append(tv) 
                                            
                                            # Check if standardized explanatory variables are available for transformation in new data columns
                                            tv_list_not_avail = []
                                            if tv_list:
                                                for tv in tv_list:
                                                    if tv not in df_new.columns:
                                                        tv_list_not_avail.append(tv)
                                                if tv_list_not_avail:
                                                    st.error("ERROR: Some variables are not available for standardization in new data: "+ ', '.join(tv_list_not_avail))
                                                    return
                                                else:
                                                    # Transform data if variables for transformation are all available in new data
                                                    # Use mean and standard deviation of original data for standardization
                                                    for tv in tv_list:
                                                        if df_new[tv].dtypes == "float64" or df_new[tv].dtypes == "int64" or df_new[tv].dtypes == "float32" or df_new[tv].dtypes == "int32":
                                                            if df[tv].std() != 0:
                                                                new_var_name = "stand_" + tv
                                                                new_var = (df_new[tv] - df[tv].mean())/df[tv].std()
                                                                df_new[new_var_name] = new_var
                                                        else:
                                                            st.error("ERROR: " + str(tv) + " is not numerical and cannot be standardized!")
                                                            return    

                                        # Normalization
                                        if sb_DM_dTrans_norm is not None:

                                            # List of normalized variables that are included as explanatory variables
                                            tv_list = []
                                            for tv in sb_DM_dTrans_norm:
                                                if "norm_"+tv in expl_var:
                                                    tv_list.append(tv) 

                                            # Check if normalized explanatory variables are available for transformation in new data columns
                                            tv_list_not_avail = []
                                            if tv_list:
                                                for tv in tv_list:
                                                    if tv not in df_new.columns:
                                                        tv_list_not_avail.append(tv)
                                                if tv_list_not_avail:
                                                    st.error("ERROR: Some variables are not available for normalization in new data: "+ ', '.join(tv_list_not_avail))
                                                    return
                                                else:
                                                    # Transform data if variables for transformation are all available in new data 
                                                    # Use min and max of original data for normalization
                                                    for tv in tv_list:
                                                        if df_new[tv].dtypes == "float64" or df_new[tv].dtypes == "int64" or df_new[tv].dtypes == "float32" or df_new[tv].dtypes == "int32":
                                                            if (df[tv].max()-df[tv].min()) != 0:
                                                                new_var_name = "norm_" + tv
                                                                new_var = (df_new[tv] - df[tv].min())/(df[tv].max()-df[tv].min())
                                                                df_new[new_var_name] = new_var 
                                                        else:
                                                            st.error("ERROR: " + str(tv) + " is not numerical and cannot be normalized!")
                                                            return  
                                        
                                        # Categorization
                                        if sb_DM_dTrans_numCat is not None: 

                                            # List of categorized variables that are included as explanatory variables
                                            tv_list = []
                                            for tv in sb_DM_dTrans_numCat:
                                                if "numCat_"+tv in expl_var:
                                                    tv_list.append(tv) 
                                            
                                            # Check if categorized explanatory variables are available for transformation in new data columns
                                            tv_list_not_avail = []
                                            if tv_list:
                                                for tv in tv_list:
                                                    if tv not in df_new.columns:
                                                        tv_list_not_avail.append(tv)
                                                if tv_list_not_avail:
                                                    st.error("ERROR: Some variables are not available for categorization in new data: "+ ', '.join(tv_list_not_avail))
                                                    return
                                                else:
                                                    # Transform data if variables for transformation are all available in new data 
                                                    # Use same categories as for original data
                                                    for tv in tv_list:
                                                        new_var_name = "numCat_" + tv
                                                        new_var = pd.DataFrame(index = df_new.index, columns = [new_var_name])
                                                        for r in df_new.index:
                                                            if df.loc[df[tv] == df_new[tv][r]].empty == False:
                                                                new_var.loc[r, new_var_name] = df["numCat_" + tv][df.loc[df[tv] == df_new[tv][r]].index[0]]
                                                            else:
                                                                st.error("ERROR: Category is missing for the value in row: "+ str(r) + ", variable: " + str(tv))
                                                                return
                                                        df_new[new_var_name] = new_var.astype('int64')
                                        
                                        # Multiplication
                                        if sb_DM_dTrans_mult != 0:

                                            # List of multiplied variables that are included as explanatory variables
                                            tv_list = []
                                            for tv in range(0, sb_DM_dTrans_mult):
                                                mult_name = "mult_" + str(multiplication_pairs.loc[tv]["Var1"]) + "_" + str(multiplication_pairs.loc[tv]["Var2"])
                                                if mult_name in expl_var:
                                                    tv_list.append(str(multiplication_pairs.loc[tv]["Var1"]))
                                                    tv_list.append(str(multiplication_pairs.loc[tv]["Var2"]))
                                            
                                            # Check if multiplied explanatory variables are available for transformation in new data columns
                                            tv_list_not_avail = []
                                            if tv_list:
                                                for tv in tv_list:
                                                    if tv not in df_new.columns:
                                                        tv_list_not_avail.append(tv)
                                                if tv_list_not_avail:
                                                    st.error("ERROR: Some variables are not available for multiplication in new data: "+ ', '.join(tv_list_not_avail))
                                                    return
                                                else:
                                                    # Transform data if variables for transformation are all available in new data 
                                                    for var in range(0, sb_DM_dTrans_mult):  
                                                        df_new = fc.var_transform_mult(df_new, multiplication_pairs.loc[var]["Var1"], multiplication_pairs.loc[var]["Var2"])

                                        # Division
                                        if sb_DM_dTrans_div != 0:

                                            # List of divided variables that are included as explanatory variables
                                            tv_list = []
                                            for tv in range(0, sb_DM_dTrans_div):
                                                mult_name = "div_" + str(division_pairs.loc[tv]["Var1"]) + "_" + str(division_pairs.loc[tv]["Var2"])
                                                if mult_name in expl_var:
                                                    tv_list.append(str(division_pairs.loc[tv]["Var1"]))
                                                    tv_list.append(str(division_pairs.loc[tv]["Var2"]))
                                            
                                        # Check if multiplied explanatory variables are available for transformation in new data columns
                                        tv_list_not_avail = []
                                        if tv_list:
                                            for tv in tv_list:
                                                if tv not in df_new.columns:
                                                    tv_list_not_avail.append(tv)
                                            if tv_list_not_avail:
                                                st.error("ERROR: Some variables are not available for division in new data: "+ ', '.join(tv_list_not_avail))
                                                return
                                            else:
                                                # Transform data if variables for transformation are all available in new data 
                                                for var in range(0, sb_DM_dTrans_div):  
                                                    df_new = fc.var_transform_div(df_new, division_pairs.loc[var]["Var1"], division_pairs.loc[var]["Var2"])

                                        # Check if explanatory variables are available as columns as well as entity and time
                                        expl_list = []
                                        for expl_incl in expl_var:
                                            if expl_incl not in df_new.columns:
                                                expl_list.append(expl_incl)
                                        if expl_list:
                                            st.error("ERROR: Some variables are missing in new data: "+ ', '.join(expl_list))
                                            return
                                        if any(a for a in df_new.columns if a == entity) and any(a for a in df_new.columns if a == time):
                                            st.info("All variables are available for predictions!")
                                        elif any(a for a in df_new.columns if a == entity) == False:
                                            st.error("ERROR: Entity variable is missing!")
                                            return
                                        elif any(a for a in df_new.columns if a == time) == False:
                                            st.error("ERROR: Time variable is missing!")
                                            return
                                        
                                        # Check if NAs are present
                                        if df_new.iloc[list(pd.unique(np.where(df_new.isnull())[0]))].shape[0] == 0:
                                            st.empty()
                                        else:
                                            df_new = df_new[list([entity]) + list([time]) + expl_var].dropna()
                                            st.warning("WARNING: Your new data set includes NAs. Rows with NAs are automatically deleted!")
                                        df_new = df_new[list([entity]) + list([time]) + expl_var]
                                
                                # Modelling data set
                                df = df[var_list]

                                # Check if NAs are present and delete them automatically
                                if np.where(df[var_list].isnull())[0].size > 0:
                                    df = df.dropna()
                                        
                                #--------------------------------------------------------------------------------------
                                # SETTINGS SUMMARY

                                st.write("")
                                # Show modelling data
                                if st.checkbox("Show modelling data"):
                                    st.write(df)
                                    st.write("Data shape: ", df.shape[0],  " rows and ", df.shape[1], " columns")
                                    
                                    # Download link for modelling data
                                    output = BytesIO()
                                    excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                                    df.to_excel(excel_file, sheet_name="modelling_data")
                                    excel_file.save()
                                    excel_file = output.getvalue()
                                    b64 = base64.b64encode(excel_file)
                                    dl_file_name= "Modelling data__" + df_name + ".xlsx"
                                    st.markdown(
                                        f"""
                                    <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download modelling data</a>
                                    """,
                                    unsafe_allow_html=True)
                                st.write("")

                                # Show prediction data
                                if do_modprednew == "Yes":
                                    if new_data_pred is not None:
                                        if st.checkbox("Show new data for predictions"):
                                            st.write(df_new)
                                            st.write("Data shape: ", df_new.shape[0],  " rows and ", df_new.shape[1], " columns")

                                            # Download link for forecast data
                                            output = BytesIO()
                                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                                            df_new.to_excel(excel_file, sheet_name="new_data")
                                            excel_file.save()
                                            excel_file = output.getvalue()
                                            b64 = base64.b64encode(excel_file)
                                            dl_file_name= "New data for predictions__" + df_name + ".xlsx"
                                            st.markdown(
                                                f"""
                                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download new data for predictions</a>
                                            """,
                                            unsafe_allow_html=True)
                                        st.write("")

                                # Show modelling settings
                                if st.checkbox('Show a summary of modelling settings', value = False): 
                                    
                                    #--------------------------------------------------------------------------------------
                                    # ALOGRITHMS
                                    
                                    st.write("Algorithms summary:")
                                    st.write("- ",PDM_alg)
                                    st.write("- Covariance type: ", PDM_cov_type)
                                    if PDM_cov_type2 is not None:
                                        st.write("- Cluster type: ", PDM_cov_type2)
                                    st.write("")

                                    #--------------------------------------------------------------------------------------
                                    # SETTINGS

                                    # General settings summary
                                    st.write("General settings summary:")
                                    # Modelling formula
                                    if expl_var != False:
                                        st.write("- Modelling formula:", response_var, "~",  ' + '.join(expl_var))
                                        st.write("- Entity:", entity)
                                        st.write("- Time:", time)
                                    if do_modval == "Yes":
                                        # Train/ test ratio
                                        if train_frac != False:
                                            st.write("- Train/ test ratio:", str(round(train_frac*100)), "% / ", str(round(100-train_frac*100)), "%")
                                        # Validation runs
                                        if val_runs != False:
                                            st.write("- Validation runs:", str(val_runs))
                                    st.write("")
                                    st.write("")
                                
                                #--------------------------------------------------------------------------------------
                                # RUN MODELS

                                # Models are run on button click
                                st.write("")
                                run_models = st.button("Run model")
                                st.write("")

                                # Run everything on button click
                                if run_models:

                                    # Check if new data available
                                    if do_modprednew == "Yes":
                                        if new_data_pred is None:
                                            st.error("ERROR: Please upload new data for additional model predictions or select 'No'!")
                                            return

                                    # Define clustered cov matrix "entity", "time", "both"
                                    cluster_entity = True
                                    cluster_time = False
                                    if PDM_cov_type == "clustered":
                                        if PDM_cov_type2 == "entity":
                                            cluster_entity = True
                                            cluster_time = False
                                        if PDM_cov_type2 == "time":
                                            cluster_entity = False
                                            cluster_time = True
                                        if PDM_cov_type2 == "both":
                                            cluster_entity = True
                                            cluster_time = True

                                    # Prepare data
                                    data = df.set_index([entity, time])
                                    Y_data = data[response_var]
                                    X_data1 = data[expl_var] # for efe, tfe, twfe
                                    X_data2 = sm.add_constant(data[expl_var]) # for re, pool

                                    # Model validation
                                    if do_modval == "Yes":

                                        # Progress bar
                                        st.info("Validation progress")
                                        my_bar = st.progress(0.0)
                                        progress1 = 0

                                        # Model validation
                                        # R²
                                        model_eval_r2 = pd.DataFrame(index = range(val_runs), columns = [response_var])
                                        # MSE
                                        model_eval_mse = pd.DataFrame(index = range(val_runs), columns = ["Value"])
                                        # RMSE
                                        model_eval_rmse = pd.DataFrame(index = range(val_runs), columns = ["Value"])
                                        # MAE
                                        model_eval_mae = pd.DataFrame(index = range(val_runs), columns = ["Value"])
                                        # MaxERR
                                        model_eval_maxerr = pd.DataFrame(index = range(val_runs), columns = ["Value"])
                                        # EVRS
                                        model_eval_evrs = pd.DataFrame(index = range(val_runs), columns = ["Value"])
                                        # SSR
                                        model_eval_ssr = pd.DataFrame(index = range(val_runs), columns = ["Value"])

                                        # Model validation summary
                                        model_eval_mean = pd.DataFrame(index = ["% VE", "MSE", "RMSE", "MAE", "MaxErr", "EVRS", "SSR"], columns = ["Value"])
                                        model_eval_sd = pd.DataFrame(index = ["% VE", "MSE", "RMSE", "MAE", "MaxErr", "EVRS", "SSR"], columns = ["Value"])

                                        # Collect all residuals in test runs
                                        resdiuals_allruns = {}

                                        for val in range(val_runs):
                                            
                                            # Split data into train/ test data
                                            if PDM_alg != "Pooled" and PDM_alg != "Random Effects":
                                                X_data = X_data1.copy()
                                            if PDM_alg == "Pooled" or PDM_alg == "Random Effects":
                                                X_data = X_data2.copy()
                                            X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, train_size = train_frac, random_state = val)

                                            # Train selected panel model
                                            # efe
                                            if PDM_alg == "Entity Fixed Effects":
                                                panel_model_efe_val = PanelOLS(Y_train, X_train, entity_effects = True, time_effects = False)
                                                panel_model_fit_efe_val = panel_model_efe_val.fit(cov_type = PDM_cov_type, cluster_entity = cluster_entity, cluster_time = cluster_time, debiased = True, auto_df = True) 
                                            # tfe
                                            if PDM_alg == "Time Fixed Effects":
                                                panel_model_tfe_val = PanelOLS(Y_train, X_train, entity_effects = False, time_effects = True)
                                                panel_model_fit_tfe_val = panel_model_tfe_val.fit(cov_type = PDM_cov_type, cluster_entity = cluster_entity, cluster_time = cluster_time, debiased = True, auto_df = True) 
                                            # twfe
                                            if PDM_alg == "Two-ways Fixed Effects":
                                                panel_model_twfe_val = PanelOLS(Y_train, X_train, entity_effects = True, time_effects = True)
                                                panel_model_fit_twfe_val = panel_model_twfe_val.fit(cov_type = PDM_cov_type, cluster_entity = cluster_entity, cluster_time = cluster_time, debiased = True, auto_df = True) 
                                            # re
                                            if PDM_alg == "Random Effects":
                                                panel_model_re_val = RandomEffects(Y_train, X_train)
                                                panel_model_fit_re_val = panel_model_re_val.fit(cov_type = PDM_cov_type, cluster_entity = cluster_entity, cluster_time = cluster_time, debiased = True, auto_df = True)
                                            # pool
                                            if PDM_alg == "Pooled":
                                                panel_model_pool_val = PooledOLS(Y_train, X_train)
                                                panel_model_fit_pool_val = panel_model_pool_val.fit(cov_type = PDM_cov_type, cluster_entity = cluster_entity, cluster_time = cluster_time, debiased = True, auto_df = True)                   
                                            # save selected model
                                            if PDM_alg == "Entity Fixed Effects":
                                                panel_model_fit_val = panel_model_fit_efe_val
                                            if PDM_alg == "Time Fixed Effects":
                                                panel_model_fit_val = panel_model_fit_tfe_val
                                            if PDM_alg == "Two-ways Fixed Effects":
                                                panel_model_fit_val = panel_model_fit_twfe_val
                                            if PDM_alg == "Random Effects":
                                                panel_model_fit_val = panel_model_fit_re_val
                                            if PDM_alg == "Pooled":
                                                panel_model_fit_val = panel_model_fit_pool_val
                                            
                                            # Extract effects
                                            if PDM_alg != "Pooled":
                                                comb_effects = panel_model_fit_val.estimated_effects
                                            ent_effects = pd.DataFrame(index = X_train.reset_index()[entity].drop_duplicates(), columns = ["Value"])
                                            time_effects = pd.DataFrame(index = sorted(list(X_train.reset_index()[time].drop_duplicates())), columns = ["Value"])

                                            # Use LSDV for estimating effects
                                            if PDM_alg == "Entity Fixed Effects":
                                                X_train_mlr = pd.concat([X_train.reset_index(drop = True), pd.get_dummies(X_train.reset_index()[entity])], axis = 1)
                                                Y_train_mlr = Y_train.reset_index(drop = True)
                                                model_mlr_val = sm.OLS(Y_train_mlr, X_train_mlr)
                                                model_mlr_fit_val = model_mlr_val.fit()
                                                for e in ent_effects.index:
                                                    ent_effects.loc[e]["Value"] = model_mlr_fit_val.params[e]
                                                for t in time_effects.index:
                                                    time_effects.loc[t]["Value"] = 0
                                            if PDM_alg == "Time Fixed Effects":
                                                X_train_mlr = pd.concat([X_train.reset_index(drop = True), pd.get_dummies(X_train.reset_index()[time])], axis = 1)
                                                Y_train_mlr = Y_train.reset_index(drop = True)
                                                model_mlr_val = sm.OLS(Y_train_mlr, X_train_mlr)
                                                model_mlr_fit_val = model_mlr_val.fit()
                                                for e in ent_effects.index:
                                                    ent_effects.loc[e]["Value"] = 0
                                                for t in time_effects.index:
                                                    time_effects.loc[t]["Value"] = model_mlr_fit_val.params[t]
                                            if PDM_alg == "Two-ways Fixed Effects":
                                                X_train_mlr = pd.concat([X_train.reset_index(drop = True), pd.get_dummies(X_train.reset_index()[entity]), pd.get_dummies(X_train.reset_index()[time])], axis = 1)
                                                Y_train_mlr = Y_train.reset_index(drop = True)
                                                model_mlr_val = sm.OLS(Y_train_mlr, X_train_mlr)
                                                model_mlr_fit_val = model_mlr_val.fit()
                                                for e in ent_effects.index:
                                                    ent_effects.loc[e]["Value"] = model_mlr_fit_val.params[e]
                                                for t in time_effects.index:
                                                    time_effects.loc[t]["Value"] = model_mlr_fit_val.params[t]
                                            if PDM_alg == "Random Effects":
                                                for e in ent_effects.index:
                                                    ent_effects.loc[e]["Value"] = comb_effects.loc[e,].reset_index(drop = True).iloc[0][0]
                                            
                                            # Prediction for Y_test (without including effects)
                                            Y_test_pred = panel_model_fit_val.predict(X_test)

                                            # Add effects for predictions
                                            for p in range(Y_test_pred.size):
                                                
                                                entity_ind = Y_test_pred.index[p][0]
                                                time_ind = Y_test_pred.index[p][1]
                                                
                                                # if effects are available, add effect
                                                if PDM_alg == "Entity Fixed Effects":
                                                    if any(a for a in ent_effects.index if a == entity_ind):
                                                        effect = ent_effects.loc[entity_ind][0]
                                                        Y_test_pred["predictions"].loc[entity_ind, time_ind] = Y_test_pred["predictions"].loc[entity_ind, time_ind] + effect
                                                if PDM_alg == "Time Fixed Effects":
                                                    if any(a for a in time_effects.index if a == time_ind):
                                                        effect = time_effects.loc[time_ind][0]
                                                        Y_test_pred["predictions"].loc[entity_ind, time_ind] = Y_test_pred["predictions"].loc[entity_ind, time_ind] + effect
                                                if PDM_alg == "Two-ways Fixed Effects":
                                                    if any(a for a in time_effects.index if a == time_ind):
                                                        effect_time = time_effects.loc[time_ind][0]
                                                    else: effect_time = 0
                                                    if any(a for a in ent_effects.index if a == entity_ind):
                                                        effect_entity = ent_effects.loc[entity_ind][0]
                                                    else: effect_entity = 0    
                                                    Y_test_pred["predictions"].loc[entity_ind, time_ind] = Y_test_pred["predictions"].loc[entity_ind, time_ind] + effect_entity + effect_time
                                                if PDM_alg == "Random Effects":
                                                    if any(a for a in ent_effects.index if a == entity_ind):
                                                        effect = ent_effects.loc[entity_ind][0]
                                                        Y_test_pred["predictions"].loc[entity_ind, time_ind] = Y_test_pred["predictions"].loc[entity_ind, time_ind] + effect

                                            # Adjust format
                                            Y_test_pred = Y_test_pred.reset_index()["predictions"]
                                            Y_test = Y_test.reset_index()[response_var]

                                            # Save R² for test data
                                            model_eval_r2.iloc[val][response_var] = r2_score(Y_test, Y_test_pred)

                                            # Save MSE for test data
                                            model_eval_mse.iloc[val]["Value"] = mean_squared_error(Y_test, Y_test_pred, squared = True)

                                            # Save RMSE for test data
                                            model_eval_rmse.iloc[val]["Value"] = mean_squared_error(Y_test, Y_test_pred, squared = False)

                                            # Save MAE for test data
                                            model_eval_mae.iloc[val]["Value"] = mean_absolute_error(Y_test, Y_test_pred)

                                            # Save MaxERR for test data
                                            model_eval_maxerr.iloc[val]["Value"] = max_error(Y_test, Y_test_pred)

                                            # Save explained variance regression score for test data
                                            model_eval_evrs.iloc[val]["Value"] = explained_variance_score(Y_test, Y_test_pred)

                                            # Save sum of squared residuals for test data
                                            model_eval_ssr.iloc[val]["Value"] = ((Y_test-Y_test_pred)**2).sum()

                                            # Save residual values for test data 
                                            res = Y_test-Y_test_pred
                                            resdiuals_allruns[val] = res

                                            progress1 += 1
                                            my_bar.progress(progress1/(val_runs))
                                        
                                        # Calculate mean performance statistics
                                        # Mean
                                        model_eval_mean.loc["% VE"]["Value"] = model_eval_r2[response_var].mean()
                                        model_eval_mean.loc["MSE"]["Value"] = model_eval_mse["Value"].mean()
                                        model_eval_mean.loc["RMSE"]["Value"] = model_eval_rmse["Value"].mean()
                                        model_eval_mean.loc["MAE"]["Value"] = model_eval_mae["Value"].mean()
                                        model_eval_mean.loc["MaxErr"]["Value"] = model_eval_maxerr["Value"].mean()
                                        model_eval_mean.loc["EVRS"]["Value"] = model_eval_evrs["Value"].mean()
                                        model_eval_mean.loc["SSR"]["Value"] = model_eval_ssr["Value"].mean()
                                        # Sd
                                        model_eval_sd.loc["% VE"]["Value"] = model_eval_r2[response_var].std()
                                        model_eval_sd.loc["MSE"]["Value"] = model_eval_mse["Value"].std()
                                        model_eval_sd.loc["RMSE"]["Value"] = model_eval_rmse["Value"].std()
                                        model_eval_sd.loc["MAE"]["Value"] = model_eval_mae["Value"].std()
                                        model_eval_sd.loc["MaxErr"]["Value"] = model_eval_maxerr["Value"].std()
                                        model_eval_sd.loc["EVRS"]["Value"] = model_eval_evrs["Value"].std()
                                        model_eval_sd.loc["SSR"]["Value"] = model_eval_ssr["Value"].std()
                                        # Residuals 
                                        residuals_collection = pd.DataFrame()
                                        for x in resdiuals_allruns: 
                                            residuals_collection = residuals_collection.append(pd.DataFrame(resdiuals_allruns[x]), ignore_index = True)
                                        residuals_collection.columns = [response_var]
                                    
                                        # Collect validation results
                                        model_val_results = {}
                                        model_val_results["mean"] = model_eval_mean
                                        model_val_results["sd"] = model_eval_sd
                                        model_val_results["residuals"] = residuals_collection
                                        model_val_results["variance explained"] = model_eval_r2

                                    # Full model
                                    # Progress bar
                                    st.info("Full model progress")
                                    my_bar_fm = st.progress(0.0)
                                    progress2 = 0
                                    # efe
                                    panel_model_efe = PanelOLS(Y_data, X_data1, entity_effects = True, time_effects = False)
                                    panel_model_fit_efe = panel_model_efe.fit(cov_type = PDM_cov_type, cluster_entity = cluster_entity, cluster_time = cluster_time, debiased = True, auto_df = True) 
                                    # tfe
                                    panel_model_tfe = PanelOLS(Y_data, X_data1, entity_effects = False, time_effects = True)
                                    panel_model_fit_tfe = panel_model_tfe.fit(cov_type = PDM_cov_type, cluster_entity = cluster_entity, cluster_time = cluster_time, debiased = True, auto_df = True) 
                                    # twfe
                                    panel_model_twfe = PanelOLS(Y_data, X_data1, entity_effects = True, time_effects = True)
                                    panel_model_fit_twfe = panel_model_twfe.fit(cov_type = PDM_cov_type, cluster_entity = cluster_entity, cluster_time = cluster_time, debiased = True, auto_df = True) 
                                    # re
                                    panel_model_re = RandomEffects(Y_data, X_data2)
                                    panel_model_fit_re = panel_model_re.fit(cov_type = PDM_cov_type, cluster_entity = cluster_entity, cluster_time = cluster_time, debiased = True, auto_df = True)
                                    # pool
                                    panel_model_pool = PooledOLS(Y_data, X_data2)
                                    panel_model_fit_pool = panel_model_pool.fit(cov_type = PDM_cov_type, cluster_entity = cluster_entity, cluster_time = cluster_time, debiased = True, auto_df = True)                   
                                    # save selected model
                                    if PDM_alg == "Entity Fixed Effects":
                                        panel_model_fit = panel_model_fit_efe
                                    if PDM_alg == "Time Fixed Effects":
                                        panel_model_fit = panel_model_fit_tfe
                                    if PDM_alg == "Two-ways Fixed Effects":
                                        panel_model_fit = panel_model_fit_twfe
                                    if PDM_alg == "Random Effects":
                                        panel_model_fit = panel_model_fit_re
                                    if PDM_alg == "Pooled":
                                        panel_model_fit = panel_model_fit_pool
                                        
                                    # Entity information
                                    ent_inf = pd.DataFrame(index = ["No. entities", "Avg observations", "Median observations", "Min observations", "Max observations"], columns = ["Value"])
                                    ent_inf.loc["No. entities"] = panel_model_fit.entity_info["total"]
                                    ent_inf.loc["Avg observations"] = panel_model_fit.entity_info["mean"]
                                    ent_inf.loc["Median observations"] = panel_model_fit.entity_info["median"]
                                    ent_inf.loc["Min observations"] = panel_model_fit.entity_info["min"]
                                    ent_inf.loc["Max observations"] = panel_model_fit.entity_info["max"]

                                    # Time information
                                    time_inf = pd.DataFrame(index = ["No. time periods", "Avg observations", "Median observations", "Min observations", "Max observations"], columns = ["Value"])
                                    time_inf.loc["No. time periods"] = panel_model_fit.time_info["total"]
                                    time_inf.loc["Avg observations"] = panel_model_fit.time_info["mean"]
                                    time_inf.loc["Median observations"] = panel_model_fit.time_info["median"]
                                    time_inf.loc["Min observations"] = panel_model_fit.time_info["min"]
                                    time_inf.loc["Max observations"] = panel_model_fit.time_info["max"]

                                    # Regression information
                                    reg_inf = pd.DataFrame(index = ["Dep. variable", "Estimator", "Method", "No. observations", "DF residuals", "DF model", "Covariance type"], columns = ["Value"])
                                    reg_inf.loc["Dep. variable"] = response_var
                                    reg_inf.loc["Estimator"] =  panel_model_fit.name
                                    if PDM_alg == "Entity Fixed Effects" or PDM_alg == "Time Fixed Effects" or "Two-ways Fixed":
                                        reg_inf.loc["Method"] = "Within"
                                    if PDM_alg == "Random Effects":
                                        reg_inf.loc["Method"] = "Quasi-demeaned"
                                    if PDM_alg == "Pooled":
                                        reg_inf.loc["Method"] = "Least squares"
                                    reg_inf.loc["No. observations"] = panel_model_fit.nobs
                                    reg_inf.loc["DF residuals"] = panel_model_fit.df_resid
                                    reg_inf.loc["DF model"] = panel_model_fit.df_model
                                    reg_inf.loc["Covariance type"] = panel_model_fit._cov_type

                                    # Regression statistics
                                    fitted = df[response_var]-panel_model_fit.resids.values
                                    obs = df[response_var]
                                    reg_stats = pd.DataFrame(index = ["R²", "R² (between)", "R² (within)", "R² (overall)", "Log-likelihood", "SST", "SST (overall)"], columns = ["Value"])
                                    reg_stats.loc["R²"] = panel_model_fit._r2
                                    reg_stats.loc["R² (between)"] = panel_model_fit._c2b**2
                                    reg_stats.loc["R² (within)"] = panel_model_fit._c2w**2
                                    reg_stats.loc["R² (overall)"] = panel_model_fit._c2o**2
                                    reg_stats.loc["Log-likelihood"] = panel_model_fit._loglik
                                    reg_stats.loc["SST"] = panel_model_fit.total_ss
                                    reg_stats.loc["SST (overall)"] = ((obs-obs.mean())**2).sum()

                                    # Overall performance metrics (with effects)
                                    reg_overall = pd.DataFrame(index = ["% VE", "MSE", "RMSE", "MAE", "MaxErr", "EVRS", "SSR"], columns = ["Value"])
                                    reg_overall.loc["% VE"] = r2_score(obs, fitted)
                                    reg_overall.loc["MSE"] = mean_squared_error(obs, fitted, squared = True)
                                    reg_overall.loc["RMSE"] = mean_squared_error(obs, fitted, squared = False)
                                    reg_overall.loc["MAE"] = mean_absolute_error(obs, fitted)
                                    reg_overall.loc["MaxErr"] = max_error(obs, fitted)
                                    reg_overall.loc["EVRS"] = explained_variance_score(obs, fitted)
                                    reg_overall.loc["SSR"] = ((obs-fitted)**2).sum()

                                    # ANOVA
                                    if PDM_alg == "Pooled":
                                        Y_data_mlr = df[response_var]
                                        X_data_mlr = sm.add_constant(df[expl_var])
                                        full_model_mlr = sm.OLS(Y_data_mlr, X_data_mlr)
                                        full_model_fit = full_model_mlr.fit()
                                        reg_anova = pd.DataFrame(index = ["Regression", "Residual", "Total"], columns = ["DF", "SS", "MS", "F-statistic"])
                                        reg_anova.loc["Regression"]["DF"] = full_model_fit.df_model
                                        reg_anova.loc["Regression"]["SS"] = full_model_fit.ess
                                        reg_anova.loc["Regression"]["MS"] = full_model_fit.ess/full_model_fit.df_model
                                        reg_anova.loc["Regression"]["F-statistic"] = full_model_fit.fvalue
                                        reg_anova.loc["Residual"]["DF"] = full_model_fit.df_resid
                                        reg_anova.loc["Residual"]["SS"] = full_model_fit.ssr
                                        reg_anova.loc["Residual"]["MS"] = full_model_fit.ssr/full_model_fit.df_resid
                                        reg_anova.loc["Residual"]["F-statistic"] = ""
                                        reg_anova.loc["Total"]["DF"] = full_model_fit.df_resid + full_model_fit.df_model
                                        reg_anova.loc["Total"]["SS"] = full_model_fit.ssr + full_model_fit.ess
                                        reg_anova.loc["Total"]["MS"] = ""
                                        reg_anova.loc["Total"]["F-statistic"] = ""

                                    # Coefficients
                                    if PDM_alg == "Entity Fixed Effects" or PDM_alg == "Time Fixed Effects" or "Two-ways Fixed Effects":
                                        reg_coef = pd.DataFrame(index = expl_var, columns = ["coeff", "std err", "t-statistic", "p-value", "lower 95%", "upper 95%"])
                                        for c in expl_var:
                                            reg_coef.loc[c]["coeff"] = panel_model_fit.params[expl_var.index(c)]
                                            reg_coef.loc[c]["std err"] = panel_model_fit.std_errors.loc[c]
                                            reg_coef.loc[c]["t-statistic"] = panel_model_fit.tstats.loc[c]
                                            reg_coef.loc[c]["p-value"] = panel_model_fit.pvalues.loc[c]
                                            reg_coef.loc[c]["lower 95%"] = panel_model_fit.conf_int(level = 0.95).loc[c]["lower"]
                                            reg_coef.loc[c]["upper 95%"] = panel_model_fit.conf_int(level = 0.95).loc[c]["upper"]
                                    if PDM_alg == "Random Effects" or PDM_alg == "Pooled":
                                        reg_coef = pd.DataFrame(index = ["const"]+ expl_var, columns = ["coeff", "std err", "t-statistic", "p-value", "lower 95%", "upper 95%"])
                                        for c in ["const"] + expl_var:
                                            reg_coef.loc[c]["coeff"] = panel_model_fit.params[(["const"]+ expl_var).index(c)]
                                            reg_coef.loc[c]["std err"] = panel_model_fit.std_errors.loc[c]
                                            reg_coef.loc[c]["t-statistic"] = panel_model_fit.tstats.loc[c]
                                            reg_coef.loc[c]["p-value"] = panel_model_fit.pvalues.loc[c]
                                            reg_coef.loc[c]["lower 95%"] = panel_model_fit.conf_int(level = 0.95).loc[c]["lower"]
                                            reg_coef.loc[c]["upper 95%"] = panel_model_fit.conf_int(level = 0.95).loc[c]["upper"]
                                    
                                    # Effects
                                    reg_ent_effects = pd.DataFrame(index = df[entity].drop_duplicates(), columns = ["Value"])
                                    reg_time_effects = pd.DataFrame(index = sorted(list(df[time].drop_duplicates())), columns = ["Value"])
                                    reg_comb_effects = panel_model_fit.estimated_effects
                                    reg_comb_effects.columns = ["Value"]
                                    
                                    # Use LSDV for estimating effects
                                    Y_data_mlr = df[response_var]
                                    if PDM_alg == "Pooled" or PDM_alg == "Random Effects":
                                        X_data_mlr = sm.add_constant(df[expl_var])
                                    else: X_data_mlr = df[expl_var]

                                    if PDM_alg == "Entity Fixed Effects":
                                        X_data_mlr = pd.concat([X_data_mlr, pd.get_dummies(df[entity])], axis = 1)
                                        model_mlr = sm.OLS(Y_data_mlr, X_data_mlr)
                                        model_mlr_fit = model_mlr.fit()
                                        for e in reg_ent_effects.index:
                                            reg_ent_effects.loc[e]["Value"] = model_mlr_fit.params[e]
                                        for t in reg_time_effects.index:
                                            reg_time_effects.loc[t]["Value"] = 0
                                    if PDM_alg == "Time Fixed Effects":
                                        X_data_mlr = pd.concat([X_data_mlr, pd.get_dummies(df[time])], axis = 1)
                                        model_mlr = sm.OLS(Y_data_mlr, X_data_mlr)
                                        model_mlr_fit = model_mlr.fit()
                                        for e in reg_ent_effects.index:
                                            reg_ent_effects.loc[e]["Value"] = 0
                                        for t in reg_time_effects.index:
                                            reg_time_effects.loc[t]["Value"] = model_mlr_fit.params[t]
                                    if PDM_alg == "Two-ways Fixed Effects":
                                        X_data_mlr = pd.concat([X_data_mlr, pd.get_dummies(df[entity]), pd.get_dummies(df[time])], axis = 1)
                                        model_mlr = sm.OLS(Y_data_mlr, X_data_mlr)
                                        model_mlr_fit = model_mlr.fit()
                                        for e in reg_ent_effects.index:
                                            reg_ent_effects.loc[e]["Value"] = model_mlr_fit.params[e]
                                        for t in reg_time_effects.index:
                                            reg_time_effects.loc[t]["Value"] = model_mlr_fit.params[t]
                                    if PDM_alg == "Random Effects":
                                        for e in reg_ent_effects.index:
                                            reg_ent_effects.loc[e]["Value"] = reg_comb_effects.loc[e,].reset_index(drop = True).iloc[0][0]
                                        for t in reg_time_effects.index:
                                            reg_time_effects.loc[t]["Value"] = 0

                                    # New predictions
                                    if df_new.empty == False:

                                        data_new = df_new.set_index([entity, time])
                                        X_data1_new = data_new[expl_var] # for efe, tfe, twfe
                                        X_data2_new = sm.add_constant(data_new[expl_var]) # for re, pool
                                        
                                        if PDM_alg != "Pooled" and PDM_alg != "Random Effects":
                                            X_data_new = X_data1_new.copy()
                                        if PDM_alg == "Pooled" or PDM_alg == "Random Effects":
                                            X_data_new = X_data2_new.copy()
                                        
                                        # Prediction for new prediction data (without including effects)
                                        Y_pred_new = panel_model_fit.predict(X_data_new)
                                        
                                        # Add effects for new predictions
                                        for p in range(Y_pred_new.size):
                                            
                                            entity_ind = Y_pred_new.index[p][0]
                                            time_ind = Y_pred_new.index[p][1]
                                            
                                            # if effects are available, add effect
                                            if PDM_alg == "Entity Fixed Effects":
                                                if any(a for a in reg_ent_effects.index if a == entity_ind):
                                                    effect = reg_ent_effects.loc[entity_ind][0]
                                                    Y_pred_new["predictions"].loc[entity_ind, time_ind] = Y_pred_new["predictions"].loc[entity_ind, time_ind] + effect
                                            if PDM_alg == "Time Fixed Effects":
                                                if any(a for a in reg_time_effects.index if a == time_ind):
                                                    effect = reg_time_effects.loc[time_ind][0]
                                                    Y_pred_new["predictions"].loc[entity_ind, time_ind] = Y_pred_new["predictions"].loc[entity_ind, time_ind] + effect
                                            if PDM_alg == "Two-ways Fixed Effects":
                                                if any(a for a in reg_time_effects.index if a == time_ind):
                                                    effect_time = reg_time_effects.loc[time_ind][0]
                                                else: effect_time = 0
                                                if any(a for a in reg_ent_effects.index if a == entity_ind):
                                                    effect_entity = reg_ent_effects.loc[entity_ind][0]
                                                else: effect_entity = 0    
                                                Y_pred_new["predictions"].loc[entity_ind, time_ind] = Y_pred_new["predictions"].loc[entity_ind, time_ind] + effect_entity + effect_time
                                            if PDM_alg == "Random Effects":
                                                if any(a for a in reg_ent_effects.index if a == entity_ind):
                                                    effect = reg_ent_effects.loc[entity_ind][0]
                                                    Y_pred_new["predictions"].loc[entity_ind, time_ind] = Y_pred_new["predictions"].loc[entity_ind, time_ind] + effect
                                    
                                    # Variance decomposition
                                    if PDM_alg == "Random Effects":
                                        reg_var_decomp = pd.DataFrame(index = ["idiosyncratic", "individual"], columns = ["variance", "share"])
                                        reg_theta = pd.DataFrame(index = ["theta"], columns = df[entity].drop_duplicates())
                                        reg_var_decomp.loc["idiosyncratic"]["variance"] = panel_model_fit.variance_decomposition["Residual"]
                                        reg_var_decomp.loc["individual"]["variance"] = panel_model_fit.variance_decomposition["Effects"]
                                        reg_var_decomp.loc["idiosyncratic"]["share"] = panel_model_fit.variance_decomposition["Residual"]/(panel_model_fit.variance_decomposition["Residual"]+panel_model_fit.variance_decomposition["Effects"])
                                        reg_var_decomp.loc["individual"]["share"] = panel_model_fit.variance_decomposition["Effects"]/(panel_model_fit.variance_decomposition["Residual"]+panel_model_fit.variance_decomposition["Effects"])
                                        reg_theta.loc["theta"] = list(panel_model_fit.theta.values)
                                        for j in reg_theta.columns:
                                            reg_theta.loc["theta"][j] = reg_theta.loc["theta"][j][0]
                                    
                                    # Statistical tests
                                    if PDM_alg == "Entity Fixed Effects":
                                        if PDM_cov_type == "homoskedastic":
                                            reg_test = pd.DataFrame(index = ["test statistic", "p-value", "distribution"], columns = ["F-test (non-robust)", "F-test (robust)", "F-test (poolability)", "Hausman-test"])
                                        else:
                                            reg_test = pd.DataFrame(index = ["test statistic", "p-value", "distribution"], columns = ["F-test (non-robust)", "F-test (robust)", "F-test (poolability)"])
                                    else:
                                        reg_test = pd.DataFrame(index = ["test statistic", "p-value", "distribution"], columns = ["F-test (non-robust)", "F-test (robust)", "F-test (poolability)"])
                                    if PDM_alg == "Pooled" or PDM_alg == "Random Effects":
                                        reg_test = pd.DataFrame(index = ["test statistic", "p-value", "distribution"], columns = ["F-test (non-robust)", "F-test (robust)"])
                                    reg_test.loc["test statistic"]["F-test (non-robust)"] = panel_model_fit.f_statistic.stat
                                    reg_test.loc["p-value"]["F-test (non-robust)"] = panel_model_fit.f_statistic.pval
                                    reg_test.loc["distribution"]["F-test (non-robust)"] = "F(" + str(panel_model_fit.f_statistic.df) + ", " + str(panel_model_fit.f_statistic.df_denom) + ")"
                                    reg_test.loc["test statistic"]["F-test (robust)"] = panel_model_fit.f_statistic_robust.stat 
                                    reg_test.loc["p-value"]["F-test (robust)"] = panel_model_fit.f_statistic_robust.pval
                                    reg_test.loc["distribution"]["F-test (robust)"] = "F(" + str(panel_model_fit.f_statistic_robust.df) + ", " + str(panel_model_fit.f_statistic_robust.df_denom) + ")"
                                    if PDM_alg != "Pooled" and PDM_alg != "Random Effects" :
                                        reg_test.loc["test statistic"]["F-test (poolability)"] = panel_model_fit.f_pooled.stat
                                        reg_test.loc["p-value"]["F-test (poolability)"] = panel_model_fit.f_pooled.pval
                                        reg_test.loc["distribution"]["F-test (poolability)"] = "F(" + str(panel_model_fit.f_pooled.df) + ", " + str(panel_model_fit.f_pooled.df_denom) + ")"
                                    if PDM_alg == "Entity Fixed Effects":
                                        if PDM_cov_type == "homoskedastic":
                                            reg_test.loc["test statistic"]["Hausman-test"] = fc.hausman_test(panel_model_fit, panel_model_fit_re)[0] 
                                            reg_test.loc["p-value"]["Hausman-test"] = fc.hausman_test(panel_model_fit, panel_model_fit_re)[2] 
                                            reg_test.loc["distribution"]["Hausman-test"] = "Chi²(" + str(fc.hausman_test(panel_model_fit, panel_model_fit_re)[1])  + ")"
                            
                                    # Heteroskedasticity tests
                                    reg_het_test = pd.DataFrame(index = ["test statistic", "p-value"], columns = ["Breusch-Pagan test", "White test (without int.)", "White test (with int.)"])
                                    if PDM_alg == "Pooled":
                                        # Create datasets
                                        Y_data_mlr = df[response_var]
                                        X_data_mlr = sm.add_constant(df[expl_var])
                                        # Create MLR models 
                                        full_model_mlr = sm.OLS(Y_data_mlr, X_data_mlr)
                                        full_model_fit = full_model_mlr.fit()
                                        # Breusch-Pagan heteroscedasticity test
                                        bp_result = sm.stats.diagnostic.het_breuschpagan(full_model_fit.resid, full_model_fit.model.exog) 
                                        reg_het_test.loc["test statistic"]["Breusch-Pagan test"] = bp_result[0]
                                        reg_het_test.loc["p-value"]["Breusch-Pagan test"] = bp_result[1]
                                        # White heteroscedasticity test with interaction
                                        white_int_result = sm.stats.diagnostic.het_white(full_model_fit.resid, full_model_fit.model.exog)
                                        reg_het_test.loc["test statistic"]["White test (with int.)"] = white_int_result[0]
                                        reg_het_test.loc["p-value"]["White test (with int.)"] = white_int_result[1]
                                        # White heteroscedasticity test without interaction
                                        X_data_mlr_white = X_data_mlr
                                        for i in expl_var: 
                                            X_data_mlr_white[i+ "_squared"] = X_data_mlr_white[i]**2
                                        white = sm.OLS(full_model_fit.resid**2, X_data_mlr_white)
                                        del X_data_mlr_white
                                        white_fit = white.fit()
                                        white_statistic = white_fit.rsquared*data.shape[0]
                                        white_p_value = stats.chi2.sf(white_statistic,len(white_fit.model.exog_names)-1)
                                        reg_het_test.loc["test statistic"]["White test (without int.)"] = white_statistic
                                        reg_het_test.loc["p-value"]["White test (without int.)"] = white_p_value
                                    
                                    # Residuals distribution
                                    reg_resid = pd.DataFrame(index = ["min", "25%-Q", "median", "75%-Q", "max"], columns = ["Value"])
                                    reg_resid.loc["min"]["Value"] = panel_model_fit.resids.min()
                                    reg_resid.loc["25%-Q"]["Value"] = panel_model_fit.resids.quantile(q = 0.25)
                                    reg_resid.loc["median"]["Value"] = panel_model_fit.resids.quantile(q = 0.5)
                                    reg_resid.loc["75%-Q"]["Value"] = panel_model_fit.resids.quantile(q = 0.75)
                                    reg_resid.loc["max"]["Value"] = panel_model_fit.resids.max()

                                    # Save full model results
                                    model_full_results = {}
                                    model_full_results["Entity information"] = ent_inf
                                    model_full_results["Time information"] = time_inf
                                    model_full_results["Regression information"] = reg_inf
                                    model_full_results["Regression statistics"] = reg_stats
                                    model_full_results["Overall performance"] = reg_overall
                                    if PDM_alg == "Pooled":
                                        model_full_results["ANOVA"] = reg_anova
                                    model_full_results["Coefficients"] = reg_coef
                                    model_full_results["Entity effects"] = reg_ent_effects
                                    model_full_results["Time effects"] = reg_time_effects
                                    model_full_results["Combined effects"] = reg_comb_effects
                                    if PDM_alg == "Random Effects":
                                        model_full_results["Variance decomposition"] = reg_var_decomp
                                        model_full_results["Theta"] = reg_theta
                                    model_full_results["tests"] = reg_test
                                    model_full_results["hetTests"] = reg_het_test
                                    model_full_results["Residuals"] = reg_resid
                                    
                                    progress2 += 1
                                    my_bar_fm.progress(progress2/1)
                                    # Success message
                                    st.success('Model run successfully!')
                    else: st.error("ERROR: No data available for Modelling!")

    #++++++++++++++++++++++
    # PDM OUTPUT

    # Show only if model was run (no further widgets after run models or the full page reloads)
    if run_models:
        st.write("")
        st.write("")
        st.header("**Model outputs**")

        #--------------------------------------------------------------------------------------
        # FULL MODEL OUTPUT

        full_output = st.expander("Full model output", expanded = False)
        with full_output:

            if model_full_results is not None:

                st.markdown("**Correlation Matrix & 2D-Histogram**")
                # Define variable selector
                var_sel_cor = alt.selection_single(fields=['variable', 'variable2'], clear=False, 
                                    init={'variable': response_var, 'variable2': response_var})
                # Calculate correlation data
                corr_data = df[[response_var] + expl_var].corr().stack().reset_index().rename(columns={0: "correlation", 'level_0': "variable", 'level_1': "variable2"})
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
                value_columns = df[[response_var] + expl_var]
                df_2dbinned = pd.concat([fc.compute_2d_histogram(var1, var2, df) for var1 in value_columns for var2 in value_columns])
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
                if sett_hints:
                    st.info(str(fc.learning_hints("mod_cor")))
                st.write("")

                #-------------------------------------------------------------

                # Regression output
                st.markdown("**Regression output**")

                full_out_col1, full_out_col2 = st.columns(2)
                with full_out_col1:
                    # Entity information
                    st.write("Entity information:")
                    st.table(model_full_results["Entity information"].style.set_precision(user_precision))
                with full_out_col2:
                    # Time information
                    st.write("Time period information:")
                    st.table(model_full_results["Time information"].style.set_precision(user_precision))
                if sett_hints:
                    st.info(str(fc.learning_hints("mod_pd_information")))
                st.write("")

                full_out_col3, full_out_col4 = st.columns(2)
                with full_out_col3:
                    # Regression information
                    st.write("Regression information:")
                    st.table(model_full_results["Regression information"].style.set_precision(user_precision))
                with full_out_col4:
                    # Regression statistics
                    st.write("Regression statistics:")
                    st.table(model_full_results["Regression statistics"].style.set_precision(user_precision))
                if sett_hints:
                    st.info(str(fc.learning_hints("mod_pd_regression")))
                st.write("")

                # Overall performance (with effects)
                full_out_col_op1, full_out_col_op2 = st.columns(2)
                with full_out_col_op1:
                    if PDM_alg != "Pooled":
                        st.write("Overall performance (with effects):")
                    if PDM_alg == "Pooled":
                        st.write("Overall performance :")
                    st.table(model_full_results["Overall performance"].style.set_precision(user_precision))
                # Residuals
                with full_out_col_op2:
                    st.write("Residuals:")
                    st.table(model_full_results["Residuals"].style.set_precision(user_precision))     
                if sett_hints:
                    st.info(str(fc.learning_hints("mod_pd_overallPerf")))
                st.write("")

                # Coefficients
                st.write("Coefficients:")
                st.table(model_full_results["Coefficients"].style.set_precision(user_precision))
                if sett_hints:
                    st.info(str(fc.learning_hints("mod_pd_coef")))
                st.write("") 

                # Effects
                if PDM_alg != "Pooled":
                    full_out_col5, full_out_col6 = st.columns(2)
                    with full_out_col5:
                        st.write("Entity effects:")
                        st.write(model_full_results["Entity effects"].style.set_precision(user_precision))
                    with full_out_col6:
                        st.write("Time effects:")
                        st.write(model_full_results["Time effects"].style.set_precision(user_precision))
                    full_out_col7, full_out_col8 = st.columns(2)
                    with full_out_col7:
                        st.write("Combined effects:")
                        st.write(model_full_results["Combined effects"]) 
                    with full_out_col8: 
                        st.write("")
                    if sett_hints:
                        st.info(str(fc.learning_hints("mod_pd_effects")))
                    st.write("")  

                # ANOVA
                if PDM_alg == "Pooled":
                    st.write("ANOVA:")
                    st.table(model_full_results["ANOVA"].style.set_precision(user_precision))
                    if sett_hints:
                        st.info(str(fc.learning_hints("mod_pd_anova")))
                    st.write("")  

                # Statistical tests
                if PDM_alg == "Random Effects":
                    full_out_col_re1, full_out_col_re2 = st.columns(2)
                    with full_out_col_re1:
                        st.write("Variance decomposition:")
                        st.table(model_full_results["Variance decomposition"].style.set_precision(user_precision))
                    with full_out_col_re2:
                        st.write("Theta:")
                        st.table(model_full_results["Theta"].transpose().style.set_precision(user_precision))
                    if sett_hints:
                        st.info(str(fc.learning_hints("mod_pd_varDecRE")))
                    st.write("")
                    st.write("F-tests:")
                    st.table(model_full_results["tests"].transpose().style.set_precision(user_precision)) 
                    if sett_hints:
                        st.info(str(fc.learning_hints("mod_pd_testRE")))
                    st.write("")    
                if PDM_alg == "Entity Fixed Effects":
                    if PDM_cov_type == "homoskedastic":
                        st.write("F-tests and Hausman-test:")
                    else: st.write("F-tests:")
                    st.table(model_full_results["tests"].transpose().style.set_precision(user_precision))
                    if PDM_cov_type == "homoskedastic":
                        if sett_hints:
                            st.info(str(fc.learning_hints("mod_pd_testEFE_homosk")))
                    else:
                        if sett_hints:
                            st.info(str(fc.learning_hints("mod_pd_testEFE")))
                    st.write("")  
                if PDM_alg != "Entity Fixed Effects" and PDM_alg != "Random Effects":
                    st.write("F-tests:")
                    st.table(model_full_results["tests"].transpose().style.set_precision(user_precision))
                    if PDM_alg == "Pooled":
                        if sett_hints:
                            st.info(str(fc.learning_hints("mod_pd_test_pooled")))
                    else: 
                        if sett_hints:
                            st.info(str(fc.learning_hints("mod_pd_test"))) 
                    st.write("") 

                # Heteroskedasticity tests
                if PDM_alg == "Pooled":
                    st.write("Heteroskedasticity tests:")
                    st.table(model_full_results["hetTests"].transpose().style.set_precision(user_precision))
                    if sett_hints:
                        st.info(str(fc.learning_hints("mod_md_MLR_hetTest"))) 
                    st.write("")          
                
                # Graphical output
                full_out_col10, full_out_col11 = st.columns(2)
                fitted_withEff = df[response_var]-panel_model_fit.resids.values
                with full_out_col10:
                    st.write("Observed vs Fitted:")
                    observed_fitted_data = pd.DataFrame()
                    observed_fitted_data["Observed"] = df[response_var]
                    observed_fitted_data["Fitted"] = list(fitted_withEff)
                    observed_fitted_data["Index"] = df.index
                    observed_fitted = alt.Chart(observed_fitted_data, height = 200).mark_circle(size=20).encode(
                        x = alt.X("Fitted", title = "fitted", scale = alt.Scale(domain = [min(observed_fitted_data["Fitted"]), max(observed_fitted_data["Fitted"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                        y = alt.Y("Observed", title = "observed", scale = alt.Scale(zero = False),  axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                        tooltip = ["Observed", "Fitted", "Index"]
                    )
                    observed_fitted_plot = observed_fitted + observed_fitted.transform_regression("Fitted", "Observed").mark_line(size = 2, color = "darkred")
                    st.altair_chart(observed_fitted_plot, use_container_width = True)
                with full_out_col11:
                    st.write("Residuals vs Fitted:")
                    residuals_fitted_data = pd.DataFrame()
                    residuals_fitted_data["Residuals"] = panel_model_fit.resids.values
                    residuals_fitted_data["Fitted"] = list(fitted_withEff)
                    residuals_fitted_data["Index"] = df.index
                    residuals_fitted = alt.Chart(residuals_fitted_data, height = 200).mark_circle(size=20).encode(
                        x = alt.X("Fitted", title = "fitted", scale = alt.Scale(domain = [min(residuals_fitted_data["Fitted"]), max(residuals_fitted_data["Fitted"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                        y = alt.Y("Residuals", title = "residuals", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                        tooltip = ["Residuals", "Fitted", "Index"]
                    )
                    residuals_fitted_plot = residuals_fitted + residuals_fitted.transform_loess("Fitted", "Residuals", bandwidth = 0.5).mark_line(size = 2, color = "darkred")
                    st.altair_chart(residuals_fitted_plot, use_container_width = True)
                if sett_hints:
                    st.info(str(fc.learning_hints("mod_md_MLR_obsResVsFit"))) 
                st.write("")
                if PDM_alg == "Pooled":
                    full_out_col12, full_out_col13 = st.columns(2)
                    with full_out_col12:
                        st.write("Normal QQ-plot:")
                        residuals = panel_model_fit.resids.values
                        qq_plot_data = pd.DataFrame()
                        qq_plot_data["StandResiduals"] = (residuals - residuals.mean())/residuals.std()
                        qq_plot_data["Index"] = df.index
                        qq_plot_data = qq_plot_data.sort_values(by = ["StandResiduals"])
                        qq_plot_data["Theoretical quantiles"] = stats.probplot(residuals, dist="norm")[0][0]
                        qq_plot = alt.Chart(qq_plot_data, height = 200).mark_circle(size=20).encode(
                            x = alt.X("Theoretical quantiles", title = "theoretical quantiles", scale = alt.Scale(domain = [min(qq_plot_data["Theoretical quantiles"]), max(qq_plot_data["Theoretical quantiles"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            y = alt.Y("StandResiduals", title = "stand. residuals", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            tooltip = ["StandResiduals", "Theoretical quantiles", "Index"]
                        )
                        line = alt.Chart(
                            pd.DataFrame({"Theoretical quantiles": [min(qq_plot_data["Theoretical quantiles"]), max(qq_plot_data["Theoretical quantiles"])], "StandResiduals": [min(qq_plot_data["Theoretical quantiles"]), max(qq_plot_data["Theoretical quantiles"])]})).mark_line(size = 2, color = "darkred").encode(
                                    alt.X("Theoretical quantiles"),
                                    alt.Y("StandResiduals"),
                        )
                        st.altair_chart(qq_plot + line, use_container_width = True)
                        if sett_hints:
                            st.info(str(fc.learning_hints("mod_md_MLR_qqplot"))) 
                        st.write("")
                    with full_out_col13:
                        st.write("Scale-Location:")
                        scale_location_data = pd.DataFrame()
                        residuals = panel_model_fit.resids.values
                        scale_location_data["SqrtStandResiduals"] = np.sqrt(abs((residuals - residuals.mean())/residuals.std()))
                        scale_location_data["Fitted"] = panel_model_fit._fitted.values
                        scale_location_data["Index"] = df.index
                        scale_location = alt.Chart(scale_location_data, height = 200).mark_circle(size=20).encode(
                            x = alt.X("Fitted", title = "fitted", scale = alt.Scale(domain = [min(scale_location_data["Fitted"]), max(scale_location_data["Fitted"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            y = alt.Y("SqrtStandResiduals", title = "sqrt(|stand. residuals|)", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            tooltip = ["SqrtStandResiduals", "Fitted", "Index"]
                        )
                        scale_location_plot = scale_location + scale_location.transform_loess("Fitted", "SqrtStandResiduals", bandwidth = 0.5).mark_line(size = 2, color = "darkred")
                        st.altair_chart(scale_location_plot, use_container_width = True)
                        if sett_hints:
                            st.info(str(fc.learning_hints("mod_md_MLR_scaleLoc"))) 
                        st.write("")

                    full_out_col14, full_out_col15 = st.columns(2)
                    Y_data_mlr = df[response_var]
                    X_data_mlr = sm.add_constant(df[expl_var])
                    full_model_mlr = sm.OLS(Y_data_mlr, X_data_mlr)
                    full_model_fit = full_model_mlr.fit()
                    with full_out_col14:
                        st.write("Residuals vs Leverage:")
                        residuals_leverage_data = pd.DataFrame()
                        residuals = panel_model_fit.resids.values
                        residuals_leverage_data["StandResiduals"] = (residuals - residuals.mean())/residuals.std()
                        residuals_leverage_data["Leverage"] = full_model_fit.get_influence().hat_matrix_diag
                        residuals_leverage_data["Index"] = df.index
                        residuals_leverage = alt.Chart(residuals_leverage_data, height = 200).mark_circle(size=20).encode(
                            x = alt.X("Leverage", title = "leverage", scale = alt.Scale(domain = [min(residuals_leverage_data["Leverage"]), max(residuals_leverage_data["Leverage"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            y = alt.Y("StandResiduals", title = "stand. residuals", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            tooltip = ["StandResiduals","Leverage", "Index"]
                        )
                        residuals_leverage_plot = residuals_leverage + residuals_leverage.transform_loess("Leverage", "StandResiduals", bandwidth = 0.5).mark_line(size = 2, color = "darkred")
                        st.altair_chart(residuals_leverage_plot, use_container_width = True)
                    with full_out_col15:
                        st.write("Cook's distance:")
                        cooksD_data = pd.DataFrame()
                        cooksD_data["CooksD"] = full_model_fit.get_influence().cooks_distance[0]
                        cooksD_data["Index"] = df.index
                        cooksD = alt.Chart(cooksD_data, height = 200).mark_bar(size = 2).encode(
                            x = alt.X("Index", title = "index", scale = alt.Scale(domain = [-1, max(cooksD_data["Index"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            y = alt.Y("CooksD", title = "Cook's distance", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            tooltip = ["CooksD", "Index"]
                        )
                        st.altair_chart(cooksD, use_container_width = True)
                    if sett_hints:
                        st.info(str(fc.learning_hints("mod_md_MLR_resVsLev_cooksD")))
                
                # Download link for full model output
                output = BytesIO()
                excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                model_full_results["Entity information"].to_excel(excel_file, sheet_name="entity_information")
                model_full_results["Time information"].to_excel(excel_file, sheet_name="time_period_information")
                model_full_results["Regression information"].to_excel(excel_file, sheet_name="regression_information")
                model_full_results["Regression statistics"].to_excel(excel_file, sheet_name="regression_statistics")
                model_full_results["Overall performance"].to_excel(excel_file, sheet_name="overall_performance")
                model_full_results["Residuals"].to_excel(excel_file, sheet_name="residuals")
                model_full_results["Coefficients"].to_excel(excel_file, sheet_name="coefficients")
                if PDM_alg != "Pooled":
                    model_full_results["Entity effects"].to_excel(excel_file, sheet_name="entity_effects")
                    model_full_results["Time effects"].to_excel(excel_file, sheet_name="time_effects")
                    model_full_results["Combined effects"].to_excel(excel_file, sheet_name="combined_effects")
                if PDM_alg == "Pooled":
                    model_full_results["ANOVA"].to_excel(excel_file, sheet_name="ANOVA")
                if PDM_alg == "Random Effects":
                    model_full_results["Variance decomposition"].to_excel(excel_file, sheet_name="variance_decomposition")
                    model_full_results["Theta"].to_excel(excel_file, sheet_name="theta")
                    model_full_results["tests"].to_excel(excel_file, sheet_name="statistical_tests")
                if PDM_alg == "Entity Fixed Effects":
                    model_full_results["tests"].to_excel(excel_file, sheet_name="statistical_tests")
                if PDM_alg != "Entity Fixed Effects" and PDM_alg != "Random Effects":
                    model_full_results["tests"].to_excel(excel_file, sheet_name="statistical_tests")
                if PDM_alg == "Pooled":
                    model_full_results["hetTests"].to_excel(excel_file, sheet_name="heteroskedasticity_tests")
                excel_file.save()
                excel_file = output.getvalue()
                b64 = base64.b64encode(excel_file)
                dl_file_name= "Full model output__" + PDM_alg + "__" + df_name + ".xlsx"
                st.markdown(
                    f"""
                <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download full model output</a>
                """,
                unsafe_allow_html=True)
                st.write("")

        #--------------------------------------------------------------------------------------
        # FULL MODEL PREDICTIONS

        prediction_output = st.expander("Full model predictions", expanded = False)
        with prediction_output:
            pred_col1, pred_col2 = st.columns(2)
            with pred_col1:
                st.write("Predictions for original data:")
                pred_orig = pd.DataFrame(fitted)
                pred_orig = pred_orig.join(df[[entity, time]])
                pred_orig = pred_orig.set_index([entity, time])
                st.write(pred_orig)
            with pred_col2:
                if do_modprednew == "Yes":
                    st.write("Predictions for new data:")
                    Y_pred_new.columns = [response_var]
                    st.write(Y_pred_new)
        
            #-------------------------------------------------------------
                        
            # Download links for prediction data
            output = BytesIO()
            predictions_excel = pd.ExcelWriter(output, engine="xlsxwriter")
            pred_orig.to_excel(predictions_excel, sheet_name="pred_orig")
            if do_modprednew == "Yes":
                Y_pred_new.to_excel(predictions_excel, sheet_name="pred_new")
            predictions_excel.save()
            predictions_excel = output.getvalue()
            b64 = base64.b64encode(predictions_excel)
            dl_file_name= "Full model predictions__" + PDM_alg + "__" + df_name + ".xlsx"
            st.markdown(
                f"""
            <a href="data:file/predictions_excel;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download full model predictions</a>
            """,
            unsafe_allow_html=True)
            st.write("")

        #--------------------------------------------------------------------------------------
        # VALIDATION OUTPUT
        
        if do_modval == "Yes":
            if PDM_alg == "Pooled":
                validation_output_name = "Validation output"
            if PDM_alg != "Pooled":
                validation_output_name = "Validation output (with effects)"
            val_output = st.expander(validation_output_name, expanded = False)
            with val_output:
                if model_val_results is not None:
                    val_col1, val_col2 = st.columns(2)

                    with val_col1:
                        # Metrics
                        st.write("Means of metrics across validation runs:")
                        st.table(model_val_results["mean"].style.set_precision(user_precision))
                    with val_col2:
                        # Metrics
                        st.write("SDs of metrics across validation runs:")
                        st.table(model_val_results["sd"].style.set_precision(user_precision))
                    if sett_hints:
                        st.info(str(fc.learning_hints("mod_pd_val_metrics"))) 
                    st.write("")
                    
                    val_col3, val_col4 = st.columns(2)
                    with val_col3:
                        # Residuals boxplot
                        if model_val_results["residuals"] is not None:
                            st.write("Boxplot of residuals across validation runs:")
                            residual_results = model_val_results["residuals"]
                            residuals_bplot = pd.melt(residual_results, ignore_index = False, var_name = "Variable", value_name = "Residuals")
                            residuals_boxchart = alt.Chart(residuals_bplot, height = 200).mark_boxplot(color = "#1f77b4", median = dict(color = "darkred")).encode(
                                x = alt.X("Residuals", title = "residuals", scale = alt.Scale(domain = [min(residuals_bplot["Residuals"]), max(residuals_bplot["Residuals"])])),
                                y = alt.Y("Variable", scale = alt.Scale(zero = False), title = None)
                            ).configure_axis(
                                labelFontSize = 12,
                                titleFontSize = 12
                            )
                            residuals_plot = residuals_boxchart #+ residuals_scatter
                            st.altair_chart(residuals_plot, use_container_width=True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_pd_val_resBoxplot")))
                            st.write("")
                    with val_col4:
                        # Variance explained boxplot
                        if model_val_results["variance explained"] is not None:
                            st.write("Boxplot of % VE across validation runs:")
                            ve_results = model_val_results["variance explained"]
                            ve_bplot = pd.melt(ve_results, ignore_index = False, var_name = "Variable", value_name = "% VE")
                            ve_boxchart = alt.Chart(ve_bplot, height = 200).mark_boxplot(color = "#1f77b4", median = dict(color = "darkred")).encode(
                                x = alt.X("% VE", scale = alt.Scale(domain = [min(ve_bplot["% VE"]), max(ve_bplot["% VE"])])),
                                y = alt.Y("Variable", title = None)
                            ).configure_axis(
                                labelFontSize = 12,
                                titleFontSize = 12
                            )
                            st.altair_chart(ve_boxchart, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_pd_val_VEBoxplot")))
                            st.write("")

                    # Residuals
                    if model_val_results["residuals"] is not None:
                        model_val_res = pd.DataFrame(index = ["min", "25%-Q", "median", "75%-Q", "max"], columns = ["Value"])
                        model_val_res.loc["min"]["Value"] = model_val_results["residuals"][response_var].min()
                        model_val_res.loc["25%-Q"]["Value"] = model_val_results["residuals"][response_var].quantile(q = 0.25)
                        model_val_res.loc["median"]["Value"] = model_val_results["residuals"][response_var].quantile(q = 0.5)
                        model_val_res.loc["75%-Q"]["Value"] = model_val_results["residuals"][response_var].quantile(q = 0.75)
                        model_val_res.loc["max"]["Value"] = model_val_results["residuals"][response_var].max()
                        st.write("Residuals distribution across all validation runs:")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.table(model_val_res.style.set_precision(user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_pd_val_res")))
                        with col2:
                            st.write("")
                        st.write("")

                   # Download link for validation output
                    output = BytesIO()
                    excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                    model_val_results["mean"].to_excel(excel_file, sheet_name="performance_metrics_mean")
                    model_val_results["sd"].to_excel(excel_file, sheet_name="performance_metrics_sd")
                    model_val_res.to_excel(excel_file, sheet_name="residuals_distribution")
                    excel_file.save()
                    excel_file = output.getvalue()
                    b64 = base64.b64encode(excel_file)
                    dl_file_name = "Validation output__" + PDM_alg + "__"  + df_name + ".xlsx"
                    st.markdown(
                        f"""
                    <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download validation output</a>
                    """,
                    unsafe_allow_html=True)
                    st.write("")   

#--------------------------------------------------------------------------------------