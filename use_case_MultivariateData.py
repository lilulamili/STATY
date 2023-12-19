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
import elements as el
import functions as fc
import modelling as ml
import os
import altair as alt
import altair 
import itertools
import statsmodels.api as sm
from scipy import stats 
import sys
import platform
import base64
from io import BytesIO
from pygam import LinearGAM, LogisticGAM, s
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler 
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import VisualizeNN as VisNN


#----------------------------------------------------------------------------------------------

def app():

    # Clear cache
    #st.runtime.legacy_caching.clear_cache()

    # Hide traceback in error messages (comment out for de-bugging)
    sys.tracebacklimit = 2000

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
    #++++++++++++++++++++++++++++++++++++++++++++
    # RESET INPUT
    #Session state
    if 'key' not in st.session_state:
        st.session_state['key'] = 0
    if 'model_tuning_results' not in st.session_state:
        st.session_state['model_tuning_results'] = None
    if 'model_val_results' not in st.session_state:
        st.session_state['model_val_results'] = None
    if 'model_full_results' not in st.session_state:
        st.session_state['model_full_results'] = None
    if 'full_model_ann_sk' not in st.session_state:
        st.session_state['full_model_ann_sk'] = None
    if 'ann_finalPara' not in st.session_state:
        st.session_state['ann_finalPara'] = None
    if 'ann_tuning_results' not in st.session_state:
        st.session_state['ann_tuning_results'] = None
                                     
    reset_clicked = st.sidebar.button("Reset all your input")
    if reset_clicked:
        st.session_state['key'] = st.session_state['key'] + 1
        st.session_state['model_tuning_results'] = None
        st.session_state['model_full_results'] = None
        st.session_state['full_model_ann_sk'] = None
        st.session_state['ann_finalPara'] = None
        st.session_state['ann_tuning_results'] = None
        st.session_state['model_val_results'] = None
        st.runtime.legacy_caching.clear_cache()
    st.sidebar.markdown("")
    
    def in_wid_change():
        st.session_state['model_tuning_results'] = None
        st.session_state['model_full_results'] = None
        st.session_state['full_model_ann_sk'] = None
        st.session_state['ann_finalPara'] = None
        st.session_state['ann_tuning_results'] = None
        st.session_state['model_val_results'] = None

    # Analysis type
    analysis_type = st.selectbox("What kind of analysis would you like to conduct?", ["Regression", "Multi-class classification", "Data decomposition"], on_change=in_wid_change)

    st.header("**Multivariate data**")

    if analysis_type == "Regression":
        st.markdown("Get your data ready for powerfull methods: Artificial Neural Networks, Boosted Regression Trees, Random Forest, Generalized Additive Models, Multiple Linear Regression, and Logistic Regression! Let STATY do data cleaning, variable transformations, visualizations and deliver you the stats you need. Specify your data processing preferences and start exploring your data stories right below... ")
    if analysis_type == "Multi-class classification":
        st.markdown("Get your data ready for powerfull multi-class classification methods! Let STATY do data cleaning, variable transformations, visualizations and deliver you the stats you need. Specify your data processing preferences and start exploring your data stories right below... ")
    if analysis_type == "Data decomposition":
        st.markdown("Decompose your data with Principal Component Analysis or Factor Analysis! Let STATY do data cleaning, variable transformations, visualizations and deliver you the stats you need. Specify your data processing preferences and start exploring your data stories right below... ")

    #------------------------------------------------------------------------------------------

    

    #++++++++++++++++++++++++++++++++++++++++++++
    # DATA IMPORT

    # File upload section
    df_dec = st.sidebar.radio("Get data", ["Use example dataset", "Upload data"])
    uploaded_data=None
    if df_dec == "Upload data":
        #st.subheader("Upload your data")
        #uploaded_data = st.sidebar.file_uploader("Make sure that dot (.) is a decimal separator!", type=["csv", "txt"])
        separator_expander=st.sidebar.expander('Upload settings')
        with separator_expander:
                      
            a4,a5=st.columns(2)
            with a4:
                dec_sep=a4.selectbox("Decimal sep.",['.',','])

            with a5:
                col_sep=a5.selectbox("Column sep.",[';',  ','  , '|', '\s+', '\t','other'])
                if col_sep=='other':
                    col_sep=st.text_input('Specify your column separator')     

            a4,a5=st.columns(2)  
            with a4:    
                thousands_sep=a4.selectbox("Thousands x sep.",[None,'.', ' ','\s+', 'other'])
                if thousands_sep=='other':
                    thousands_sep=st.text_input('Specify your thousands separator')  
             
            with a5:    
                encoding_val=a5.selectbox("Encoding",[None,'utf_8','utf_8_sig','utf_16_le','cp1140','cp1250','cp1251','cp1252','cp1253','cp1254','other'])
                if encoding_val=='other':
                    encoding_val=st.text_input('Specify your encoding')  
        
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
            if analysis_type == "Regression" or analysis_type == "Data decomposition":
                df = pd.read_csv("default data/WHR_2021.csv", sep = ";|,|\t",engine='python')
                df_name="WHR_2021" 
            if analysis_type == "Multi-class classification":
                df = pd.read_csv("default data/iris.csv", sep = ";|,|\t",engine='python')
                df_name="iris" 
    else:
        if analysis_type == "Regression" or analysis_type == "Data decomposition":
            df = pd.read_csv("default data/WHR_2021.csv", sep = ";|,|\t",engine='python') 
            df_name="WHR_2021" 
        if analysis_type == "Multi-class classification":
            df = pd.read_csv("default data/iris.csv", sep = ";|,|\t",engine='python')
            df_name="iris" 
    st.sidebar.markdown("")
        
    #Basic data info
    n_rows = df.shape[0]
    n_cols = df.shape[1]  

    #++++++++++++++++++++++++++++++++++++++++++++
    # SETTINGS

    settings_expander=st.sidebar.expander('Settings')
    with settings_expander:
        st.caption("**Precision**")
        user_precision=int(st.number_input('Number of digits after the decimal point',min_value=0,max_value=10,step=1,value=4))
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
    fc.theme_func_dl_button()

    #------------------------------------------------------------------------------------------

    
    #++++++++++++++++++++++++++++++++++++++++++++
    # DATA PREPROCESSING & VISUALIZATION

    # Check if enough data is available
    if n_rows > 0 and n_cols > 0:
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

        dev_expander_dsPre = st.expander("Explore raw data info and stats ", expanded = False)
        with dev_expander_dsPre:

            # Default data description:
            if uploaded_data == None:
                if analysis_type == "Regression" or analysis_type == "Data decomposition":
                    if st.checkbox("Show data description", value = False):          
                        st.markdown("**Data source:**")
                        st.markdown("The data come from the Gallup World Poll surveys from 2018 to 2020. For more details see the [World Happiness Report 2021](https://worldhappiness.report/).")
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
                        col2.write("residual of regressing national average of response to the question regarding money donations in the past month on GDP per capita")

                        col1,col2=st.columns(2) 
                        col1.write("Perceptions of corruption")
                        col2.write("the national average of the survey responses to the corresponding question")
                        
                        col1,col2=st.columns(2) 
                        col1.write("Positive affect")
                        col2.write("the  average  of  three  positive  affect  measures (happiness,  laugh  and  enjoyment)")
                        
                        col1,col2=st.columns(2)
                        col1.write("Negative affect (worry, sadness and anger)")
                        col2.write("the  average  of  three  negative  affect  measures  (worry, sadness and anger)")

                        st.markdown("")
                if analysis_type == "Multi-class classification":
                    if st.checkbox("Show data description", value = False):          
                        st.markdown("**Data source:**")
                        st.markdown("The data come from Fisher's Iris data set. See [here] (https://archive.ics.uci.edu/ml/datasets/iris) for more information.")
                        st.markdown("**Citation:**")
                        st.markdown("Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. Annals of Eugenics, 7(2): 179–188. doi: [10.1111/j.1469-1809.1936.tb02137.x] (https://doi.org/10.1111%2Fj.1469-1809.1936.tb02137.x)")
                        st.markdown("**Variables in the dataset:**")

                        col1,col2=st.columns(2) 
                        col1.write("class_category")
                        col2.write("Numerical category for 'class': Iris Setosa (0), Iris Versicolour (1), and Iris Virginica (2)")
                        
                        col1,col2=st.columns(2) 
                        col1.write("class")
                        col2.write("Iris Setosa, Iris Versicolour, and Iris Virginica")
                        
                        col1,col2=st.columns(2) 
                        col1.write("sepal length")
                        col2.write("sepal length in cm")
                        
                        col1,col2=st.columns(2)
                        col1.write("sepal width")
                        col2.write("sepal width in cm")
                        
                        col1,col2=st.columns(2) 
                        col1.write("petal length")
                        col2.write("petal length in cm")
                        
                        col1,col2=st.columns(2) 
                        col1.write("petal width")
                        col2.write("petal width in cm")                       
                        
                        st.markdown("")

            # Show raw data & data info
            df_summary = fc.data_summary(df) 
            if st.checkbox("Show raw data ", value = False):      
                st.write(df)

                st.write("Data shape: ", n_rows,  " rows and ", n_cols, " columns")
            if df[df.duplicated()].shape[0] > 0 or df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0] > 0:
                check_nasAnddupl=st.checkbox("Show duplicates and NAs info ", value = False) 
                if check_nasAnddupl:      
                    if df[df.duplicated()].shape[0] > 0:
                        st.write("Number of duplicates: ", df[df.duplicated()].shape[0])
                        st.write("Duplicate row index: ", ', '.join(map(str,list(df.index[df.duplicated()]))))
                    if df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0] > 0:
                        st.write("Number of rows with NAs: ", df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0])
                        st.write("Rows with NAs: ", ', '.join(map(str,list(pd.unique(np.where(df.isnull())[0])))))
                
            # Show variable info 
            if st.checkbox('Show variable info ', value = False): 
                st.write(df_summary["Variable types"])
        
            # Show summary statistics (raw data)
            if st.checkbox('Show summary statistics (raw data) ', value = False): 
                st.write(df_summary["ALL"].style.format(precision=user_precision))
                
                # Download link for summary statistics
                output = BytesIO()
                excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                df_summary["Variable types"].to_excel(excel_file, sheet_name="variable_info")
                df_summary["ALL"].to_excel(excel_file, sheet_name="summary_statistics")
                excel_file.close()
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
                
        #---------------------------------
        # DATA PROCESSING       
        #---------------------------------
        df=el.data_processing(df_name,df, n_rows,n_cols,sett_hints, user_precision,in_wid_change)
        

    
    #---------------------------------
    # DATA VISUALIZATION
    #---------------------------------
    data_visualization_container = st.container()
    with data_visualization_container:
        st.write("")
        st.write("")
        st.header("**Data visualization**")

        dev_expander_dv = st.expander("Explore visualization types ", expanded = False)
        with dev_expander_dv:
            if df.shape[1] > 0 and df.shape[0] > 0:
            
                st.write('**Variable selection**')
                varl_sel_options = df.columns
                var_sel = st.selectbox('Select variable for visualizations', varl_sel_options)

                if df[var_sel].dtypes == "float64" or df[var_sel].dtypes == "float32" or df[var_sel].dtypes == "int64" or df[var_sel].dtypes == "int32":
                    a4, a5 = st.columns(2)
                    with a4:
                        st.write('**Scatterplot with LOESS line**')
                        yy_options = df.columns
                        yy = st.selectbox('Select variable for y-axis', yy_options)
                        if df[yy].dtypes == "float64" or df[yy].dtypes == "float32" or df[yy].dtypes == "int64" or df[yy].dtypes == "int32":
                            fig_data = pd.DataFrame()
                            fig_data[yy] = df[yy]
                            fig_data[var_sel] = df[var_sel]
                            fig_data["Index"] = df.index
                            fig = alt.Chart(fig_data).mark_circle().encode(
                                x = alt.X(var_sel, scale = alt.Scale(domain = [min(fig_data[var_sel]), max(fig_data[var_sel])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                y = alt.Y(yy, scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                tooltip = [yy, var_sel, "Index"]
                            )
                            st.altair_chart(fig + fig.transform_loess(var_sel, yy).mark_line(size = 2, color = "darkred"), use_container_width=True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("dv_scatterplot")))
                        else: st.error("ERROR: Please select a numeric variable for the y-axis!")   
                    with a5:
                        st.write('**Histogram**')
                        binNo = st.slider("Select maximum number of bins", 5, 100, 25)
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
                        boxplot = alt.Chart(boxplot_data).mark_boxplot(size = 100, color = "#1f77b4", median = dict(color = "darkred"),).encode(
                            y = alt.Y(var_sel, scale = alt.Scale(zero = False)),
                            tooltip = [var_sel, "Index"]
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
                        qqplot_data = qqplot_data.sort_values(by = [var_sel])
                        qqplot_data["Theoretical quantiles"] = stats.probplot(var_values, dist="norm")[0][0]
                        qqplot = alt.Chart(qqplot_data).mark_circle(size=20).encode(
                            x = alt.X("Theoretical quantiles", title = "theoretical quantiles", scale = alt.Scale(domain = [min(qqplot_data["Theoretical quantiles"]), max(qqplot_data["Theoretical quantiles"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            y = alt.Y(var_sel, title = str(var_sel), scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            tooltip = [var_sel, "Theoretical quantiles", "Index"]
                        )
                        st.altair_chart(qqplot + qqplot.transform_regression('Theoretical quantiles', var_sel).mark_line(size = 2, color = "darkred"), use_container_width = True)
                        if sett_hints:
                            st.info(str(fc.learning_hints("dv_qqplot")))
                else: st.error("ERROR: Please select a numeric variable!") 
                
            else: st.error("ERROR: No data available for Data Visualization!")  

            # scatter matrix
            #Check If variables are numeric
            num_cols=[]
            for column in df:            
                if df[column].dtypes in ('float', 'float64', 'int','int64'):                    
                    num_cols.append(column)
            if len(num_cols)>1:
                show_scatter_matrix=st.checkbox('Show scatter matrix',value=False,key= st.session_state['key'])
                if show_scatter_matrix==True:
                    multi_var_sel = st.multiselect('Select variables for scatter matrix', num_cols, num_cols)

                    if len(multi_var_sel)<2:
                        st.error("ERROR: Please choose at least two variables fro a scatterplot")
                    else:
                       #Plot scatter matrix:
                        scatter_matrix=alt.Chart(df[multi_var_sel]).mark_circle().encode(
                            x=alt.X(alt.repeat("column"), type='quantitative'),
                            y=alt.Y(alt.repeat("row"), type='quantitative')
                        ).properties(
                                width=150,
                                height=150
                        ).repeat(
                                row=multi_var_sel,
                                column=multi_var_sel
                        ).interactive()
                        st.altair_chart(scatter_matrix, use_container_width=True)
                             
    #------------------------------------------------------------------------------------------

    # REGRESSION
    
    if analysis_type == "Regression":

        #++++++++++++++++++++++++++++++++++++++++++++
        # MACHINE LEARNING (PREDICTIVE DATA ANALYSIS)

        st.write("")
        st.write("")
        
        data_machinelearning_container = st.container()
        with data_machinelearning_container:
            st.header("**Multivariate data modelling**")
            st.markdown("Go for creating predictive models of your data using classical and machine learning techniques!  STATY will take care of the modelling for you, so you can put your focus on results interpretation and communication! ")

            ml_settings = st.expander("Specify models ", expanded = False)
            with ml_settings:
                
                # Initial status for running models
                run_models = False
                sb_ML_alg = "NA"
                do_hypTune = "No"
                do_modval = "No"
                do_hypTune_no = "No hyperparameter tuning"
                final_hyPara_values="None"
                model_val_results = None
                model_full_results = None
                gam_finalPara = None
                brt_finalPara = None
                brt_tuning_results = None
                rf_finalPara = None
                rf_tuning_results = None
                ann_finalPara = None
                ann_tuning_results = None
                MLR_intercept = None
                MLR_cov_type = None
                MLR_finalPara = None
                MLR_model = "OLS"
                LR_cov_type = None
                LR_finalPara = None
                LR_finalPara = None

                if df.shape[1] > 0 and df.shape[0] > 0:
                    #--------------------------------------------------------------------------------------
                    # GENERAL SETTINGS
                
                    st.markdown("**Variable selection**")
                    
                    # Variable categories
                    df_summary_model = fc.data_summary(df)
                    var_cat = df_summary_model["Variable types"].loc["category"]
                    
                    # Response variable
                    response_var_options = df.columns
                    response_var = st.selectbox("Select response variable", response_var_options, on_change=in_wid_change)
                    
                    # Check if response variable is numeric and has no NAs
                    response_var_message_num = False
                    response_var_message_na = False
                    response_var_message_cat = False

                    if var_cat.loc[response_var] == "string/binary" or var_cat.loc[response_var] == "bool/binary":
                        response_var_message_num = "ERROR: Please transform the binary response variable into a numeric binary categorization in data processing preferences!"
                    elif var_cat.loc[response_var] == "string/categorical" or var_cat.loc[response_var] == "other" or var_cat.loc[response_var] == "string/single":
                        response_var_message_num = "ERROR: Please select a numeric or binary response variable!"
                    elif var_cat.loc[response_var] == "categorical":
                        response_var_message_cat = "WARNING: Non-continuous variables are treated as continuous!"
                    
                    if response_var_message_num != False:
                        st.error(response_var_message_num)
                    if response_var_message_cat != False:
                        st.warning(response_var_message_cat)

                    # Continue if everything is clean for response variable
                    if response_var_message_num == False and response_var_message_na == False:
                        # Select explanatory variables
                        expl_var_options = df.columns
                        expl_var_options = expl_var_options[expl_var_options.isin(df.drop(response_var, axis = 1).columns)]
                        expl_var = st.multiselect("Select explanatory variables", expl_var_options, on_change=in_wid_change)
                        var_list = list([response_var]) + list(expl_var)

                        # Check if explanatory variables are numeric
                        expl_var_message_num = False
                        expl_var_message_na = False
                        if any(a for a in df[expl_var].dtypes if a != "float64" and a != "float32" and a != "int64" and a != "int64" and a != "bool" and a != "int32"): 
                            expl_var_not_num = df[expl_var].select_dtypes(exclude=["int64", "int32", "float64", "float32", "bool"]).columns
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

                            st.markdown("**Specify modelling algorithms**")

                            # Select algorithms based on chosen response variable
                            # Binary (has to be integer or float)
                            if var_cat.loc[response_var] == "binary":
                                algorithms = ["Multiple Linear Regression", "Logistic Regression", "Generalized Additive Models", "Random Forest", "Boosted Regression Trees", "Artificial Neural Networks"]
                                response_var_type = "binary"
                            
                            # Multi-class (has to be integer, currently treated as continuous response)
                            elif var_cat.loc[response_var] == "categorical":
                                algorithms = ["Multiple Linear Regression", "Generalized Additive Models", "Random Forest", "Boosted Regression Trees", "Artificial Neural Networks"]
                                response_var_type = "continuous"
                            # Continuous
                            elif var_cat.loc[response_var] == "numeric":
                                algorithms = ["Multiple Linear Regression", "Generalized Additive Models", "Random Forest", "Boosted Regression Trees", "Artificial Neural Networks"]
                                response_var_type = "continuous"

                            alg_list = list(algorithms)
                            sb_ML_alg = st.multiselect("Select modelling techniques", alg_list, alg_list, on_change=in_wid_change)
                            
                            # MLR + binary info message
                            if any(a for a in sb_ML_alg if a == "Multiple Linear Regression") and response_var_type == "binary":
                                st.warning("WARNING: For Multiple Linear Regression only the full model output will be determined.")

                            st.markdown("**Model-specific settings**")
                            # Multiple Linear Regression settings
                            if any(a for a in sb_ML_alg if a == "Multiple Linear Regression"):
                                MLR_finalPara = pd.DataFrame(index = ["value"], columns = ["intercept", "covType"])
                                MLR_intercept = "Yes"
                                MLR_cov_type = "non-robust"
                                MLR_finalPara["intercept"] = MLR_intercept
                                MLR_finalPara["covType"] = MLR_cov_type
                                if st.checkbox("Adjust settings for Multiple Linear Regression", on_change=in_wid_change):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        MLR_intercept = st.selectbox("Include intercept", ["Yes", "No"], on_change=in_wid_change)
                                    with col2:
                                        MLR_cov_type = st.selectbox("Covariance type", ["non-robust", "HC0", "HC1", "HC2", "HC3"], on_change=in_wid_change)
                                    MLR_finalPara["intercept"] = MLR_intercept
                                    MLR_finalPara["covType"] = MLR_cov_type
                                    st.write("") 

                            # Logistic Regression settings
                            if any(a for a in sb_ML_alg if a == "Logistic Regression"):
                                LR_finalPara = pd.DataFrame(index = ["value"], columns = ["intercept", "covType"])
                                LR_intercept = "Yes"
                                LR_cov_type = "non-robust"
                                LR_finalPara["intercept"] = LR_intercept
                                LR_finalPara["covType"] = LR_cov_type
                                if st.checkbox("Adjust settings for Logistic Regression", on_change=in_wid_change):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        LR_intercept = st.selectbox("Include intercept   ", ["Yes", "No"], on_change=in_wid_change)
                                    with col2:
                                        LR_cov_type = st.selectbox("Covariance type", ["non-robust", "HC0"], on_change=in_wid_change)
                                    LR_finalPara["intercept"] = LR_intercept
                                    LR_finalPara["covType"] = LR_cov_type
                                    st.write("") 
                            
                            # Generalized Additive Models settings
                            if any(a for a in sb_ML_alg if a == "Generalized Additive Models"):
                                gam_finalPara = pd.DataFrame(index = ["value"], columns = ["intercept", "number of splines", "spline order", "lambda"])
                                gam_finalPara["intercept"] = "Yes"
                                gam_finalPara["number of splines"] = 20
                                gam_finalPara["spline order"] = 3
                                gam_finalPara["lambda"] = 0.6
                                gam_lam_search = "No"
                                if st.checkbox("Adjust settings for Generalized Additive Models", on_change=in_wid_change):
                                    gam_finalPara = pd.DataFrame(index = ["value"], columns = ["intercept", "number of splines", "spline order", "lambda"])
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        gam_intercept = st.selectbox("Include intercept ", ["Yes", "No"], on_change=in_wid_change)
                                    gam_finalPara["intercept"] = gam_intercept
                                    with col2:
                                        gam_lam_search = st.selectbox("Search for lambda ", ["No", "Yes"], on_change=in_wid_change)
                                    if gam_lam_search == "Yes":
                                        ls_col1, ls_col2, ls_col3 = st.columns(3)
                                        with ls_col1:
                                            ls_min = st.number_input("Minimum lambda value", value=0.001, step=1e-3, min_value=0.001, format="%.3f", on_change=in_wid_change) 
                                        with ls_col2:
                                            ls_max = st.number_input("Maximum lambda value", value=100.000, step=1e-3, min_value=0.002, format="%.3f", on_change=in_wid_change)
                                        with ls_col3:
                                            ls_number = st.number_input("Lambda values per variable", value=50, min_value=2, on_change=in_wid_change)
                                        if ls_number**len(expl_var) > 10000:
                                            st.warning("WARNING: Your grid has " + str(ls_number**len(expl_var)) + " combinations. Please note that searching for lambda will take a lot of time!")
                                        else:
                                            st.info("Your grid has " + str(ls_number**len(expl_var)) + " combinations.")
                                    if gam_lam_search == "No":
                                        gam_col1, gam_col2, gam_col3 = st.columns(3)
                                    if gam_lam_search == "Yes":
                                        gam_col1, gam_col2= st.columns(2)
                                    gam_nos_values = []
                                    gam_so_values = []
                                    gam_lam_values = []
                                    for gset in range(0,len(expl_var)):
                                        var_name = expl_var[gset]
                                        with gam_col1:
                                            nos = st.number_input("Number of splines (" + var_name + ")", value = 20, min_value=1, on_change=in_wid_change) 
                                            gam_nos_values.append(nos)
                                        with gam_col2:
                                            so = st.number_input("Spline order (" + var_name + ")", value = 3, min_value=3, on_change=in_wid_change) 
                                            gam_so_values.append(so)
                                        if gam_lam_search == "No":
                                            with gam_col3: 
                                                lam = st.number_input("Lambda (" + var_name + ")", value = 0.6, min_value=0.001, step=1e-3, format="%.3f", on_change=in_wid_change) 
                                                gam_lam_values.append(lam) 
                                        if nos <= so:
                                            st.error("ERROR: Please make sure that the number of splines is greater than the spline order for "+ str(expl_var[gset]) + "!")
                                            return
                                    if gam_lam_search == "Yes":
                                        lam = np.round(np.linspace(ls_min, ls_max, ls_number),3)
                                        if len(expl_var) == 1:
                                            gam_lam_values = lam
                                        else:
                                            gam_lam_values = [lam] * len(expl_var)
                                        
                                    gam_finalPara.at["value", "number of splines"] = gam_nos_values
                                    gam_finalPara.at["value","spline order"] = gam_so_values  
                                    gam_finalPara.at["value","lambda"] = gam_lam_values
                                    st.write("")  

                            # Save hyperparameter values for machine learning methods
                            final_hyPara_values = {}

                            # Random Forest settings
                            if any(a for a in sb_ML_alg if a == "Random Forest"):
                                rf_finalPara = pd.DataFrame(index = ["value"], columns = ["number of trees", "maximum tree depth", "maximum number of features", "sample rate"])
                                rf_finalPara["number of trees"] = [100]
                                rf_finalPara["maximum tree depth"] = [None]
                                rf_finalPara["maximum number of features"] = [len(expl_var)]
                                rf_finalPara["sample rate"] = [0.99]
                                final_hyPara_values["rf"] = rf_finalPara
                                if st.checkbox("Adjust settings for Random Forest ", on_change=in_wid_change):  
                                    col1, col2 = st.columns(2)
                                    col3, col4 = st.columns(2)
                                    with col1:
                                        rf_finalPara["number of trees"] = st.number_input("Number of trees", value=100, step=1, min_value=1, on_change=in_wid_change) 
                                    with col3:
                                        rf_mtd_sel = st.selectbox("Specify maximum tree depth ", ["No", "Yes"])
                                        if rf_mtd_sel == "No":
                                            rf_finalPara["maximum tree depth"] = [None]
                                        if rf_mtd_sel == "Yes":
                                            rf_finalPara["maximum tree depth"] = st.slider("Maximum tree depth ", value=20, step=1, min_value=1, max_value=50, on_change=in_wid_change)
                                    if len(expl_var) >1:
                                        with col4:
                                            rf_finalPara["maximum number of features"] = st.slider("Maximum number of features ", value=len(expl_var), step=1, min_value=1, max_value=len(expl_var), on_change=in_wid_change)
                                        with col2:
                                            rf_finalPara["sample rate"] = st.slider("Sample rate ", value=0.99, step=0.01, min_value=0.5, max_value=0.99, on_change=in_wid_change)
                                    else:
                                        with col2:
                                            rf_finalPara["sample rate"] = st.slider("Sample rate ", value=0.99, step=0.01, min_value=0.5, max_value=0.99, on_change=in_wid_change)
                                    final_hyPara_values["rf"] = rf_finalPara 
                                    st.write("") 

                            # Boosted Regression Trees settings 
                            if any(a for a in sb_ML_alg if a == "Boosted Regression Trees"):
                                brt_finalPara = pd.DataFrame(index = ["value"], columns = ["number of trees", "learning rate", "maximum tree depth", "sample rate"])
                                brt_finalPara["number of trees"] = [100]
                                brt_finalPara["learning rate"] = [0.1]
                                brt_finalPara["maximum tree depth"] = [3]
                                brt_finalPara["sample rate"] = [1]
                                final_hyPara_values["brt"] = brt_finalPara
                                if st.checkbox("Adjust settings for Boosted Regression Trees ", on_change=in_wid_change):
                                    col1, col2 = st.columns(2)
                                    col3, col4 = st.columns(2)
                                    with col1:
                                        brt_finalPara["number of trees"] = st.number_input("Number of trees ", value=100, step=1, min_value=1, on_change=in_wid_change) 
                                    with col2:
                                        brt_finalPara["learning rate"] = st.slider("Learning rate ", value=0.1, min_value=0.001, max_value=0.1 , step=1e-3, format="%.3f", on_change=in_wid_change)
                                    with col3:
                                        brt_finalPara["maximum tree depth"] = st.slider("Maximum tree depth ", value=3, step=1, min_value=1, max_value=30, on_change=in_wid_change)
                                    with col4:
                                        brt_finalPara["sample rate"] = st.slider("Sample rate ", value=1.0, step=0.01, min_value=0.5, max_value=1.0, on_change=in_wid_change)
                                    final_hyPara_values["brt"] = brt_finalPara
                                    st.write("")  

                            # Artificial Neural Networks settings 
                            if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                                ann_finalPara = pd.DataFrame(index = ["value"], columns = ["weight optimization solver", "maximum number of iterations", "activation function", "hidden layer sizes", "learning rate", "L² regularization"])
                                ann_finalPara["weight optimization solver"] = ["adam"]
                                ann_finalPara["maximum number of iterations"] = [200]
                                ann_finalPara["activation function"] = ["relu"]
                                ann_finalPara["hidden layer sizes"] = [(100,)]
                                ann_finalPara["learning rate"] = [0.001]
                                ann_finalPara["L² regularization"] = [0.0001]
                                final_hyPara_values["ann"] = ann_finalPara
                                if st.checkbox("Adjust settings for Artificial Neural Networks ", on_change=in_wid_change): 
                                    col1, col2 = st.columns(2)
                                    col3, col4 = st.columns(2)
                                    col5, col6 = st.columns(2)
                                    with col1:
                                        ann_finalPara["weight optimization solver"] = st.selectbox("Weight optimization solver ", ["adam"], on_change=in_wid_change)
                                    with col2:
                                        ann_finalPara["activation function"] = st.selectbox("Activation function ", ["relu", "identity", "logistic", "tanh"], on_change=in_wid_change)
                                    with col3:
                                        ann_finalPara["maximum number of iterations"] = st.slider("Maximum number of iterations ", value=200, step=1, min_value=10, max_value=1000, on_change=in_wid_change) 
                                    with col4:
                                        ann_finalPara["learning rate"] = st.slider("Learning rate  ", min_value=0.0001, max_value=0.01, value=0.001, step=1e-4, format="%.4f", on_change=in_wid_change)
                                    with col5:
                                        number_hidden_layers = st.selectbox("Number of hidden layers", [1, 2, 3], on_change=in_wid_change)
                                        if number_hidden_layers == 1:
                                            number_nodes1 = st.slider("Number of nodes in hidden layer", 5, 500, 100, on_change=in_wid_change)
                                            ann_finalPara["hidden layer sizes"] = [(number_nodes1,)]
                                        if number_hidden_layers == 2:
                                            number_nodes1 = st.slider("Number of neurons in first hidden layer", 5, 500, 100, on_change=in_wid_change)
                                            number_nodes2 = st.slider("Number of neurons in second hidden layer", 5, 500, 100, on_change=in_wid_change)
                                            ann_finalPara["hidden layer sizes"] = [(number_nodes1,number_nodes2,)]
                                        if number_hidden_layers == 3:
                                            number_nodes1 = st.slider("Number of neurons in first hidden layer", 5, 500, 100, on_change=in_wid_change)
                                            number_nodes2 = st.slider("Number of neurons in second hidden layer", 5, 500, 100, on_change=in_wid_change)
                                            number_nodes3 = st.slider("Number of neurons in third hidden layer", 5, 500, 100, on_change=in_wid_change)
                                            ann_finalPara["hidden layer sizes"] = [(number_nodes1,number_nodes2,number_nodes3,)]
                                    with col6:
                                        ann_finalPara["L² regularization"] = st.slider("L² regularization  ", min_value=0.00001, max_value=0.001, value=0.0001, step=1e-5, format="%.5f", on_change=in_wid_change)                                

                            #--------------------------------------------------------------------------------------
                            # HYPERPARAMETER TUNING SETTINGS
                            
                            if len(sb_ML_alg) >= 1:

                                # Depending on algorithm selection different hyperparameter settings are shown
                                if any(a for a in sb_ML_alg if a == "Random Forest") or any(a for a in sb_ML_alg if a == "Boosted Regression Trees") or any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                                    # General settings
                                    st.markdown("**Hyperparameter-tuning settings**")
                                    do_hypTune = st.selectbox("Use hyperparameter-tuning", ["No", "Yes"], on_change=in_wid_change)
                                
                                    # Save hyperparameter values for all algorithms
                                    hyPara_values = {}
                                    
                                    # No hyperparameter-tuning
                                    if do_hypTune == "No":
                                        do_hypTune_no = "Default hyperparameter values are used!"

                                    # Hyperparameter-tuning 
                                    elif do_hypTune == "Yes":
                                        st.warning("WARNING: Hyperparameter-tuning can take a lot of time! For tips, please [contact us](mailto:staty@quant-works.de?subject=Staty-App).")
                                        
                                        # Further general settings
                                        hypTune_method = st.selectbox("Hyperparameter-search method", ["random grid-search", "grid-search", "Bayes optimization", "sequential model-based optimization"], on_change=in_wid_change)
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            hypTune_nCV = st.slider("Select number for n-fold cross-validation", 2, 10, 5, on_change=in_wid_change)

                                        if hypTune_method == "random grid-search" or hypTune_method == "Bayes optimization" or hypTune_method == "sequential model-based optimization":
                                            with col2:
                                                hypTune_iter = st.slider("Select number of iterations for search", 20, 1000, 20, on_change=in_wid_change)
                                        else:
                                            hypTune_iter = False

                                        st.markdown("**Model-specific tuning settings**")
                                        # Random Forest settings
                                        if any(a for a in sb_ML_alg if a == "Random Forest"):
                                            rf_tunePara = pd.DataFrame(index = ["min", "max"], columns = ["number of trees", "maximum tree depth", "maximum number of features", "sample rate"])
                                            rf_tunePara["number of trees"] = [50, 500]
                                            rf_tunePara["maximum tree depth"] = [None, None]
                                            rf_tunePara["maximum number of features"] = [1, len(expl_var)]
                                            rf_tunePara["sample rate"] = [0.8, 0.99]
                                            hyPara_values["rf"] = rf_tunePara
                                            if st.checkbox("Adjust tuning settings for Random Forest", on_change=in_wid_change):
                                                col1, col2 = st.columns(2)
                                                col3, col4 = st.columns(2)
                                                with col1:
                                                    rf_tunePara["number of trees"] = st.slider("Range for number of trees ", 50, 1000, [50, 500], on_change=in_wid_change)
                                                with col3:
                                                    rf_mtd_choice = st.selectbox("Specify maximum tree depth", ["No", "Yes"], on_change=in_wid_change)
                                                    if rf_mtd_choice == "Yes":
                                                        rf_tunePara["maximum tree depth"] = st.slider("Range for maximum tree depth ", 1, 50, [2, 10], on_change=in_wid_change)
                                                    else:
                                                        rf_tunePara["maximum tree depth"] = [None, None]
                                                with col4:
                                                    if len(expl_var) > 1:
                                                        rf_tunePara["maximum number of features"] = st.slider("Range for maximum number of features", 1, len(expl_var), [1, len(expl_var)], on_change=in_wid_change)
                                                    else:
                                                        rf_tunePara["maximum number of features"] = [1,1]
                                                with col2:
                                                    rf_tunePara["sample rate"] = st.slider("Range for sample rate ", 0.5, 0.99, [0.8, 0.99], on_change=in_wid_change)
                                                hyPara_values["rf"] = rf_tunePara

                                        # Boosted Regression Trees settings
                                        if any(a for a in sb_ML_alg if a == "Boosted Regression Trees"):
                                            brt_tunePara = pd.DataFrame(index = ["min", "max"], columns = ["number of trees", "learning rate", "maximum tree depth", "sample rate"])
                                            brt_tunePara["number of trees"] = [50, 500]
                                            brt_tunePara["learning rate"] = [0.001, 0.010]
                                            brt_tunePara["maximum tree depth"] = [2, 10]
                                            brt_tunePara["sample rate"] = [0.8, 1.0]
                                            hyPara_values["brt"] = brt_tunePara
                                            if st.checkbox("Adjust tuning settings for Boosted Regression Trees", on_change=in_wid_change):
                                                col1, col2 = st.columns(2)
                                                col3, col4 = st.columns(2)
                                                with col1:
                                                    brt_tunePara["number of trees"] = st.slider("Range for number of trees", 50, 1000, [50, 500], on_change=in_wid_change)
                                                with col2:
                                                    brt_tunePara["learning rate"] = st.slider("Range for learning rate", 0.001, 0.1, [0.001, 0.02], step=1e-3, format="%.3f", on_change=in_wid_change) 
                                                with col3:
                                                    brt_tunePara["maximum tree depth"] = st.slider("Range for maximum tree depth", 1, 30, [2, 10], on_change=in_wid_change)
                                                with col4:
                                                    brt_tunePara["sample rate"] = st.slider("Range for sample rate", 0.5, 1.0, [0.8, 1.0], on_change=in_wid_change)
                                                hyPara_values["brt"] = brt_tunePara

                                        # Artificial Neural Networks settings
                                        if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                                            ann_tunePara = pd.DataFrame(index = ["min", "max"], columns = ["weight optimization solver", "maximum number of iterations", "activation function", "number of hidden layers", "nodes per hidden layer", "learning rate","L² regularization"])# "learning rate schedule", "momentum", "epsilon"])
                                            ann_tunePara["weight optimization solver"] = list([["adam"], "NA"])
                                            ann_tunePara["maximum number of iterations"] = [100, 200]
                                            ann_tunePara["activation function"] = list([["relu"], "NA"])
                                            ann_tunePara["number of hidden layers"] = list([1, "NA"])
                                            ann_tunePara["nodes per hidden layer"] = [50, 100]
                                            ann_tunePara["learning rate"] = [0.0001, 0.002]
                                            ann_tunePara["L² regularization"] = [0.00001, 0.0002]
                                            hyPara_values["ann"] = ann_tunePara
                                            if st.checkbox("Adjust tuning settings for Artificial Neural Networks", on_change=in_wid_change):
                                                col1, col2 = st.columns(2)
                                                col3, col4 = st.columns(2)
                                                col5, col6 = st.columns(2)
                                                with col1:
                                                    weight_opt_list = st.selectbox("Weight optimization solver  ", ["adam"], on_change=in_wid_change)
                                                    if len(weight_opt_list) == 0:
                                                        weight_opt_list = ["adam"]
                                                        st.warning("WARNING: Default value used 'adam'")
                                                    ann_tunePara["weight optimization solver"] = list([[weight_opt_list], "NA"])
                                                with col2:
                                                    ann_tunePara["maximum number of iterations"] = st.slider("Maximum number of iterations (epochs) ", 10, 1000, [100, 200], on_change=in_wid_change)
                                                with col3:
                                                    act_func_list = st.multiselect("Activation function ", ["identity", "logistic", "tanh", "relu"], ["relu"], on_change=in_wid_change)
                                                    if len(act_func_list) == 0:
                                                        act_func_list = ["relu"]
                                                        st.warning("WARNING: Default value used 'relu'")
                                                    ann_tunePara["activation function"] = list([act_func_list, "NA"])
                                                with col5:
                                                    number_hidden_layers = st.selectbox("Number of hidden layers ", [1, 2, 3], on_change=in_wid_change)
                                                    ann_tunePara["number of hidden layers"]  = list([number_hidden_layers, "NA"])
                                                    # Cases for hidden layers
                                                    if number_hidden_layers == 1:
                                                        ann_tunePara["nodes per hidden layer"] = st.slider("Number of nodes in hidden layer ", 5, 500, [50, 100], on_change=in_wid_change)
                                                    if number_hidden_layers == 2:
                                                        number_nodes1 = st.slider("Number of neurons in first hidden layer ", 5, 500, [50, 100], on_change=in_wid_change)
                                                        number_nodes2 = st.slider("Number of neurons in second hidden layer ", 5, 500, [50, 100], on_change=in_wid_change)
                                                        min_nodes = list([number_nodes1[0], number_nodes2[0]])
                                                        max_nodes = list([number_nodes1[1], number_nodes2[1]])
                                                        ann_tunePara["nodes per hidden layer"] = list([min_nodes, max_nodes])
                                                    if number_hidden_layers == 3:
                                                        number_nodes1 = st.slider("Number of neurons in first hidden layer ", 5, 500, [50, 100], on_change=in_wid_change)
                                                        number_nodes2 = st.slider("Number of neurons in second hidden layer ", 5, 500, [50, 100], on_change=in_wid_change)
                                                        number_nodes3 = st.slider("Number of neurons in third hidden layer ", 5, 500, [50, 100], on_change=in_wid_change)
                                                        min_nodes = list([number_nodes1[0], number_nodes2[0], number_nodes3[0]])
                                                        max_nodes = list([number_nodes1[1], number_nodes2[1], number_nodes3[1]])
                                                        ann_tunePara["nodes per hidden layer"] = list([min_nodes, max_nodes])
                                                with col6:
                                                    if weight_opt_list == "adam": 
                                                        ann_tunePara["learning rate"] = st.slider("Range for learning rate ", 0.0001, 0.01, [0.0001, 0.002], step=1e-4, format="%.4f", on_change=in_wid_change)
                                                with col4:
                                                    ann_tunePara["L² regularization"] = st.slider("L² regularization parameter ", 0.0, 0.001, [0.00001, 0.0002], step=1e-5, format="%.5f", on_change=in_wid_change)
                                                hyPara_values["ann"] = ann_tunePara
                                        
                                #--------------------------------------------------------------------------------------
                                # VALIDATION SETTINGS

                                st.markdown("**Validation settings**")
                                do_modval= st.selectbox("Use model validation", ["No", "Yes"], on_change=in_wid_change)

                                if do_modval == "Yes":
                                    col1, col2 = st.columns(2)
                                    # Select training/ test ratio
                                    with col1: 
                                        train_frac = st.slider("Select training data size", 0.5, 0.95, 0.8, on_change=in_wid_change)

                                    # Select number for validation runs
                                    with col2:
                                        val_runs = st.slider("Select number for validation runs", 5, 100, 10, on_change=in_wid_change)

                                #--------------------------------------------------------------------------------------
                                # PREDICTION SETTINGS

                                st.markdown("**Model predictions**")
                                do_modprednew = st.selectbox("Use model prediction for new data", ["No", "Yes"], on_change=in_wid_change)

                                if do_modprednew == "Yes":
                                    # Upload new data
                                    new_data_pred = st.file_uploader("  ", type=["csv", "txt"], on_change=in_wid_change)

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
                                        
                                        # Check if explanatory variables are available as columns
                                        expl_list = []
                                        for expl_incl in expl_var:
                                            if expl_incl not in df_new.columns:
                                                expl_list.append(expl_incl)
                                        if expl_list:
                                            st.error("ERROR: Some variables are missing in new data: "+ ', '.join(expl_list))
                                            return
                                        else:
                                            st.info("All variables are available for predictions!")
                                        
                                        # Check if NAs are present and delete them automatically
                                        if df_new.iloc[list(pd.unique(np.where(df_new.isnull())[0]))].shape[0] == 0:
                                            st.empty()
                                        else:
                                            df_new = df_new[expl_var].dropna()
                                            st.warning("WARNING: Your new data set includes NAs. Rows with NAs are automatically deleted!")
                                        df_new = df_new[expl_var]
                                    
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
                                    excel_file.close()
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
                                            excel_file.close()
                                            excel_file = output.getvalue()
                                            b64 = base64.b64encode(excel_file)
                                            dl_file_name= "New data for predictions__" + df_name + ".xlsx"
                                            st.markdown(
                                                f"""
                                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download new data for predictions</a>
                                            """,
                                            unsafe_allow_html=True)
                                        st.write("")

                                # Show machine learning summary
                                if st.checkbox('Show a summary of machine learning settings', value = False): 
                                    
                                    #--------------------------------------------------------------------------------------
                                    # ALGORITHMS
                                    
                                    st.write("Algorithms summary:")
                                    st.write("- Models:",  ', '.join(sb_ML_alg))
                                    if any(a for a in sb_ML_alg if a == "Multiple Linear Regression"):
                                        # st.write("- Multiple Linear Regression model: ", MLR_model)
                                        st.write("- Multiple Linear Regression including intercept: ", MLR_intercept)
                                        st.write("- Multiple Linear Regression covariance type: ", MLR_cov_type)
                                    if any(a for a in sb_ML_alg if a == "Logistic Regression"):
                                        st.write("- Logistic Regression including intercept: ", LR_intercept)
                                        st.write("- Logistic Regression covariance type: ", LR_cov_type)
                                    if any(a for a in sb_ML_alg if a == "Generalized Additive Models"):
                                        st.write("- Generalized Additive Models parameters: ")
                                        st.write(gam_finalPara)
                                    if any(a for a in sb_ML_alg if a == "Random Forest") and do_hypTune == "No":
                                        st.write("- Random Forest parameters: ")
                                        st.write(rf_finalPara)
                                    if any(a for a in sb_ML_alg if a == "Boosted Regression Trees") and do_hypTune == "No":
                                        st.write("- Boosted Regression Trees parameters: ")
                                        st.write(brt_finalPara)
                                    if any(a for a in sb_ML_alg if a == "Artificial Neural Networks") and do_hypTune == "No":
                                        st.write("- Artificial Neural Networks parameters: ")
                                        st.write(ann_finalPara)
                                    st.write("")

                                    #--------------------------------------------------------------------------------------
                                    # SETTINGS

                                    # Hyperparameter settings summary
                                    if any(a for a in sb_ML_alg if a == "Artificial Neural Networks" or a == "Boosted Regression Trees" or a == "Random Forest"):
                                        st.write("Hyperparameter-tuning settings summary:")
                                        if do_hypTune == "No":
                                            st.write("- ", do_hypTune_no)
                                            st.write("")
                                        if do_hypTune == "Yes":
                                            st.write("- Search method:", hypTune_method)
                                            st.write("- ", hypTune_nCV, "-fold cross-validation")
                                            if hypTune_method == "random grid-search" or hypTune_method == "Bayes optimization" or hypTune_method == "sequential model-based optimization":
                                                st.write("- ", hypTune_iter, "iterations in search")
                                                st.write("")
                                            # Random Forest summary
                                            if any(a for a in sb_ML_alg if a == "Random Forest"):
                                                st.write("Random Forest tuning settings summary:")
                                                st.write(rf_tunePara)
                                            # Boosted Regression Trees summary
                                            if any(a for a in sb_ML_alg if a == "Boosted Regression Trees"):
                                                st.write("Boosted Regression Trees tuning settings summary:")
                                                st.write(brt_tunePara)
                                            # Artificial Neural Networks summary
                                            if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                                                st.write("Artificial Neural Networks tuning settings summary:")
                                                st.write(ann_tunePara.style.format({"L² regularization": "{:.5}"}))
                                                #st.caption("** Learning rate is only used in adam")
                                            st.write("")

                                    # General settings summary
                                    st.write("General settings summary:")
                                    st.write("- Response variable type: ", response_var_type)
                                    # Modelling formula
                                    if expl_var != False:
                                        st.write("- Modelling formula:", response_var, "~",  ' + '.join(expl_var))
                                    if do_modval == "Yes":
                                        # Train/ test ratio
                                        if train_frac != False:
                                            st.write("- Train/ test ratio:", str(round(train_frac*100)), "% / ", str(round(100-train_frac*100)), "%")
                                        # Validation runs
                                        if val_runs != False:
                                            st.write("- Validation runs:", str(val_runs))
                                    st.write("")
                            
                                #--------------------------------------------------------------------------------------
                                # RUN MODELS

                                # Models are run on button click
                                st.write("")
                                run_models = st.button("Run models")
                                st.write("")
                                
                                if run_models:

                                    # Check if new data available
                                    if do_modprednew == "Yes":
                                        if new_data_pred is None:
                                            st.error("ERROR: Please upload new data for additional model predictions or select 'No'!")
                                            return

                                    #Hyperparameter   
                                    if do_hypTune == "Yes":

                                        # Tuning
                                        model_tuning_results = ml.model_tuning(df, sb_ML_alg, hypTune_method, hypTune_iter, hypTune_nCV, hyPara_values, response_var_type, response_var, expl_var)
                                    
                                        # Save final hyperparameters
                                        # Random Forest
                                        if any(a for a in sb_ML_alg if a == "Random Forest"):
                                            rf_tuning_results = model_tuning_results["rf tuning"]
                                            rf_finalPara = pd.DataFrame(index = ["value"], columns = ["number of trees", "maximum tree depth", "maximum number of features", "sample rate"])
                                            rf_finalPara["number of trees"] = [rf_tuning_results.loc["value"]["number of trees"]]
                                            if [rf_tuning_results.loc["value"]["maximum tree depth"]][0] == "None":
                                                rf_finalPara["maximum tree depth"] = None
                                            else:
                                                rf_finalPara["maximum tree depth"] = [rf_tuning_results.loc["value"]["maximum tree depth"]]
                                            rf_finalPara["maximum number of features"] = [rf_tuning_results.loc["value"]["maximum number of features"]]
                                            rf_finalPara["sample rate"] = [rf_tuning_results.loc["value"]["sample rate"]]
                                            final_hyPara_values["rf"] = rf_finalPara
                                        # Boosted Regression Trees
                                        if any(a for a in sb_ML_alg if a == "Boosted Regression Trees"):
                                            brt_tuning_results = model_tuning_results["brt tuning"]
                                            brt_finalPara = pd.DataFrame(index = ["value"], columns = ["number of trees", "learning rate", "maximum tree depth", "sample rate"])
                                            brt_finalPara["number of trees"] = [brt_tuning_results.loc["value"]["number of trees"]]
                                            brt_finalPara["learning rate"] = [brt_tuning_results.loc["value"]["learning rate"]]
                                            brt_finalPara["maximum tree depth"] = [brt_tuning_results.loc["value"]["maximum tree depth"]]
                                            brt_finalPara["sample rate"] = [brt_tuning_results.loc["value"]["sample rate"]]
                                            final_hyPara_values["brt"] = brt_finalPara
                                        # Artificial Neural Networks
                                        if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                                            ann_tuning_results = model_tuning_results["ann tuning"]
                                            ann_finalPara = pd.DataFrame(index = ["value"], columns = ["weight optimization solver", "maximum number of iterations", "activation function", "hidden layer sizes", "learning rate", "L² regularization"]) #"learning rate schedule", "momentum", "epsilon"])
                                            ann_finalPara["weight optimization solver"] = [ann_tuning_results.loc["value"]["weight optimization solver"]]
                                            ann_finalPara["maximum number of iterations"] = [ann_tuning_results.loc["value"]["maximum number of iterations"]]
                                            ann_finalPara["activation function"] = [ann_tuning_results.loc["value"]["activation function"]]
                                            ann_finalPara["hidden layer sizes"] = [ann_tuning_results.loc["value"]["hidden layer sizes"]]
                                            ann_finalPara["learning rate"] = [ann_tuning_results.loc["value"]["learning rate"]]
                                            #ann_finalPara["learning rate schedule"] = [ann_tuning_results.loc["value"]["learning rate schedule"]]
                                            #ann_finalPara["momentum"] = [ann_tuning_results.loc["value"]["momentum"]]
                                            ann_finalPara["L² regularization"] = [ann_tuning_results.loc["value"]["L² regularization"]]
                                            #ann_finalPara["epsilon"] = [ann_tuning_results.loc["value"]["epsilon"]]
                                            final_hyPara_values["ann"] = ann_finalPara

                                    # Lambda search for GAM
                                    if any(a for a in sb_ML_alg if a == "Generalized Additive Models"):
                                        if gam_lam_search == "Yes":
                                            st.info("Lambda search")
                                            my_bar = st.progress(0.0)
                                            progress = 0
                                            Y_data_gam = df[response_var]
                                            X_data_gam = df[expl_var]
                                            nos = gam_finalPara["number of splines"][0]
                                            so = gam_finalPara["spline order"][0]
                                            lams = gam_lam_values
                                            if response_var_type == "continuous":
                                                if gam_finalPara["intercept"][0] == "Yes":
                                                    gam_grid = LinearGAM(n_splines = nos, spline_order = so, fit_intercept = True).gridsearch(X_data_gam.values, Y_data_gam.values, lam=lams)
                                                    gam_finalPara.at["value", "lambda"] = gam_grid.lam
                                                if gam_finalPara["intercept"][0] == "No":
                                                    gam_grid = LinearGAM(n_splines = nos, spline_order = so, fit_intercept = False).gridsearch(X_data_gam.values, Y_data_gam.values, lam=lams)
                                                    gam_finalPara.at["value", "lambda"] = gam_grid.lam
                                            if response_var_type == "binary":
                                                if gam_finalPara["intercept"][0] == "Yes":
                                                    gam_grid = LogisticGAM(n_splines = nos, spline_order = so, fit_intercept = True).gridsearch(X_data_gam.values, Y_data_gam.values, lam=lams)
                                                    gam_finalPara.at["value", "lambda"] = gam_grid.lam
                                                if gam_finalPara["intercept"][0] == "No":
                                                    gam_grid = LogisticGAM(n_splines = nos, spline_order = so, fit_intercept = False).gridsearch(X_data_gam.values, Y_data_gam.values, lam=lams)
                                                    gam_finalPara.at["value", "lambda"] = gam_grid.lam
                                            progress += 1
                                            my_bar.progress(progress/1)
                                        
                                    # Model validation
                                    if do_modval == "Yes":
                                        model_val_results = ml.model_val(df, sb_ML_alg, MLR_model, train_frac, val_runs, response_var_type, response_var, expl_var, final_hyPara_values, gam_finalPara, MLR_finalPara, LR_finalPara)
                                    
                                    # Full model (depending on prediction for new data)
                                    if do_modprednew == "Yes":
                                        if new_data_pred is not None:
                                            if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                                                model_full_results, full_model_ann_sk = ml.model_full(df, df_new, sb_ML_alg, MLR_model, MLR_finalPara, LR_finalPara, response_var_type, response_var, expl_var, final_hyPara_values, gam_finalPara)
                                            else: 
                                                model_full_results = ml.model_full(df, df_new, sb_ML_alg, MLR_model, MLR_finalPara, LR_finalPara, response_var_type, response_var, expl_var, final_hyPara_values, gam_finalPara)
                                    if do_modprednew == "No":
                                        df_new = pd.DataFrame()
                                        if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                                            model_full_results, full_model_ann_sk = ml.model_full(df, df_new, sb_ML_alg, MLR_model, MLR_finalPara, LR_finalPara, response_var_type, response_var, expl_var, final_hyPara_values, gam_finalPara)
                                        else:
                                            model_full_results = ml.model_full(df, df_new, sb_ML_alg, MLR_model, MLR_finalPara, LR_finalPara, response_var_type, response_var, expl_var, final_hyPara_values, gam_finalPara)
                                    # Success message
                                    st.success('Models run successfully!')
                                    if do_hypTune == "Yes":
                                        st.session_state['model_tuning_results'] = model_tuning_results  
                                    st.session_state['model_full_results'] = model_full_results
                                    if do_modval == "Yes":  
                                        st.session_state['model_val_results'] = model_val_results  
                                    if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                                        st.session_state['full_model_ann_sk'] = full_model_ann_sk
                                        st.session_state['ann_finalPara'] = ann_finalPara
                                        if do_hypTune == "Yes":
                                            st.session_state['ann_tuning_results'] = ann_tuning_results                        
                else: st.error("ERROR: No data available for Modelling!") 

        #++++++++++++++++++++++
        # ML OUTPUT

        # Show only if models were run
        if st.session_state['model_full_results'] is not None and 'expl_var' in locals():

            model_tuning_results = st.session_state['model_tuning_results']
            model_full_results = st.session_state['model_full_results']
            model_val_results = st.session_state['model_val_results']

            if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                full_model_ann_sk = st.session_state['full_model_ann_sk']
                ann_finalPara = st.session_state['ann_finalPara']
                ann_tuning_results = st.session_state['ann_tuning_results']
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
                    corr_plot1 = (corr_plot + text).properties(width = 400, height = 400)
                    correlation_plot = correlation_plot.properties(padding = {"left": 50, "top": 5, "right": 5, "bottom": 50})
                    # hist_2d_plot = scat_plot.properties(height = 350)
                    if response_var_type == "continuous":
                        st.altair_chart(correlation_plot, use_container_width = True)
                    if response_var_type == "binary":
                        st.altair_chart(correlation_plot, use_container_width = True)
                    if sett_hints:
                        st.info(str(fc.learning_hints("mod_cor")))
                    st.write("")
                    
                    #-------------------------------------------------------------

                    # Continuous response variable
                    if response_var_type == "continuous":

                        # MLR specific output
                        if any(a for a in sb_ML_alg if a == "Multiple Linear Regression"):
                            st.markdown("**Multiple Linear Regression**")
                        
                            # Regression information
                            fm_mlr_reg_col1, fm_mlr_reg_col2 = st.columns(2)
                            with fm_mlr_reg_col1:
                                st.write("Regression information:")
                                st.table(model_full_results["MLR information"].style.format(precision=user_precision))
                            # Regression statistics
                            with fm_mlr_reg_col2:
                                st.write("Regression statistics:")
                                st.table(model_full_results["MLR statistics"].style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_MLR_regStat")))
                            st.write("")
                            # Coefficients
                            st.write("Coefficients:")
                            st.table(model_full_results["MLR coefficients"].style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_MLR_coef")))
                            st.write("")
                            # ANOVA
                            st.write("ANOVA:")
                            st.table(model_full_results["MLR ANOVA"].style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_MLR_ANOVA")))
                            st.write("")
                            # Heteroskedasticity tests
                            if MLR_intercept == "Yes":
                                st.write("Heteroskedasticity tests:")
                                st.table(model_full_results["MLR hetTest"].style.format(precision=user_precision))
                                if sett_hints:
                                    st.info(str(fc.learning_hints("mod_md_MLR_hetTest")))
                                st.write("")
                            # Variable importance (via permutation)
                            fm_mlr_reg2_col1, fm_mlr_reg2_col2 = st.columns(2)
                            with fm_mlr_reg2_col1: 
                                st.write("Variable importance (via permutation):")
                                mlr_varImp_table = model_full_results["MLR variable importance"]
                                st.table(mlr_varImp_table.style.format(precision=user_precision))
                                st.write("")
                            with fm_mlr_reg2_col2: 
                                st.write("")
                                st.write("")
                                st.write("")
                                mlr_varImp_plot_data = model_full_results["MLR variable importance"]
                                mlr_varImp_plot_data["Variable"] = mlr_varImp_plot_data.index
                                mlr_varImp = alt.Chart(mlr_varImp_plot_data, height = 200).mark_bar().encode(
                                    x = alt.X("mean", title = "variable importance", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Variable", title = None, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), sort = None),
                                    tooltip = ["Variable", "mean"]
                                )
                                st.altair_chart(mlr_varImp, use_container_width = True) 
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_MLR_varImp")))
                            st.write("")                        
                            # Graphical output
                            fm_mlr_figs_col1, fm_mlr_figs_col2 = st.columns(2)
                            with fm_mlr_figs_col1:
                                st.write("Observed vs Fitted:")
                                observed_fitted_data = pd.DataFrame()
                                observed_fitted_data["Observed"] = df[response_var]
                                observed_fitted_data["Fitted"] = model_full_results["MLR fitted"]
                                observed_fitted_data["Index"] = df.index
                                observed_fitted = alt.Chart(observed_fitted_data, height = 200).mark_circle(size=20).encode(
                                    x = alt.X("Fitted", title = "fitted", scale = alt.Scale(domain = [min(observed_fitted_data["Fitted"]), max(observed_fitted_data["Fitted"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Observed", title = "observed", scale = alt.Scale(zero = False),  axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Observed", "Fitted", "Index"]
                                )
                                observed_fitted_plot = observed_fitted + observed_fitted.transform_regression("Fitted", "Observed").mark_line(size = 2, color = "darkred")
                                st.altair_chart(observed_fitted_plot, use_container_width = True)
                            with fm_mlr_figs_col2:
                                st.write("Residuals vs Fitted:")
                                residuals_fitted_data = pd.DataFrame()
                                residuals_fitted_data["Residuals"] = model_full_results["residuals"]["Multiple Linear Regression"]
                                residuals_fitted_data["Fitted"] = model_full_results["MLR fitted"]
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
                            fm_mlr_figs1_col1, fm_mlr_figs1_col2 = st.columns(2)
                            with fm_mlr_figs1_col1:
                                st.write("Normal QQ-plot:")
                                residuals = model_full_results["residuals"]["Multiple Linear Regression"]
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
                            with fm_mlr_figs1_col2:
                                st.write("Scale-Location:")
                                scale_location_data = pd.DataFrame()
                                residuals = model_full_results["residuals"]["Multiple Linear Regression"]
                                scale_location_data["SqrtStandResiduals"] = np.sqrt(abs((residuals - residuals.mean())/residuals.std()))
                                scale_location_data["Fitted"] = model_full_results["MLR fitted"]
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
                            fm_mlr_figs2_col1, fm_mlr_figs2_col2 = st.columns(2)
                            with fm_mlr_figs2_col1:
                                st.write("Residuals vs Leverage:")
                                residuals_leverage_data = pd.DataFrame()
                                residuals = model_full_results["residuals"]["Multiple Linear Regression"]
                                residuals_leverage_data["StandResiduals"] = (residuals - residuals.mean())/residuals.std()
                                residuals_leverage_data["Leverage"] = model_full_results["MLR leverage"]
                                residuals_leverage_data["Index"] = df.index
                                residuals_leverage = alt.Chart(residuals_leverage_data, height = 200).mark_circle(size=20).encode(
                                    x = alt.X("Leverage", title = "leverage", scale = alt.Scale(domain = [min(residuals_leverage_data["Leverage"]), max(residuals_leverage_data["Leverage"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("StandResiduals", title = "stand. residuals", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["StandResiduals","Leverage", "Index"]
                                )
                                residuals_leverage_plot = residuals_leverage + residuals_leverage.transform_loess("Leverage", "StandResiduals", bandwidth = 0.5).mark_line(size = 2, color = "darkred")
                                st.altair_chart(residuals_leverage_plot, use_container_width = True)
                            with fm_mlr_figs2_col2:
                                st.write("Cook's distance:")
                                cooksD_data = pd.DataFrame()
                                cooksD_data["CooksD"] = model_full_results["MLR Cooks distance"]
                                cooksD_data["Index"] = df.index
                                cooksD = alt.Chart(cooksD_data, height = 200).mark_bar(size = 2).encode(
                                    x = alt.X("Index", title = "index", scale = alt.Scale(domain = [-1, max(cooksD_data["Index"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("CooksD", title = "Cook's distance", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["CooksD", "Index"]
                                )
                                st.altair_chart(cooksD, use_container_width = True)  
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_MLR_resVsLev_cooksD")))

                            # Download link for MLR output
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            model_full_results["MLR information"].to_excel(excel_file, sheet_name="regression_information")
                            model_full_results["MLR statistics"].to_excel(excel_file, sheet_name="regression_statistics")
                            model_full_results["MLR coefficients"].to_excel(excel_file, sheet_name="coefficients")
                            model_full_results["MLR ANOVA"].to_excel(excel_file, sheet_name="ANOVA")
                            model_full_results["MLR hetTest"].to_excel(excel_file, sheet_name="heteroskedasticity_tests")
                            mlr_varImp_table.to_excel(excel_file, sheet_name="variable_importance")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name= "MLR full model output__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download Multiple Linear Regression full model output</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("")

                        # GAM specific output
                        if any(a for a in sb_ML_alg if a == "Generalized Additive Models"):
                            st.markdown("**Generalized Additive Models**")

                            fm_gam_reg_col1, fm_gam_reg_col2 = st.columns(2)
                            # Regression information
                            with fm_gam_reg_col1:
                                st.write("Regression information:")
                                st.table(model_full_results["GAM information"].style.format(precision=user_precision))
                            # Regression statistics
                            with fm_gam_reg_col2:
                                st.write("Regression statistics:")
                                st.table(model_full_results["GAM statistics"].style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_GAM_regStat")))
                            st.write("")
                            # Feature significance
                            st.write("Feature significance:")
                            st.table(model_full_results["GAM feature significance"].style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_GAM_featSig")))
                            st.write("")
                            # Variable importance (via permutation)
                            fm_gam_figs1_col1, fm_gam_figs1_col2 = st.columns(2)
                            with fm_gam_figs1_col1:
                                st.write("Variable importance (via permutation):")
                                gam_varImp_table = model_full_results["GAM variable importance"]
                                st.table(gam_varImp_table.style.format(precision=user_precision))
                                st.write("")
                            with fm_gam_figs1_col2:
                                st.write("")
                                st.write("")
                                st.write("")
                                gam_varImp_plot_data = model_full_results["GAM variable importance"]
                                gam_varImp_plot_data["Variable"] = gam_varImp_plot_data.index
                                gam_varImp = alt.Chart(gam_varImp_plot_data, height = 200).mark_bar().encode(
                                    x = alt.X("mean", title = "variable importance", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Variable", title = None, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), sort = None),
                                    tooltip = ["Variable", "mean"]
                                )
                                st.altair_chart(gam_varImp, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_GAM_varImp"))) 
                            st.write("") 
                            # Partial dependence plots
                            st.write("Partial dependence plots:")    
                            fm_gam_figs3_col1, fm_gam_figs3_col2 = st.columns(2)
                            for pd_var in expl_var:
                                pd_data_gam = pd.DataFrame(columns = [pd_var])
                                pd_data_gam[pd_var] = model_full_results["GAM partial dependence"][pd_var]["x_values"]
                                pd_data_gam["Partial dependence"] = model_full_results["GAM partial dependence"][pd_var]["pd_values"]
                                pd_data_gam["Lower 95%"] = model_full_results["GAM partial dependence"][pd_var]["lower_95"]
                                pd_data_gam["Upper 95%"] = model_full_results["GAM partial dependence"][pd_var]["upper_95"]
                                pd_chart_gam = alt.Chart(pd_data_gam, height = 200).mark_line(color = "darkred").encode(
                                    x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Partial dependence", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["GAM partial dependence min/max"]["min"].min(), model_full_results["GAM partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Upper 95%", "Partial dependence", "Lower 95%"] + [pd_var]
                                )
                                pd_data_ticks_gam = pd.DataFrame(columns = [pd_var])
                                pd_data_ticks_gam[pd_var] = df[pd_var]
                                pd_data_ticks_gam["y"] = [model_full_results["GAM partial dependence min/max"]["min"].min()] * df.shape[0]
                                pd_ticks_gam = alt.Chart(pd_data_ticks_gam, height = 200).mark_tick(size = 5, thickness = 1).encode(
                                    x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), scale = alt.Scale(domain = [pd_data_ticks_gam[pd_var].min(), pd_data_ticks_gam[pd_var].max()])),
                                    y = alt.Y("y", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["GAM partial dependence min/max"]["min"].min(), model_full_results["GAM partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = [pd_var]
                                )
                                pd_data_gam_lower = pd.DataFrame(columns = [pd_var])
                                pd_data_gam_lower[pd_var] = model_full_results["GAM partial dependence"][pd_var]["x_values"]
                                pd_data_gam_lower["Lower 95%"] = model_full_results["GAM partial dependence"][pd_var]["lower_95"]
                                pd_chart_gam_lower = alt.Chart(pd_data_gam_lower, height = 200).mark_line(strokeDash=[1,1], color = "darkred").encode(
                                    x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Lower 95%", title = "", scale = alt.Scale(domain = [model_full_results["GAM partial dependence min/max"]["min"].min(), model_full_results["GAM partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Lower 95%"] + [pd_var]
                                )
                                pd_data_gam_upper = pd.DataFrame(columns = [pd_var])
                                pd_data_gam_upper[pd_var] = model_full_results["GAM partial dependence"][pd_var]["x_values"]
                                pd_data_gam_upper["Upper 95%"] = model_full_results["GAM partial dependence"][pd_var]["upper_95"]
                                pd_chart_gam_upper = alt.Chart(pd_data_gam_upper, height = 200).mark_line(strokeDash=[1,1], color = "darkred").encode(
                                    x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Upper 95%", title = "", scale = alt.Scale(domain = [model_full_results["GAM partial dependence min/max"]["min"].min(), model_full_results["GAM partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Upper 95%"] + [pd_var]
                                )
                                if expl_var.index(pd_var)%2 == 0:
                                    with fm_gam_figs3_col1:
                                        st.altair_chart(pd_ticks_gam + pd_chart_gam_lower + pd_chart_gam_upper + pd_chart_gam, use_container_width = True)
                                if expl_var.index(pd_var)%2 == 1:
                                    with fm_gam_figs3_col2:
                                        st.altair_chart(pd_ticks_gam + pd_chart_gam_lower + pd_chart_gam_upper + pd_chart_gam, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_GAM_partDep")))
                            st.write("")         
                            # Further graphical output
                            fm_gam_figs4_col1, fm_gam_figs4_col2 = st.columns(2)
                            with fm_gam_figs4_col1:
                                st.write("Observed vs Fitted:")
                                observed_fitted_data = pd.DataFrame()
                                observed_fitted_data["Observed"] = df[response_var]
                                observed_fitted_data["Fitted"] = model_full_results["GAM fitted"]
                                observed_fitted_data["Index"] = df.index
                                observed_fitted = alt.Chart(observed_fitted_data, height = 200).mark_circle(size=20).encode(
                                    x = alt.X("Fitted", title = "fitted", scale = alt.Scale(domain = [min(observed_fitted_data["Fitted"]), max(observed_fitted_data["Fitted"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Observed", title = "observed", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Observed", "Fitted", "Index"]
                                )
                                observed_fitted_plot = observed_fitted + observed_fitted.transform_regression("Fitted", "Observed").mark_line(size = 2, color = "darkred")
                                st.altair_chart(observed_fitted_plot, use_container_width = True)
                            with fm_gam_figs4_col2:
                                st.write("Residuals vs Fitted:")
                                residuals_fitted_data = pd.DataFrame()
                                residuals_fitted_data["Residuals"] = model_full_results["residuals"]["Generalized Additive Models"]
                                residuals_fitted_data["Fitted"] = model_full_results["GAM fitted"]
                                residuals_fitted_data["Index"] = df.index
                                residuals_fitted = alt.Chart(residuals_fitted_data, height = 200).mark_circle(size=20).encode(
                                    x = alt.X("Fitted", title = "fitted", scale = alt.Scale(domain = [min(residuals_fitted_data["Fitted"]), max(residuals_fitted_data["Fitted"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Residuals", title = "residuals", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Residuals", "Fitted", "Index"]
                                )
                                residuals_fitted_plot = residuals_fitted + residuals_fitted.transform_loess("Fitted", "Residuals", bandwidth = 0.5).mark_line(size = 2, color = "darkred")
                                st.altair_chart(residuals_fitted_plot, use_container_width = True) 
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_GAM_obsResVsFit")))

                            # Download link for GAM output
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            model_full_results["GAM information"].to_excel(excel_file, sheet_name="regression_information")
                            model_full_results["GAM statistics"].to_excel(excel_file, sheet_name="regression_statistics")
                            model_full_results["GAM feature significance"].to_excel(excel_file, sheet_name="feature_significance")
                            gam_varImp_table.to_excel(excel_file, sheet_name="variable_importance")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "GAM full model output__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download Generalized Additive Models full model output</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("")

                        # RF specific output
                        if any(a for a in sb_ML_alg if a == "Random Forest"):
                            st.markdown("**Random Forest**")

                            fm_rf_reg_col1, fm_rf_reg_col2 = st.columns(2)
                            # Regression information
                            with fm_rf_reg_col1:
                                st.write("Regression information:")
                                st.table(model_full_results["RF information"].style.format(precision=user_precision))
                            # Regression statistics
                            with fm_rf_reg_col2:
                                st.write("Regression statistics:")
                                rf_error_est = pd.DataFrame(index = ["MSE", "RMSE", "MAE", "Residual SE"], columns = ["Value"])
                                rf_error_est.loc["MSE"] = model_full_results["model comparison"].loc["MSE"]["Random Forest"]
                                rf_error_est.loc["RMSE"] = model_full_results["model comparison"].loc["RMSE"]["Random Forest"]
                                rf_error_est.loc["MAE"] =  model_full_results["model comparison"].loc["MAE"]["Random Forest"]
                                rf_error_est.loc["Residual SE"] = model_full_results["RF Residual SE"]
                                st.table(rf_error_est.style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_RF_regStat")))
                            st.write("")
                            # Variable importance (via permutation)
                            fm_rf_figs1_col1, fm_rf_figs1_col2 = st.columns(2)
                            with fm_rf_figs1_col1:
                                st.write("Variable importance (via permutation):")
                                rf_varImp_table = model_full_results["RF variable importance"]
                                st.table(rf_varImp_table.style.format(precision=user_precision))
                                st.write("")
                            with fm_rf_figs1_col2:
                                st.write("")
                                st.write("")
                                st.write("")
                                rf_varImp_plot_data = model_full_results["RF variable importance"]
                                rf_varImp_plot_data["Variable"] = rf_varImp_plot_data.index
                                rf_varImp = alt.Chart(rf_varImp_plot_data, height = 200).mark_bar().encode(
                                    x = alt.X("mean", title = "variable importance", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Variable", title = None, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), sort = None),
                                    tooltip = ["Variable", "mean"]
                                )
                                st.altair_chart(rf_varImp, use_container_width = True) 
                            st.write("") 
                            fm_rf_figs2_col1, fm_rf_figs2_col2 = st.columns(2)
                            # Feature importance
                            with fm_rf_figs2_col1:
                                st.write("Feature importance (impurity-based):")
                                rf_featImp_table = model_full_results["RF feature importance"]
                                st.table(rf_featImp_table.style.format(precision=user_precision))
                                st.write("")
                            with fm_rf_figs2_col2:
                                st.write("")
                                st.write("")
                                st.write("")
                                rf_featImp_plot_data = model_full_results["RF feature importance"]
                                rf_featImp_plot_data["Variable"] = rf_featImp_plot_data.index
                                rf_featImp = alt.Chart(rf_featImp_plot_data, height = 200).mark_bar().encode(
                                    x = alt.X("Value", title = "feature importance", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Variable", title = None, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), sort = None),
                                    tooltip = ["Variable", "Value"]
                                )
                                st.altair_chart(rf_featImp, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_RF_varImp")))
                            st.write("") 
                            # Partial dependence plots
                            st.write("Partial dependence plots:")    
                            fm_rf_figs3_col1, fm_rf_figs3_col2 = st.columns(2)
                            for pd_var in expl_var:
                                pd_data_rf = pd.DataFrame(columns = [pd_var])
                                pd_data_rf[pd_var] = model_full_results["RF partial dependence"][pd_var]["values"][0]
                                pd_data_rf["Partial dependence"] = model_full_results["RF partial dependence"][pd_var]["average"][0]


                                pd_chart_rf = alt.Chart(pd_data_rf, height = 200).mark_line(color = "darkred").encode(
                                    x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Partial dependence", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["RF partial dependence min/max"]["min"].min(), model_full_results["RF partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Partial dependence"] + [pd_var]
                                )
                                pd_data_ticks_rf = pd.DataFrame(columns = [pd_var])
                                pd_data_ticks_rf[pd_var] = df[pd_var]
                                pd_data_ticks_rf["y"] = [model_full_results["RF partial dependence min/max"]["min"].min()] * df.shape[0]
                                pd_ticks_rf = alt.Chart(pd_data_ticks_rf, height = 200).mark_tick(size = 5, thickness = 1).encode(
                                    x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), scale = alt.Scale(domain = [pd_data_ticks_rf[pd_var].min(), pd_data_ticks_rf[pd_var].max()])),
                                    y = alt.Y("y", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["RF partial dependence min/max"]["min"].min(), model_full_results["RF partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = [pd_var]
                                )
                                if expl_var.index(pd_var)%2 == 0:
                                    with fm_rf_figs3_col1:
                                        st.altair_chart(pd_ticks_rf + pd_chart_rf, use_container_width = True)
                                if expl_var.index(pd_var)%2 == 1:
                                    with fm_rf_figs3_col2:
                                        st.altair_chart(pd_ticks_rf + pd_chart_rf, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_RF_partDep")))
                            st.write("")         
                            # Further graphical output
                            fm_rf_figs4_col1, fm_rf_figs4_col2 = st.columns(2)
                            with fm_rf_figs4_col1:
                                st.write("Observed vs Fitted:")
                                observed_fitted_data = pd.DataFrame()
                                observed_fitted_data["Observed"] = df[response_var]
                                observed_fitted_data["Fitted"] = model_full_results["RF fitted"]
                                observed_fitted_data["Index"] = df.index
                                observed_fitted = alt.Chart(observed_fitted_data, height = 200).mark_circle(size=20).encode(
                                    x = alt.X("Fitted", title = "fitted", scale = alt.Scale(domain = [min(observed_fitted_data["Fitted"]), max(observed_fitted_data["Fitted"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Observed", title = "observed", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Observed", "Fitted", "Index"]
                                )
                                observed_fitted_plot = observed_fitted + observed_fitted.transform_regression("Fitted", "Observed").mark_line(size = 2, color = "darkred")
                                st.altair_chart(observed_fitted_plot, use_container_width = True)
                            with fm_rf_figs4_col2:
                                st.write("Residuals vs Fitted:")
                                residuals_fitted_data = pd.DataFrame()
                                residuals_fitted_data["Residuals"] = model_full_results["residuals"]["Random Forest"]
                                residuals_fitted_data["Fitted"] = model_full_results["RF fitted"]
                                residuals_fitted_data["Index"] = df.index
                                residuals_fitted = alt.Chart(residuals_fitted_data, height = 200).mark_circle(size=20).encode(
                                    x = alt.X("Fitted", title = "fitted", scale = alt.Scale(domain = [min(residuals_fitted_data["Fitted"]), max(residuals_fitted_data["Fitted"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Residuals", title = "residuals", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Residuals", "Fitted", "Index"]
                                )
                                residuals_fitted_plot = residuals_fitted + residuals_fitted.transform_loess("Fitted", "Residuals", bandwidth = 0.5).mark_line(size = 2, color = "darkred")
                                st.altair_chart(residuals_fitted_plot, use_container_width = True) 
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_RF_obsResVsFit")))

                            # Download link for RF output
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            model_full_results["RF information"].to_excel(excel_file, sheet_name="regression_information")
                            rf_error_est.to_excel(excel_file, sheet_name="regression_statistics")
                            rf_varImp_table.to_excel(excel_file, sheet_name="variable_importance")
                            rf_featImp_table.to_excel(excel_file, sheet_name="feature_importance")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "RF full model output__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download Random Forest full model output</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("")

                        # BRT specific output
                        if any(a for a in sb_ML_alg if a == "Boosted Regression Trees"):
                            st.markdown("**Boosted Regression Trees**")

                            fm_brt_reg_col1, fm_brt_reg_col2 = st.columns(2)
                            # Regression information
                            with fm_brt_reg_col1:
                                st.write("Regression information:")
                                st.table(model_full_results["BRT information"].style.format(precision=user_precision))
                            # Regression statistics
                            with fm_brt_reg_col2:
                                st.write("Regression statistics:")
                                brt_error_est = pd.DataFrame(index = ["MSE", "RMSE", "MAE", "Residual SE"], columns = ["Value"])
                                brt_error_est.loc["MSE"] = model_full_results["model comparison"].loc["MSE"]["Boosted Regression Trees"]
                                brt_error_est.loc["RMSE"] = model_full_results["model comparison"].loc["RMSE"]["Boosted Regression Trees"]
                                brt_error_est.loc["MAE"] =  model_full_results["model comparison"].loc["MAE"]["Boosted Regression Trees"]
                                brt_error_est.loc["Residual SE"] = model_full_results["BRT Residual SE"]
                                st.table(brt_error_est.style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_BRT_regStat")))
                            st.write("")
                            # Training score (MSE vs. number of trees)
                            st.write("Training score:")
                            train_score = pd.DataFrame(index = range(model_full_results["BRT train score"].shape[0]), columns = ["Training MSE"])
                            train_score["Training MSE"] = model_full_results["BRT train score"]
                            train_score["Trees"] = train_score.index+1
                            train_score_plot = alt.Chart(train_score, height = 200).mark_line(color = "darkred").encode(
                                x = alt.X("Trees", title = "trees", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), scale = alt.Scale(domain = [train_score["Trees"].min(), train_score["Trees"].max()])),
                                y = alt.Y("Training MSE", title = "training MSE", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                tooltip = ["Training MSE", "Trees"]
                            )     
                            st.altair_chart(train_score_plot, use_container_width = True)
                            st.write("")
                            # Variable importance (via permutation)
                            fm_brt_figs1_col1, fm_brt_figs1_col2 = st.columns(2)
                            with fm_brt_figs1_col1:
                                st.write("Variable importance (via permutation):")
                                brt_varImp_table = model_full_results["BRT variable importance"]
                                st.table(brt_varImp_table.style.format(precision=user_precision))
                                st.write("")
                            with fm_brt_figs1_col2:
                                st.write("")
                                st.write("")
                                st.write("")
                                brt_varImp_plot_data = model_full_results["BRT variable importance"]
                                brt_varImp_plot_data["Variable"] = brt_varImp_plot_data.index
                                brt_varImp = alt.Chart(brt_varImp_plot_data, height = 200).mark_bar().encode(
                                    x = alt.X("mean", title = "variable importance", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Variable", title = None, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), sort = None),
                                    tooltip = ["Variable", "mean"]
                                )
                                st.altair_chart(brt_varImp, use_container_width = True) 
                            st.write("") 
                            fm_brt_figs2_col1, fm_brt_figs2_col2 = st.columns(2)
                            # Feature importance
                            with fm_brt_figs2_col1:
                                st.write("Feature importance (impurity-based):")
                                brt_featImp_table = model_full_results["BRT feature importance"]
                                st.table(brt_featImp_table.style.format(precision=user_precision))
                                st.write("")
                            with fm_brt_figs2_col2:
                                st.write("")
                                st.write("")
                                st.write("")
                                brt_featImp_plot_data = model_full_results["BRT feature importance"]
                                brt_featImp_plot_data["Variable"] = brt_featImp_plot_data.index
                                brt_featImp = alt.Chart(brt_featImp_plot_data, height = 200).mark_bar().encode(
                                    x = alt.X("Value", title = "feature importance", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Variable", title = None, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), sort = None),
                                    tooltip = ["Variable", "Value"]
                                )
                                st.altair_chart(brt_featImp, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_BRT_varImp")))
                            st.write("") 
                            # Partial dependence plots
                            st.write("Partial dependence plots:")    
                            fm_brt_figs3_col1, fm_brt_figs3_col2 = st.columns(2)
                            for pd_var in expl_var:
                                pd_data_brt = pd.DataFrame(columns = [pd_var])                           
                                pd_data_brt[pd_var] = model_full_results["BRT partial dependence"][pd_var]["values"][0]
                                pd_data_brt["Partial dependence"] = model_full_results["BRT partial dependence"][pd_var]["average"][0]                                                           
                                pd_chart_brt = alt.Chart(pd_data_brt, height = 200).mark_line(color = "darkred").encode(
                                    x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Partial dependence", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["BRT partial dependence min/max"]["min"].min(), model_full_results["BRT partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Partial dependence"] + [pd_var]
                                )
                                pd_data_ticks_brt = pd.DataFrame(columns = [pd_var])
                                pd_data_ticks_brt[pd_var] = df[pd_var]
                                pd_data_ticks_brt["y"] = [model_full_results["BRT partial dependence min/max"]["min"].min()] * df.shape[0]
                                pd_ticks_brt = alt.Chart(pd_data_ticks_brt, height = 200).mark_tick(size = 5, thickness = 1).encode(
                                    x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), scale = alt.Scale(domain = [pd_data_ticks_brt[pd_var].min(), pd_data_ticks_brt[pd_var].max()])),
                                    y = alt.Y("y", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["BRT partial dependence min/max"]["min"].min(), model_full_results["BRT partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = [pd_var]
                                )
                                if expl_var.index(pd_var)%2 == 0:
                                    with fm_brt_figs3_col1:
                                        st.altair_chart(pd_ticks_brt + pd_chart_brt, use_container_width = True)
                                if expl_var.index(pd_var)%2 == 1:
                                    with fm_brt_figs3_col2:
                                        st.altair_chart(pd_ticks_brt + pd_chart_brt, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_BRT_partDep")))
                            st.write("")         
                            # Further graphical output
                            fm_brt_figs4_col1, fm_brt_figs4_col2 = st.columns(2)
                            with fm_brt_figs4_col1:
                                st.write("Observed vs Fitted:")
                                observed_fitted_data = pd.DataFrame()
                                observed_fitted_data["Observed"] = df[response_var]
                                observed_fitted_data["Fitted"] = model_full_results["BRT fitted"]
                                observed_fitted_data["Index"] = df.index
                                observed_fitted = alt.Chart(observed_fitted_data, height = 200).mark_circle(size=20).encode(
                                    x = alt.X("Fitted", title = "fitted", scale = alt.Scale(domain = [min(observed_fitted_data["Fitted"]), max(observed_fitted_data["Fitted"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Observed", title = "observed", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Observed", "Fitted", "Index"]
                                )
                                observed_fitted_plot = observed_fitted + observed_fitted.transform_regression("Fitted", "Observed").mark_line(size = 2, color = "darkred")
                                st.altair_chart(observed_fitted_plot, use_container_width = True)
                            with fm_brt_figs4_col2:
                                st.write("Residuals vs Fitted:")
                                residuals_fitted_data = pd.DataFrame()
                                residuals_fitted_data["Residuals"] = model_full_results["residuals"]["Boosted Regression Trees"]
                                residuals_fitted_data["Fitted"] = model_full_results["BRT fitted"]
                                residuals_fitted_data["Index"] = df.index
                                residuals_fitted = alt.Chart(residuals_fitted_data, height = 200).mark_circle(size=20).encode(
                                    x = alt.X("Fitted", title = "fitted", scale = alt.Scale(domain = [min(residuals_fitted_data["Fitted"]), max(residuals_fitted_data["Fitted"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Residuals", title = "residuals", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Residuals", "Fitted", "Index"]
                                )
                                residuals_fitted_plot = residuals_fitted + residuals_fitted.transform_loess("Fitted", "Residuals", bandwidth = 0.5).mark_line(size = 2, color = "darkred")
                                st.altair_chart(residuals_fitted_plot, use_container_width = True) 
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_BRT_obsResVsFit")))

                            # Download link for BRT output
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            model_full_results["BRT information"].to_excel(excel_file, sheet_name="regression_information")
                            brt_error_est.to_excel(excel_file, sheet_name="regression_statistics")
                            brt_varImp_table.to_excel(excel_file, sheet_name="variable_importance")
                            brt_featImp_table.to_excel(excel_file, sheet_name="feature_importance")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "BRT full model output__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download Boosted Regression Trees full model output</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("")

                        # ANN specific output
                        if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                            st.markdown("**Artificial Neural Networks**")

                            fm_ann_reg_col1, fm_ann_reg_col2 = st.columns(2)
                            # Regression information
                            with fm_ann_reg_col1:
                                st.write("Regression information:")
                                st.table(model_full_results["ANN information"].style.format(precision=user_precision))
                            # Regression statistics
                            with fm_ann_reg_col2:
                                st.write("Regression statistics:")
                                ann_error_est = pd.DataFrame(index = ["MSE", "RMSE", "MAE", "Residual SE", "Best loss"], columns = ["Value"])
                                ann_error_est.loc["MSE"] = model_full_results["model comparison"].loc["MSE"]["Artificial Neural Networks"]
                                ann_error_est.loc["RMSE"] = model_full_results["model comparison"].loc["RMSE"]["Artificial Neural Networks"]
                                ann_error_est.loc["MAE"] =  model_full_results["model comparison"].loc["MAE"]["Artificial Neural Networks"]
                                ann_error_est.loc["Residual SE"] = model_full_results["ANN Residual SE"]
                                if ann_finalPara["weight optimization solver"][0] != "lbfgs":
                                    ann_error_est.loc["Best loss"] = model_full_results["ANN loss"]
                                st.table(ann_error_est.style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_ANN_regStat")))
                            st.write("")
                            # ANN architecture
                            st.write("Artificial Neural Network architecture:")
                            coef_list = full_model_ann_sk.coefs_
                            # weight matrix
                            annviz_weights = st.checkbox("Show weight matrix")
                            if annviz_weights:
                                layer_options = ["Input Layer <-> Hidden Layer 1"]
                                if int(model_full_results["ANN information"].loc['Layers']) == 3:
                                    layer_options += ["Hidden Layer 1 <-> Output Layer"]
                                elif int(model_full_results["ANN information"].loc['Layers']) == 4:
                                    layer_options += ["Hidden Layer 1 <-> Hidden Layer 2"]
                                    layer_options += ["Hidden Layer 2 <-> Output Layer"]
                                elif int(model_full_results["ANN information"].loc['Layers']) == 5:
                                    layer_options += ["Hidden Layer 1 <-> Hidden Layer 2"]
                                    layer_options += ["Hidden Layer 2 <-> Hidden Layer 3"]
                                    layer_options += ["Hidden Layer 3 <-> Output Layer"]
                                wei_matrix = st.selectbox('Weight matrix for following Layer', layer_options)
                                output = [response_var]
                                df_weights = ml.weight_matrix_func(output, expl_var, wei_matrix, coef_list)
                                st.write(df_weights)
                            annviz_output = st.checkbox("Show Artificial Neural Network Visualization")
                            if annviz_output:
                                st.write("Select which neurons of the hidden layer should be visualized:")
                                sli_col1, sli_col2 = st.columns(2)
                                # input layer
                                in_sel_nod = (1,len(expl_var))
                                # hidden layer 1
                                hi1_nod = int(ann_finalPara['hidden layer sizes'][0][0])
                                if hi1_nod >= 10:
                                    hi1_def_max = 10
                                else:
                                    hi1_def_max = hi1_nod
                                hi1_sel_nod = sli_col1.slider('Hidden Layer 1', min_value=1, max_value=hi1_nod, value=[1,hi1_def_max])
                                hi_sel_tup = (hi1_sel_nod[1]-hi1_sel_nod[0]+1,)
                                # hidden layer 2
                                if int(model_full_results["ANN information"].loc['Layers']) >= 4:
                                    hi2_nod = int(ann_finalPara['hidden layer sizes'][0][1])
                                    if hi2_nod >= 10:
                                        hi2_def_max = 10
                                    else:
                                        hi2_def_max = hi2_nod
                                    hi2_sel_nod = sli_col2.slider('Hidden Layer 2', min_value=1, max_value=hi2_nod, value=[1,hi2_def_max])
                                    hi_sel_tup += (hi2_sel_nod[1]-hi2_sel_nod[0]+1,)
                                # hidden layer 3
                                if int(model_full_results["ANN information"].loc['Layers']) >= 5:
                                    hi3_nod = int(ann_finalPara['hidden layer sizes'][0][2])
                                    if hi3_nod >= 10:
                                        hi3_def_max = 10
                                    else:
                                        hi3_def_max = hi3_nod
                                    hi3_sel_nod = sli_col1.slider('Hidden Layer 3', min_value=1, max_value=hi3_nod, value=[1,hi3_def_max])
                                    hi_sel_tup += (hi3_sel_nod[1]-hi3_sel_nod[0]+1,)
                                
                                # ANN Visualization
                                st.write("")
                                st.warning("Very large artificial neural networks cannot be visualized clearly. Recommendation: display max. 20 neurons in one layer.")
                                numb_output = len([response_var])
                                network_structure = np.hstack(([in_sel_nod[1]-in_sel_nod[0]+1], np.asarray(hi_sel_tup), [numb_output]))
                                
                                # seperate weight matrix
                                if int(model_full_results["ANN information"].loc['Layers']) == 3:
                                    in_hi_wei = coef_list[0]
                                    hi_out_wei = coef_list[1]
                                elif int(model_full_results["ANN information"].loc['Layers']) == 4:
                                    in_hi_wei = coef_list[0]
                                    hi1_hi2_wei = coef_list[1]
                                    hi_out_wei = coef_list[2]
                                elif int(model_full_results["ANN information"].loc['Layers']) == 5:
                                    in_hi_wei = coef_list[0]
                                    hi1_hi2_wei = coef_list[1]
                                    hi2_hi3_wei = coef_list[2]
                                    hi_out_wei = coef_list[3]

                                # weights for selected nodes
                                sel_in_hi_wei = in_hi_wei[in_sel_nod[0]-1:in_sel_nod[1], hi1_sel_nod[0]-1:hi1_sel_nod[1]]
                                sel_coef_list = [sel_in_hi_wei]
                                # 1 hidden layer
                                if int(model_full_results["ANN information"].loc['Layers']) == 3:
                                    sel_hi_out_wei = hi_out_wei[hi1_sel_nod[0]-1:hi1_sel_nod[1], 0:numb_output]
                                    sel_coef_list += [sel_hi_out_wei]
                                # 2 hidden layer
                                elif int(model_full_results["ANN information"].loc['Layers']) == 4:
                                    sel_hi1_hi2_wei = hi1_hi2_wei[hi1_sel_nod[0]-1:hi1_sel_nod[1], hi2_sel_nod[0]-1:hi2_sel_nod[1]]
                                    sel_hi_out_wei = hi_out_wei[hi2_sel_nod[0]-1:hi2_sel_nod[1], 0:numb_output]
                                    sel_coef_list += [sel_hi1_hi2_wei]
                                    sel_coef_list += [sel_hi_out_wei]
                                # 3 hidden layer
                                elif int(model_full_results["ANN information"].loc['Layers']) == 5:
                                    sel_hi1_hi2_wei = hi1_hi2_wei[hi1_sel_nod[0]-1:hi1_sel_nod[1], hi2_sel_nod[0]-1:hi2_sel_nod[1]]
                                    sel_hi2_hi3_wei = hi2_hi3_wei[hi2_sel_nod[0]-1:hi2_sel_nod[1], hi3_sel_nod[0]-1:hi3_sel_nod[1]]
                                    sel_hi_out_wei = hi_out_wei[hi3_sel_nod[0]-1:hi3_sel_nod[1], 0:numb_output]
                                    sel_coef_list += [sel_hi1_hi2_wei]
                                    sel_coef_list += [sel_hi2_hi3_wei]
                                    sel_coef_list += [sel_hi_out_wei]                                
                                
                                network=VisNN.DrawNN(network_structure, sel_coef_list)
                                st.set_option('deprecation.showPyplotGlobalUse', False)
                                st.write("")
                                st.pyplot(network.draw())
                            st.write("")
                            # Loss curve (loss vs. number of iterations (epochs))
                            if ann_finalPara["weight optimization solver"][0] != "lbfgs":
                                st.write("Loss curve:")
                                loss_curve = pd.DataFrame(index = range(len(model_full_results["ANN loss curve"])), columns = ["Loss"])
                                loss_curve["Loss"] = model_full_results["ANN loss curve"]
                                loss_curve["Iterations"] = loss_curve.index+1
                                loss_curve_plot = alt.Chart(loss_curve, height = 200).mark_line(color = "darkred").encode(
                                    x = alt.X("Iterations", title = "iterations (epochs)", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), scale = alt.Scale(domain = [loss_curve["Iterations"].min(), loss_curve["Iterations"].max()])),
                                    y = alt.Y("Loss", title = "loss", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Loss", "Iterations"]
                                )     
                                st.altair_chart(loss_curve_plot, use_container_width = True)    
                            st.write("") 
                            fm_ann_figs1_col1, fm_ann_figs1_col2 = st.columns(2)
                            # Variable importance (via permutation)
                            with fm_ann_figs1_col1:
                                st.write("Variable importance (via permutation):")
                                ann_varImp_table = model_full_results["ANN variable importance"]
                                st.table(ann_varImp_table.style.format(precision=user_precision))
                                st.write("")
                            with fm_ann_figs1_col2:
                                st.write("")
                                st.write("")
                                st.write("")
                                ann_varImp_plot_data = model_full_results["ANN variable importance"]
                                ann_varImp_plot_data["Variable"] = ann_varImp_plot_data.index
                                ann_varImp = alt.Chart(ann_varImp_plot_data, height = 200).mark_bar().encode(
                                    x = alt.X("mean", title = "variable importance", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Variable", title = None, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), sort = None),
                                    tooltip = ["Variable", "mean"]
                                )
                                st.altair_chart(ann_varImp, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_ANN_varImp")))
                            st.write("") 
                            # Partial dependence plots
                            st.write("Partial dependence plots:")    
                            fm_ann_figs2_col1, fm_ann_figs2_col2 = st.columns(2)
                            for pd_var in expl_var:
                                pd_data_ann = pd.DataFrame(columns = [pd_var])
                                pd_data_ann[pd_var] = (model_full_results["ANN partial dependence"][pd_var]["values"][0]*(df[pd_var].std()))+df[pd_var].mean()
                                pd_data_ann["Partial dependence"] = model_full_results["ANN partial dependence"][pd_var]["average"][0]
                                pd_chart_ann = alt.Chart(pd_data_ann, height = 200).mark_line(color = "darkred").encode(
                                x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                y = alt.Y("Partial dependence", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["ANN partial dependence min/max"]["min"].min(), model_full_results["ANN partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                tooltip = ["Partial dependence"] + [pd_var]
                                )
                                pd_data_ticks_ann = pd.DataFrame(columns = [pd_var])
                                pd_data_ticks_ann[pd_var] = df[pd_var]
                                pd_data_ticks_ann["y"] = [model_full_results["ANN partial dependence min/max"]["min"].min()] * df.shape[0]
                                pd_ticks_ann = alt.Chart(pd_data_ticks_ann, height = 200).mark_tick(size = 5, thickness = 1).encode(
                                    x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), scale = alt.Scale(domain = [pd_data_ticks_ann[pd_var].min(), pd_data_ticks_ann[pd_var].max()])),
                                    y = alt.Y("y", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["ANN partial dependence min/max"]["min"].min(), model_full_results["ANN partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = [pd_var]
                                )
                                if expl_var.index(pd_var)%2 == 0:
                                    with fm_ann_figs2_col1:
                                        st.altair_chart(pd_ticks_ann + pd_chart_ann, use_container_width = True)
                                if expl_var.index(pd_var)%2 == 1:
                                    with fm_ann_figs2_col2:
                                        st.altair_chart(pd_ticks_ann + pd_chart_ann, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_ANN_partDep")))
                            st.write("") 
                            # Further graphical output
                            fm_ann_figs3_col1, fm_ann_figs3_col2 = st.columns(2)
                            with fm_ann_figs3_col1:
                                st.write("Observed vs Fitted:")
                                observed_fitted_data = pd.DataFrame()
                                observed_fitted_data["Observed"] = df[response_var]
                                observed_fitted_data["Fitted"] = model_full_results["ANN fitted"]
                                observed_fitted_data["Index"] = df.index
                                observed_fitted = alt.Chart(observed_fitted_data, height = 200).mark_circle(size=20).encode(
                                    x = alt.X("Fitted", title = "fitted", scale = alt.Scale(domain = [min(observed_fitted_data["Fitted"]), max(observed_fitted_data["Fitted"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Observed", title = "observed", scale = alt.Scale(zero = False),  axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Observed", "Fitted", "Index"]
                                )
                                observed_fitted_plot = observed_fitted + observed_fitted.transform_regression("Fitted", "Observed").mark_line(size = 2, color = "darkred")
                                st.altair_chart(observed_fitted_plot, use_container_width = True)
                            with fm_ann_figs3_col2:
                                st.write("Residuals vs Fitted:")
                                residuals_fitted_data = pd.DataFrame()
                                residuals_fitted_data["Residuals"] = model_full_results["residuals"]["Artificial Neural Networks"]
                                residuals_fitted_data["Fitted"] = model_full_results["ANN fitted"]
                                residuals_fitted_data["Index"] = df.index
                                residuals_fitted = alt.Chart(residuals_fitted_data, height = 200).mark_circle(size=20).encode(
                                    x = alt.X("Fitted", title = "fitted", scale = alt.Scale(domain = [min(residuals_fitted_data["Fitted"]), max(residuals_fitted_data["Fitted"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Residuals", title = "residuals", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Residuals", "Fitted", "Index"]
                                )
                                residuals_fitted_plot = residuals_fitted + residuals_fitted.transform_loess("Fitted", "Residuals", bandwidth = 0.5).mark_line(size = 2, color = "darkred")
                                st.altair_chart(residuals_fitted_plot, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_ANN_obsResVsFit")))  
                            
                            # Download link for ANN output
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            model_full_results["ANN information"].to_excel(excel_file, sheet_name="regression_information")
                            ann_error_est.to_excel(excel_file, sheet_name="regression_statistics")
                            ann_varImp_table.to_excel(excel_file, sheet_name="variable_importance")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "ANN full model output__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download Artificial Neural Networks full model output</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("") 

                        # Performance metrics across all models
                        st.markdown("**Model comparison**")
                        st.write("Performance metrics:")
                        model_comp_sort_enable = (model_full_results["model comparison"]).transpose()
                        
                        st.write(model_comp_sort_enable.style.format(precision=user_precision))
                        if len(sb_ML_alg) > 1:
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_modCompPerf")))  
                        st.write("")
                        model_full_res = pd.DataFrame(index = ["min", "25%-Q", "median", "75%-Q", "max"], columns = sb_ML_alg)
                        for m in sb_ML_alg: 
                            model_full_res.loc["min"][m] = model_full_results["residuals"][m].min()
                            model_full_res.loc["25%-Q"][m] = model_full_results["residuals"][m].quantile(q = 0.25)
                            model_full_res.loc["median"][m] = model_full_results["residuals"][m].quantile(q = 0.5)
                            model_full_res.loc["75%-Q"][m] = model_full_results["residuals"][m].quantile(q = 0.75)
                            model_full_res.loc["max"][m] = model_full_results["residuals"][m].max()
                        st.write("Residuals distribution:")
                        st.write((model_full_res).transpose().style.format(precision=user_precision))
                        if len(sb_ML_alg) > 1:
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_modCompRes")))
                        st.write("")

                        # Download link for model comparison output
                        output = BytesIO()
                        excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                        model_comp_sort_enable.to_excel(excel_file, sheet_name="performance_metrics")
                        model_full_res.transpose().to_excel(excel_file, sheet_name="residuals_distribution")
                        excel_file.close()
                        excel_file = output.getvalue()
                        b64 = base64.b64encode(excel_file)
                        dl_file_name = "Model comparison full model output__" + df_name + ".xlsx"
                        st.markdown(
                            f"""
                        <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download model comparison output</a>
                        """,
                        unsafe_allow_html=True)
                        st.write("")
                
                    #-------------------------------------------------------------

                    # Binary response variable
                    if response_var_type == "binary":
                        
                        # MLR specific output
                        if any(a for a in sb_ML_alg if a == "Multiple Linear Regression"):
                            st.markdown("**Multiple Linear Regression**")
                            # Regression information
                            fm_mlr_reg_col1, fm_mlr_reg_col2 = st.columns(2)
                            with fm_mlr_reg_col1:
                                st.write("Regression information:")
                                st.table(model_full_results["MLR information"].style.format(precision=user_precision))
                            # Regression statistics
                            with fm_mlr_reg_col2:
                                st.write("Regression statistics:")
                                st.table(model_full_results["MLR statistics"].style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_MLR_regStat")))
                            st.write("")
                            # Coefficients
                            st.write("Coefficients:")
                            st.table(model_full_results["MLR coefficients"].style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_MLR_coef")))
                            st.write("")
                            # ANOVA
                            st.write("ANOVA:")
                            st.table(model_full_results["MLR ANOVA"].style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_MLR_ANOVA")))
                            st.write("")
                            # Heteroskedasticity tests
                            if MLR_intercept == "Yes":
                                st.write("Heteroskedasticity tests:")
                                st.table(model_full_results["MLR hetTest"].style.format(precision=user_precision))
                                if sett_hints:
                                    st.info(str(fc.learning_hints("mod_md_MLR_hetTest")))
                                st.write("")
                            # Variable importance (via permutation)
                            fm_mlr_reg2_col1, fm_mlr_reg2_col2 = st.columns(2)
                            with fm_mlr_reg2_col1: 
                                st.write("Variable importance (via permutation):")
                                mlr_varImp_table = model_full_results["MLR variable importance"]
                                st.table(mlr_varImp_table.style.format(precision=user_precision))
                                st.write("")
                            with fm_mlr_reg2_col2: 
                                st.write("")
                                st.write("")
                                st.write("")
                                mlr_varImp_plot_data = model_full_results["MLR variable importance"]
                                mlr_varImp_plot_data["Variable"] = mlr_varImp_plot_data.index
                                mlr_varImp = alt.Chart(mlr_varImp_plot_data, height = 200).mark_bar().encode(
                                    x = alt.X("mean", title = "variable importance", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Variable", title = None, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), sort = None),
                                    tooltip = ["Variable", "mean"]
                                )
                                st.altair_chart(mlr_varImp, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_MLR_varImp")))
                            st.write("")                                 
                            # Graphical output
                            fm_mlr_figs_col1, fm_mlr_figs_col2 = st.columns(2)
                            with fm_mlr_figs_col1:
                                st.write("Observed vs Fitted:")
                                observed_fitted_data = pd.DataFrame()
                                observed_fitted_data["Observed"] = df[response_var]
                                observed_fitted_data["Fitted"] = model_full_results["MLR fitted"]
                                observed_fitted_data["Index"] = df.index
                                observed_fitted = alt.Chart(observed_fitted_data, height = 200).mark_circle(size=20).encode(
                                    x = alt.X("Fitted", title = "fitted", scale = alt.Scale(domain = [min(observed_fitted_data["Fitted"]), max(observed_fitted_data["Fitted"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Observed", title = "observed", scale = alt.Scale(zero = False),  axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Observed", "Fitted", "Index"]
                                )
                                observed_fitted_plot = observed_fitted + observed_fitted.transform_regression("Fitted", "Observed").mark_line(size = 2, color = "darkred")
                                st.altair_chart(observed_fitted_plot, use_container_width = True)
                            with fm_mlr_figs_col2:
                                st.write("Residuals vs Fitted:")
                                residuals_fitted_data = pd.DataFrame()
                                residuals_fitted_data["Residuals"] = model_full_results["residuals"]["Multiple Linear Regression"]
                                residuals_fitted_data["Fitted"] = model_full_results["MLR fitted"]
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
                            fm_mlr_figs1_col1, fm_mlr_figs1_col2 = st.columns(2)
                            with fm_mlr_figs1_col1:
                                st.write("Normal QQ-plot:")
                                residuals = model_full_results["residuals"]["Multiple Linear Regression"]
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
                            with fm_mlr_figs1_col2:
                                st.write("Scale-Location:")
                                scale_location_data = pd.DataFrame()
                                residuals = model_full_results["residuals"]["Multiple Linear Regression"]
                                scale_location_data["SqrtStandResiduals"] = np.sqrt(abs((residuals - residuals.mean())/residuals.std()))
                                scale_location_data["Fitted"] = model_full_results["MLR fitted"]
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
                            fm_mlr_figs2_col1, fm_mlr_figs2_col2 = st.columns(2)
                            with fm_mlr_figs2_col1:
                                st.write("Residuals vs Leverage:")
                                residuals_leverage_data = pd.DataFrame()
                                residuals = model_full_results["residuals"]["Multiple Linear Regression"]
                                residuals_leverage_data["StandResiduals"] = (residuals - residuals.mean())/residuals.std()
                                residuals_leverage_data["Leverage"] = model_full_results["MLR leverage"]
                                residuals_leverage_data["Index"] = df.index
                                residuals_leverage = alt.Chart(residuals_leverage_data, height = 200).mark_circle(size=20).encode(
                                    x = alt.X("Leverage", title = "leverage", scale = alt.Scale(domain = [min(residuals_leverage_data["Leverage"]), max(residuals_leverage_data["Leverage"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("StandResiduals", title = "stand. residuals", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["StandResiduals","Leverage", "Index"]
                                )
                                residuals_leverage_plot = residuals_leverage + residuals_leverage.transform_loess("Leverage", "StandResiduals", bandwidth = 0.5).mark_line(size = 2, color = "darkred")
                                st.altair_chart(residuals_leverage_plot, use_container_width = True)
                            with fm_mlr_figs2_col2:
                                st.write("Cook's distance:")
                                cooksD_data = pd.DataFrame()
                                cooksD_data["CooksD"] = model_full_results["MLR Cooks distance"]
                                cooksD_data["Index"] = df.index
                                cooksD = alt.Chart(cooksD_data, height = 200).mark_bar(size = 2).encode(
                                    x = alt.X("Index", title = "index", scale = alt.Scale(domain = [-1, max(cooksD_data["Index"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("CooksD", title = "Cook's distance", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["CooksD", "Index"]
                                )
                                st.altair_chart(cooksD, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_MLR_resVsLev_cooksD")))
                            
                            # Download link for MLR output
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            model_full_results["MLR information"].to_excel(excel_file, sheet_name="regression_information")
                            model_full_results["MLR statistics"].to_excel(excel_file, sheet_name="regression_statistics")
                            model_full_results["MLR coefficients"].to_excel(excel_file, sheet_name="coefficients")
                            model_full_results["MLR ANOVA"].to_excel(excel_file, sheet_name="ANOVA")
                            model_full_results["MLR hetTest"].to_excel(excel_file, sheet_name="heteroskedasticity_tests")
                            mlr_varImp_table.to_excel(excel_file, sheet_name="variable_importance")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "MLR full model output__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download Multiple Linear Regression full model output</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("") 
                        
                        # LR specific output
                        if any(a for a in sb_ML_alg if a == "Logistic Regression"):
                            st.markdown("**Logistic Regression**")
                            # Regression information
                            fm_lr_reg_col1, fm_lr_reg_col2 = st.columns(2)
                            with fm_lr_reg_col1:
                                st.write("Regression information:")
                                st.table(model_full_results["LR information"].style.format(precision=user_precision))
                            # Regression statistics
                            with fm_lr_reg_col2:
                                st.write("Regression statistics:")
                                st.table(model_full_results["LR statistics"].style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_LR_regStat")))
                            st.write("")  
                            # Coefficients
                            st.write("Coefficients:")
                            st.table(model_full_results["LR coefficients"].style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_LR_coef")))
                            st.write("")
                            # Variable importance (via permutation)
                            fm_lr_fig1_col1, fm_lr_fig1_col2 = st.columns(2)
                            with fm_lr_fig1_col1: 
                                st.write("Variable importance (via permutation):")
                                lr_varImp_table = model_full_results["LR variable importance"]
                                st.table(lr_varImp_table.style.format(precision=user_precision))
                            with fm_lr_fig1_col2: 
                                st.write("")
                                st.write("")
                                st.write("")
                                lr_varImp_plot_data = model_full_results["LR variable importance"]
                                lr_varImp_plot_data["Variable"] = lr_varImp_plot_data.index
                                lr_varImp = alt.Chart(lr_varImp_plot_data, height = 200).mark_bar().encode(
                                    x = alt.X("mean", title = "variable importance", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Variable", title = None, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), sort = None),
                                    tooltip = ["Variable", "mean"]
                                )
                                st.altair_chart(lr_varImp, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_LR_varImp"))) 
                            st.write("") 
                            fm_lr_fig_col1, fm_lr_fig_col2 = st.columns(2)
                            # Observed vs. Probability of Occurrence 
                            with fm_lr_fig_col1:
                                st.write("Observed vs. Probability of Occurrence:")
                                prob_data = pd.DataFrame(model_full_results["LR fitted"])
                                prob_data["Observed"] = df[response_var]
                                prob_data["ProbabilityOfOccurrence"] = prob_data[1]
                                prob_data["Threshold"] = model_full_results["model comparison thres"].loc["threshold"]["Logistic Regression"]
                                prob_data_plot = alt.Chart(prob_data, height = 200).mark_tick(size = 5, thickness = 1).encode(
                                    x = alt.X("ProbabilityOfOccurrence", title = "probability of occurrence", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Observed", title = "observed", scale = alt.Scale(domain = [min(prob_data["Observed"]), max(prob_data["Observed"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Observed", "ProbabilityOfOccurrence", "Threshold"]
                                )
                                thres = alt.Chart(prob_data, height = 200).mark_rule(size = 2, color = "darkred").encode(x = "Threshold", tooltip = ["Threshold"]) 
                                prob_plot = prob_data_plot + thres
                                st.altair_chart(prob_plot, use_container_width = True)
                            # ROC curve 
                            with fm_lr_fig_col2:
                                st.write("ROC curve:")
                                AUC_ROC_data = pd.DataFrame()
                                AUC_ROC_data["FPR"] = model_full_results["LR ROC curve"][0]
                                AUC_ROC_data["TPR"] = model_full_results["LR ROC curve"][1]
                                AUC_ROC_data["AUC ROC"] = model_full_results["model comparison thresInd"].loc["AUC ROC"]["Logistic Regression"]
                                AUC_ROC_data["Threshold"] = model_full_results["model comparison thres"].loc["threshold"]["Logistic Regression"]
                                AUC_ROC_plot= alt.Chart(AUC_ROC_data, height = 200).mark_line().encode(
                                    x = alt.X("FPR", title = "1 - specificity (FPR)", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("TPR", title = "sensitivity (TPR)", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["TPR", "FPR", "AUC ROC"]
                                )
                                line = alt.Chart(
                                    pd.DataFrame({"FPR": [min(AUC_ROC_data["FPR"]), max(AUC_ROC_data["FPR"])], "TPR": [min(AUC_ROC_data["FPR"]), max(AUC_ROC_data["FPR"])]})).mark_line(size = 2, color = "darkred").encode(
                                            alt.X("FPR"),
                                            alt.Y("TPR"),
                                )
                                st.altair_chart(AUC_ROC_plot + line, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_LR_thresAUC"))) 
                            st.write("") 
                            # Partial probabilities
                            st.write("Partial probability plots:")    
                            fm_lr_figs2_col1, fm_lr_figs2_col2 = st.columns(2)
                            for pp_var in expl_var:
                                pp_data = pd.DataFrame(columns = [pp_var])
                                pp_data[pp_var] = model_full_results["LR partial probabilities"][pp_var][pp_var]
                                pp_data["ProbabilityOfOccurrence"] = model_full_results["LR partial probabilities"][pp_var]["prediction"]
                                pp_data["Observed"] = df[response_var]
                                pp_chart = alt.Chart(pp_data, height = 200).mark_line(color = "darkred").encode(
                                x = alt.X(pp_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                y = alt.Y("ProbabilityOfOccurrence", title = "probability of occurrence", scale = alt.Scale(domain = [0, 1]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                tooltip = ["ProbabilityOfOccurrence"] + [pp_var]
                                )
                                obs_data_plot = alt.Chart(pp_data, height = 200).mark_tick(size = 5, thickness = 1).encode(
                                    x = alt.X(pp_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Observed", title = "probability of occurrence", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Observed", "ProbabilityOfOccurrence"] + [pp_var]
                                )
                                if expl_var.index(pp_var)%2 == 0:
                                    with fm_lr_figs2_col1:
                                        st.altair_chart(pp_chart + obs_data_plot, use_container_width = True)
                                if expl_var.index(pp_var)%2 == 1:
                                    with fm_lr_figs2_col2:
                                        st.altair_chart(pp_chart + obs_data_plot, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_LR_partProb"))) 

                            # Download link for LR output
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            model_full_results["LR information"].to_excel(excel_file, sheet_name="regression_information")
                            model_full_results["LR statistics"].to_excel(excel_file, sheet_name="regression_statistics")
                            model_full_results["LR coefficients"].to_excel(excel_file, sheet_name="coefficients")
                            lr_varImp_table.to_excel(excel_file, sheet_name="variable_importance")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "LR full model output__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download Logistic Regression full model output</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("")

                        # GAM specific output
                        if any(a for a in sb_ML_alg if a == "Generalized Additive Models"):
                            st.markdown("**Generalized Additive Models**")

                            fm_gam_reg_col1, fm_gam_reg_col2 = st.columns(2)
                            # Regression information
                            with fm_gam_reg_col1:
                                st.write("Regression information:")
                                st.table(model_full_results["GAM information"].style.format(precision=user_precision))
                            # Regression statistics
                            with fm_gam_reg_col2:
                                st.write("Regression statistics:")
                                st.table(model_full_results["GAM statistics"].style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_GAM_regStat_bin")))
                            st.write("")
                            # Feature significance
                            st.write("Feature significance:")
                            st.table(model_full_results["GAM feature significance"].style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_GAM_featSig_bin")))
                            st.write("")
                            # Variable importance (via permutation)
                            fm_gam_figs1_col1, fm_gam_figs1_col2 = st.columns(2)
                            with fm_gam_figs1_col1:
                                st.write("Variable importance (via permutation):")
                                gam_varImp_table = model_full_results["GAM variable importance"]
                                st.table(gam_varImp_table.style.format(precision=user_precision))
                                st.write("")
                            with fm_gam_figs1_col2:
                                st.write("")
                                st.write("")
                                st.write("")
                                gam_varImp_plot_data = model_full_results["GAM variable importance"]
                                gam_varImp_plot_data["Variable"] = gam_varImp_plot_data.index
                                gam_varImp = alt.Chart(gam_varImp_plot_data, height = 200).mark_bar().encode(
                                    x = alt.X("mean", title = "variable importance", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Variable", title = None, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), sort = None),
                                    tooltip = ["Variable", "mean"]
                                )
                                st.altair_chart(gam_varImp, use_container_width = True) 
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_GAM_varImp_bin"))) 
                            st.write("")
                            # Observed vs. Probability of Occurrence
                            fm_gam_figs5_col1, fm_gam_figs5_col2 = st.columns(2) 
                            with fm_gam_figs5_col1:
                                st.write("Observed vs. Probability of Occurrence:")
                                prob_data = pd.DataFrame(model_full_results["GAM fitted"])
                                prob_data["Observed"] = df[response_var]
                                prob_data["ProbabilityOfOccurrence"] = prob_data[0]
                                prob_data["Threshold"] = model_full_results["model comparison thres"].loc["threshold"]["Generalized Additive Models"]
                                prob_data_plot = alt.Chart(prob_data, height = 200).mark_tick(size = 5, thickness = 1).encode(
                                    x = alt.X("ProbabilityOfOccurrence", title = "probability of occurrence", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Observed", title = "observed", scale = alt.Scale(domain = [min(prob_data["Observed"]), max(prob_data["Observed"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Observed", "ProbabilityOfOccurrence", "Threshold"]
                                )
                                thres = alt.Chart(prob_data, height = 200).mark_rule(size = 1.5, color = "darkred").encode(x = "Threshold", tooltip = ["Threshold"]) 
                                prob_plot = prob_data_plot + thres
                                st.altair_chart(prob_plot, use_container_width = True)
                            # ROC curve 
                            with fm_gam_figs5_col2:
                                st.write("ROC curve:")
                                AUC_ROC_data = pd.DataFrame()
                                AUC_ROC_data["FPR"] = model_full_results["GAM ROC curve"][0]
                                AUC_ROC_data["TPR"] = model_full_results["GAM ROC curve"][1]
                                AUC_ROC_data["AUC ROC"] = model_full_results["model comparison thresInd"].loc["AUC ROC"]["Generalized Additive Models"]
                                AUC_ROC_data["Threshold"] = model_full_results["model comparison thres"].loc["threshold"]["Generalized Additive Models"]
                                AUC_ROC_plot= alt.Chart(AUC_ROC_data, height = 200).mark_line().encode(
                                    x = alt.X("FPR", title = "1 - specificity (FPR)", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("TPR", title = "sensitivity (TPR)", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["TPR", "FPR", "AUC ROC"]
                                )
                                line = alt.Chart(
                                    pd.DataFrame({"FPR": [min(AUC_ROC_data["FPR"]), max(AUC_ROC_data["FPR"])], "TPR": [min(AUC_ROC_data["FPR"]), max(AUC_ROC_data["FPR"])]})).mark_line(size = 2, color = "darkred").encode(
                                            alt.X("FPR"),
                                            alt.Y("TPR"),
                                )
                                st.altair_chart(AUC_ROC_plot + line, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_GAM_thresAUC")))
                            st.write("")  
                            # Partial dependence plots
                            st.write("Partial dependence plots:")    
                            fm_gam_figs3_col1, fm_gam_figs3_col2 = st.columns(2)
                            for pd_var in expl_var:
                                pd_data_gam = pd.DataFrame(columns = [pd_var])
                                pd_data_gam[pd_var] = model_full_results["GAM partial dependence"][pd_var]["x_values"]
                                pd_data_gam["Partial dependence"] = model_full_results["GAM partial dependence"][pd_var]["pd_values"]
                                pd_data_gam["Lower 95%"] = model_full_results["GAM partial dependence"][pd_var]["lower_95"]
                                pd_data_gam["Upper 95%"] = model_full_results["GAM partial dependence"][pd_var]["upper_95"]
                                pd_chart_gam = alt.Chart(pd_data_gam, height = 200).mark_line(color = "darkred").encode(
                                    x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Partial dependence", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["GAM partial dependence min/max"]["min"].min(), model_full_results["GAM partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Upper 95%", "Partial dependence", "Lower 95%"] + [pd_var]
                                )
                                pd_data_ticks_gam = pd.DataFrame(columns = [pd_var])
                                pd_data_ticks_gam[pd_var] = df[pd_var]
                                pd_data_ticks_gam["y"] = [model_full_results["GAM partial dependence min/max"]["min"].min()] * df.shape[0]
                                pd_ticks_gam = alt.Chart(pd_data_ticks_gam, height = 200).mark_tick(size = 5, thickness = 1).encode(
                                    x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), scale = alt.Scale(domain = [pd_data_ticks_gam[pd_var].min(), pd_data_ticks_gam[pd_var].max()])),
                                    y = alt.Y("y", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["GAM partial dependence min/max"]["min"].min(), model_full_results["GAM partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = [pd_var]
                                )
                                pd_data_gam_lower = pd.DataFrame(columns = [pd_var])
                                pd_data_gam_lower[pd_var] = model_full_results["GAM partial dependence"][pd_var]["x_values"]
                                pd_data_gam_lower["Lower 95%"] = model_full_results["GAM partial dependence"][pd_var]["lower_95"]
                                pd_chart_gam_lower = alt.Chart(pd_data_gam_lower, height = 200).mark_line(strokeDash=[1,1], color = "darkred").encode(
                                    x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Lower 95%", title = "", scale = alt.Scale(domain = [model_full_results["GAM partial dependence min/max"]["min"].min(), model_full_results["GAM partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Lower 95%"] + [pd_var]
                                )
                                pd_data_gam_upper = pd.DataFrame(columns = [pd_var])
                                pd_data_gam_upper[pd_var] = model_full_results["GAM partial dependence"][pd_var]["x_values"]
                                pd_data_gam_upper["Upper 95%"] = model_full_results["GAM partial dependence"][pd_var]["upper_95"]
                                pd_chart_gam_upper = alt.Chart(pd_data_gam_upper, height = 200).mark_line(strokeDash=[1,1], color = "darkred").encode(
                                    x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Upper 95%", title = "", scale = alt.Scale(domain = [model_full_results["GAM partial dependence min/max"]["min"].min(), model_full_results["GAM partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Upper 95%"] + [pd_var]
                                )
                                if expl_var.index(pd_var)%2 == 0:
                                    with fm_gam_figs3_col1:
                                        st.altair_chart(pd_ticks_gam + pd_chart_gam_lower + pd_chart_gam_upper + pd_chart_gam, use_container_width = True)
                                if expl_var.index(pd_var)%2 == 1:
                                    with fm_gam_figs3_col2:
                                        st.altair_chart(pd_ticks_gam + pd_chart_gam_lower + pd_chart_gam_upper + pd_chart_gam, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_BRT_partDep_bin")))
                            st.write("")                                 

                            # Download link for GAM output
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            model_full_results["GAM information"].to_excel(excel_file, sheet_name="regression_information")
                            model_full_results["GAM statistics"].to_excel(excel_file, sheet_name="regression_statistics")
                            model_full_results["GAM feature significance"].to_excel(excel_file, sheet_name="feature_significance")
                            gam_varImp_table.to_excel(excel_file, sheet_name="variable_importance")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "GAM full model output__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download Generalized Additive Models full model output</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("")

                        # RF specific output
                        if any(a for a in sb_ML_alg if a == "Random Forest"):
                            st.markdown("**Random Forest**")
                            
                            fm_rf_reg_col1, fm_rf_reg_col2 = st.columns(2)
                            # Regression information
                            with fm_rf_reg_col1:
                                st.write("Regression information:")
                                st.table(model_full_results["RF information"].style.format(precision=user_precision))
                            # Regression statistics
                            with fm_rf_reg_col2:
                                st.write("Regression statistics:")
                                rf_error_est = pd.DataFrame(index = ["AUC ROC", "AP", "AUC PRC", "LOG-LOSS"], columns = ["Value"])
                                rf_error_est.loc["AUC ROC"] = model_full_results["model comparison thresInd"].loc["AUC ROC"]["Random Forest"]
                                rf_error_est.loc["AP"] = model_full_results["model comparison thresInd"].loc["AP"]["Random Forest"]
                                rf_error_est.loc["AUC PRC"] =  model_full_results["model comparison thresInd"].loc["AUC PRC"]["Random Forest"]
                                rf_error_est.loc["LOG-LOSS"] = model_full_results["model comparison thresInd"].loc["LOG-LOSS"]["Random Forest"]
                                st.table(rf_error_est.style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_RF_regStat_bin"))) 
                            st.write("")
                            fm_rf_figs1_col1, fm_rf_figs1_col2 = st.columns(2)
                            # Variable importance (via permutation)
                            with fm_rf_figs1_col1:
                                st.write("Variable importance (via permutation):")
                                rf_varImp_table = model_full_results["RF variable importance"]
                                st.table(rf_varImp_table.style.format(precision=user_precision))
                                st.write("")
                            with fm_rf_figs1_col2:
                                st.write("")
                                st.write("")
                                st.write("")
                                rf_varImp_plot_data = model_full_results["RF variable importance"]
                                rf_varImp_plot_data["Variable"] = rf_varImp_plot_data.index
                                rf_varImp = alt.Chart(rf_varImp_plot_data, height = 200).mark_bar().encode(
                                    x = alt.X("mean", title = "variable importance", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Variable", title = None, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), sort = None),
                                    tooltip = ["Variable", "mean"]
                                )
                                st.altair_chart(rf_varImp, use_container_width = True) 
                            st.write("") 
                            fm_rf_figs2_col1, fm_rf_figs2_col2 = st.columns(2)
                            # Feature importance
                            with fm_rf_figs2_col1:
                                st.write("Feature importance (impurity-based):")
                                rf_featImp_table = model_full_results["RF feature importance"]
                                st.table(rf_featImp_table.style.format(precision=user_precision))
                            with fm_rf_figs2_col2:
                                st.write("")
                                st.write("")
                                st.write("")
                                rf_featImp_plot_data = model_full_results["RF feature importance"]
                                rf_featImp_plot_data["Variable"] = rf_featImp_plot_data.index
                                rf_featImp = alt.Chart(rf_featImp_plot_data, height = 200).mark_bar().encode(
                                    x = alt.X("Value", title = "feature importance", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Variable", title = None, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), sort = None),
                                    tooltip = ["Variable", "Value"]
                                )
                                st.altair_chart(rf_featImp, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_RF_varImp_bin"))) 
                            st.write("") 
                            fm_rf_figs5_col1, fm_rf_figs5_col2 = st.columns(2)
                            # Observed vs. Probability of Occurrence 
                            with fm_rf_figs5_col1:
                                st.write("Observed vs. Probability of Occurrence:")
                                prob_data = pd.DataFrame(model_full_results["RF fitted"])
                                prob_data["Observed"] = df[response_var]
                                prob_data["ProbabilityOfOccurrence"] = prob_data[1]
                                prob_data["Threshold"] = model_full_results["model comparison thres"].loc["threshold"]["Random Forest"]
                                prob_data_plot = alt.Chart(prob_data, height = 200).mark_tick(size = 5, thickness = 1).encode(
                                    x = alt.X("ProbabilityOfOccurrence", title = "probability of occurrence", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Observed", title = "observed", scale = alt.Scale(domain = [min(prob_data["Observed"]), max(prob_data["Observed"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Observed", "ProbabilityOfOccurrence", "Threshold"]
                                )
                                thres = alt.Chart(prob_data, height = 200).mark_rule(size = 1.5, color = "darkred").encode(x = "Threshold", tooltip = ["Threshold"]) 
                                prob_plot = prob_data_plot + thres
                                st.altair_chart(prob_plot, use_container_width = True)
                            # ROC curve 
                            with fm_rf_figs5_col2:
                                st.write("ROC curve:")
                                AUC_ROC_data = pd.DataFrame()
                                AUC_ROC_data["FPR"] = model_full_results["RF ROC curve"][0]
                                AUC_ROC_data["TPR"] = model_full_results["RF ROC curve"][1]
                                AUC_ROC_data["AUC ROC"] = model_full_results["model comparison thresInd"].loc["AUC ROC"]["Random Forest"]
                                AUC_ROC_data["Threshold"] = model_full_results["model comparison thres"].loc["threshold"]["Random Forest"]
                                AUC_ROC_plot= alt.Chart(AUC_ROC_data, height = 200).mark_line().encode(
                                    x = alt.X("FPR", title = "1 - specificity (FPR)", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("TPR", title = "sensitivity (TPR)", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["TPR", "FPR", "AUC ROC"]
                                )
                                line = alt.Chart(
                                    pd.DataFrame({"FPR": [min(AUC_ROC_data["FPR"]), max(AUC_ROC_data["FPR"])], "TPR": [min(AUC_ROC_data["FPR"]), max(AUC_ROC_data["FPR"])]})).mark_line(size = 2, color = "darkred").encode(
                                            alt.X("FPR"),
                                            alt.Y("TPR"),
                                )
                                st.altair_chart(AUC_ROC_plot + line, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_RF_thresAUC")))
                            st.write("") 
                            # Partial dependence plots
                            st.write("Partial dependence plots:")    
                            fm_rf_figs3_col1, fm_rf_figs3_col2 = st.columns(2)
                            for pd_var in expl_var:
                                pd_data_rf = pd.DataFrame(columns = [pd_var])
                                pd_data_rf[pd_var] = model_full_results["RF partial dependence"][pd_var]["values"][0]
                                pd_data_rf["Partial dependence"] = model_full_results["RF partial dependence"][pd_var]["average"][0]
                                pd_chart_rf = alt.Chart(pd_data_rf, height = 200).mark_line(color = "darkred").encode(
                                    x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Partial dependence", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["RF partial dependence min/max"]["min"].min(), model_full_results["RF partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Partial dependence"] + [pd_var]
                                )
                                pd_data_ticks_rf = pd.DataFrame(columns = [pd_var])
                                pd_data_ticks_rf[pd_var] = df[pd_var]
                                pd_data_ticks_rf["y"] = [model_full_results["RF partial dependence min/max"]["min"].min()] * df.shape[0]
                                pd_ticks_rf = alt.Chart(pd_data_ticks_rf, height = 200).mark_tick(size = 5, thickness = 1).encode(
                                    x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), scale = alt.Scale(domain = [pd_data_ticks_rf[pd_var].min(), pd_data_ticks_rf[pd_var].max()])),
                                    y = alt.Y("y", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["RF partial dependence min/max"]["min"].min(), model_full_results["RF partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = [pd_var]
                                )
                                if expl_var.index(pd_var)%2 == 0:
                                    with fm_rf_figs3_col1:
                                        st.altair_chart(pd_ticks_rf + pd_chart_rf, use_container_width = True)
                                if expl_var.index(pd_var)%2 == 1:
                                    with fm_rf_figs3_col2:
                                        st.altair_chart(pd_ticks_rf + pd_chart_rf, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_RF_partDep_bin")))

                            # Download link for RF output
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            model_full_results["RF information"].to_excel(excel_file, sheet_name="regression_information")
                            rf_error_est.to_excel(excel_file, sheet_name="regression_statistics")
                            rf_varImp_table.to_excel(excel_file, sheet_name="variable_importance")
                            rf_featImp_table.to_excel(excel_file, sheet_name="feature_importance")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "RF full model output__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download Random Forest full model output</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("")

                        # BRT specific output
                        if any(a for a in sb_ML_alg if a == "Boosted Regression Trees"):
                            st.markdown("**Boosted Regression Trees**")
                            
                            fm_brt_reg_col1, fm_brt_reg_col2 = st.columns(2)
                            # Regression information
                            with fm_brt_reg_col1:
                                st.write("Regression information:")
                                st.table(model_full_results["BRT information"].style.format(precision=user_precision))
                            # Regression statistics
                            with fm_brt_reg_col2:
                                st.write("Regression statistics:")
                                brt_error_est = pd.DataFrame(index = ["AUC ROC", "AP", "AUC PRC", "LOG-LOSS"], columns = ["Value"])
                                brt_error_est.loc["AUC ROC"] = model_full_results["model comparison thresInd"].loc["AUC ROC"]["Boosted Regression Trees"]
                                brt_error_est.loc["AP"] = model_full_results["model comparison thresInd"].loc["AP"]["Boosted Regression Trees"]
                                brt_error_est.loc["AUC PRC"] =  model_full_results["model comparison thresInd"].loc["AUC PRC"]["Boosted Regression Trees"]
                                brt_error_est.loc["LOG-LOSS"] = model_full_results["model comparison thresInd"].loc["LOG-LOSS"]["Boosted Regression Trees"]
                                st.table(brt_error_est.style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_BRT_regStat_bin"))) 
                            st.write("")
                            # Training score (deviance vs. number of trees)
                            st.write("Training score:")
                            train_score = pd.DataFrame(index = range(model_full_results["BRT train score"].shape[0]), columns = ["Training deviance"])
                            train_score["Training deviance"] = model_full_results["BRT train score"]
                            train_score["Trees"] = train_score.index+1
                            train_score_plot = alt.Chart(train_score, height = 200).mark_line(color = "darkred").encode(
                                x = alt.X("Trees", title = "trees", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), scale = alt.Scale(domain = [train_score["Trees"].min(), train_score["Trees"].max()])),
                                y = alt.Y("Training deviance", title = "training deviance", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                tooltip = ["Training deviance", "Trees"]
                            )     
                            st.altair_chart(train_score_plot, use_container_width = True)
                            st.write("") 

                            fm_brt_figs1_col1, fm_brt_figs1_col2 = st.columns(2)
                            # Variable importance (via permutation)
                            with fm_brt_figs1_col1:
                                st.write("Variable importance (via permutation):")
                                brt_varImp_table = model_full_results["BRT variable importance"]
                                st.table(brt_varImp_table.style.format(precision=user_precision))
                                st.write("")
                            with fm_brt_figs1_col2:
                                st.write("")
                                st.write("")
                                st.write("")
                                brt_varImp_plot_data = model_full_results["BRT variable importance"]
                                brt_varImp_plot_data["Variable"] = brt_varImp_plot_data.index
                                brt_varImp = alt.Chart(brt_varImp_plot_data, height = 200).mark_bar().encode(
                                    x = alt.X("mean", title = "variable importance", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Variable", title = None, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), sort = None),
                                    tooltip = ["Variable", "mean"]
                                )
                                st.altair_chart(brt_varImp, use_container_width = True) 
                            st.write("") 
                            fm_brt_figs2_col1, fm_brt_figs2_col2 = st.columns(2)
                            # Feature importance
                            with fm_brt_figs2_col1:
                                st.write("Feature importance (impurity-based):")
                                brt_featImp_table = model_full_results["BRT feature importance"]
                                st.table(brt_featImp_table.style.format(precision=user_precision))
                            with fm_brt_figs2_col2:
                                st.write("")
                                st.write("")
                                st.write("")
                                brt_featImp_plot_data = model_full_results["BRT feature importance"]
                                brt_featImp_plot_data["Variable"] = brt_featImp_plot_data.index
                                brt_featImp = alt.Chart(brt_featImp_plot_data, height = 200).mark_bar().encode(
                                    x = alt.X("Value", title = "feature importance", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Variable", title = None, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), sort = None),
                                    tooltip = ["Variable", "Value"]
                                )
                                st.altair_chart(brt_featImp, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_BRT_varImp_bin"))) 
                            st.write("") 
                            fm_brt_figs5_col1, fm_brt_figs5_col2 = st.columns(2)
                            # Observed vs. Probability of Occurrence 
                            with fm_brt_figs5_col1:
                                st.write("Observed vs. Probability of Occurrence:")
                                prob_data = pd.DataFrame(model_full_results["BRT fitted"])
                                prob_data["Observed"] = df[response_var]
                                prob_data["ProbabilityOfOccurrence"] = prob_data[1]
                                prob_data["Threshold"] = model_full_results["model comparison thres"].loc["threshold"]["Boosted Regression Trees"]
                                prob_data_plot = alt.Chart(prob_data, height = 200).mark_tick(size = 5, thickness = 1).encode(
                                    x = alt.X("ProbabilityOfOccurrence", title = "probability of occurrence", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Observed", title = "observed", scale = alt.Scale(domain = [min(prob_data["Observed"]), max(prob_data["Observed"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Observed", "ProbabilityOfOccurrence", "Threshold"]
                                )
                                thres = alt.Chart(prob_data, height = 200).mark_rule(size = 1.5, color = "darkred").encode(x = "Threshold", tooltip = ["Threshold"]) 
                                prob_plot = prob_data_plot + thres
                                st.altair_chart(prob_plot, use_container_width = True)
                            # ROC curve 
                            with fm_brt_figs5_col2:
                                st.write("ROC curve:")
                                AUC_ROC_data = pd.DataFrame()
                                AUC_ROC_data["FPR"] = model_full_results["BRT ROC curve"][0]
                                AUC_ROC_data["TPR"] = model_full_results["BRT ROC curve"][1]
                                AUC_ROC_data["AUC ROC"] = model_full_results["model comparison thresInd"].loc["AUC ROC"]["Boosted Regression Trees"]
                                AUC_ROC_data["Threshold"] = model_full_results["model comparison thres"].loc["threshold"]["Boosted Regression Trees"]
                                AUC_ROC_plot= alt.Chart(AUC_ROC_data, height = 200).mark_line().encode(
                                    x = alt.X("FPR", title = "1 - specificity (FPR)", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("TPR", title = "sensitivity (TPR)", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["TPR", "FPR", "AUC ROC"]
                                )
                                line = alt.Chart(
                                    pd.DataFrame({"FPR": [min(AUC_ROC_data["FPR"]), max(AUC_ROC_data["FPR"])], "TPR": [min(AUC_ROC_data["FPR"]), max(AUC_ROC_data["FPR"])]})).mark_line(size = 2, color = "darkred").encode(
                                            alt.X("FPR"),
                                            alt.Y("TPR"),
                                )
                                st.altair_chart(AUC_ROC_plot + line, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_BRT_thresAUC")))
                            st.write("") 

                            # Partial dependence plots
                            st.write("Partial dependence plots:")    
                            fm_brt_figs3_col1, fm_brt_figs3_col2 = st.columns(2)
                            for pd_var in expl_var:
                                pd_data_brt = pd.DataFrame(columns = [pd_var])
                                pd_data_brt[pd_var] = model_full_results["BRT partial dependence"][pd_var]["values"][0]
                                pd_data_brt["Partial dependence"] = model_full_results["BRT partial dependence"][pd_var]["average"][0]
                                pd_chart_brt = alt.Chart(pd_data_brt, height = 200).mark_line(color = "darkred").encode(
                                x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                y = alt.Y("Partial dependence", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["BRT partial dependence min/max"]["min"].min(), model_full_results["BRT partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                tooltip = ["Partial dependence"] + [pd_var]
                                )
                                pd_data_ticks_brt = pd.DataFrame(columns = [pd_var])
                                pd_data_ticks_brt[pd_var] = df[pd_var]
                                pd_data_ticks_brt["y"] = [model_full_results["BRT partial dependence min/max"]["min"].min()] * df.shape[0]
                                pd_ticks_brt = alt.Chart(pd_data_ticks_brt, height = 200).mark_tick(size = 5, thickness = 1).encode(
                                    x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), scale = alt.Scale(domain = [pd_data_ticks_brt[pd_var].min(), pd_data_ticks_brt[pd_var].max()])),
                                    y = alt.Y("y", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["BRT partial dependence min/max"]["min"].min(), model_full_results["BRT partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = [pd_var]
                                )
                                if expl_var.index(pd_var)%2 == 0:
                                    with fm_brt_figs3_col1:
                                        st.altair_chart(pd_ticks_brt + pd_chart_brt, use_container_width = True)
                                if expl_var.index(pd_var)%2 == 1:
                                    with fm_brt_figs3_col2:
                                        st.altair_chart(pd_ticks_brt + pd_chart_brt, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_BRT_partDep_bin")))

                            # Download link for BRT output
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            model_full_results["BRT information"].to_excel(excel_file, sheet_name="regression_information")
                            brt_error_est.to_excel(excel_file, sheet_name="regression_statistics")
                            brt_varImp_table.to_excel(excel_file, sheet_name="variable_importance")
                            brt_featImp_table.to_excel(excel_file, sheet_name="feature_importance")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "BRT full model output__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download Boosted Regression Trees full model output</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("")
            
                        # ANN specific output
                        if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                            st.markdown("**Artificial Neural Networks**")

                            fm_ann_reg_col1, fm_ann_reg_col2 = st.columns(2)
                            # Regression information
                            with fm_ann_reg_col1:
                                st.write("Regression information:")
                                st.table(model_full_results["ANN information"].style.format(precision=user_precision))
                            # Regression statistics
                            with fm_ann_reg_col2:
                                st.write("Regression statistics:")
                                ann_error_est = pd.DataFrame(index = ["AUC ROC", "AP", "AUC PRC", "LOG-LOSS", "Best loss"], columns = ["Value"])
                                ann_error_est.loc["AUC ROC"] = model_full_results["model comparison thresInd"].loc["AUC ROC"]["Artificial Neural Networks"]
                                ann_error_est.loc["AP"] = model_full_results["model comparison thresInd"].loc["AP"]["Artificial Neural Networks"]
                                ann_error_est.loc["AUC PRC"] =  model_full_results["model comparison thresInd"].loc["AUC PRC"]["Artificial Neural Networks"]
                                ann_error_est.loc["LOG-LOSS"] =  model_full_results["model comparison thresInd"].loc["LOG-LOSS"]["Artificial Neural Networks"]
                                if ann_finalPara["weight optimization solver"][0] != "lbfgs":
                                    ann_error_est.loc["Best loss"] =  model_full_results["ANN loss"]
                                st.table(ann_error_est.style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_ANN_regStat_bin")))
                            st.write("")
                            # ANN architecture
                            st.write("Artificial Neural Network architecture:")
                            coef_list = full_model_ann_sk.coefs_
                            # weight matrix
                            annviz_weights = st.checkbox("Show weight matrix")
                            if annviz_weights:
                                layer_options = ["Input Layer <-> Hidden Layer 1"]
                                if int(model_full_results["ANN information"].loc['Layers']) == 3:
                                    layer_options += ["Hidden Layer 1 <-> Output Layer"]
                                elif int(model_full_results["ANN information"].loc['Layers']) == 4:
                                    layer_options += ["Hidden Layer 1 <-> Hidden Layer 2"]
                                    layer_options += ["Hidden Layer 2 <-> Output Layer"]
                                elif int(model_full_results["ANN information"].loc['Layers']) == 5:
                                    layer_options += ["Hidden Layer 1 <-> Hidden Layer 2"]
                                    layer_options += ["Hidden Layer 2 <-> Hidden Layer 3"]
                                    layer_options += ["Hidden Layer 3 <-> Output Layer"]
                                wei_matrix = st.selectbox('Weight matrix for following Layer', layer_options)
                                output = [response_var]
                                df_weights = ml.weight_matrix_func(output, expl_var, wei_matrix, coef_list)
                                st.write(df_weights)
                            annviz_output = st.checkbox("Show Artificial Neural Network Visualization")
                            if annviz_output:
                                st.write("Select which neurons of the hidden layer should be visualized:")
                                sli_col1, sli_col2 = st.columns(2)
                                # input layer
                                in_sel_nod = (1,len(expl_var))
                                # hidden layer 1
                                hi1_nod = int(ann_finalPara['hidden layer sizes'][0][0])
                                if hi1_nod >= 10:
                                    hi1_def_max = 10
                                else:
                                    hi1_def_max = hi1_nod
                                hi1_sel_nod = sli_col1.slider('Hidden Layer 1', min_value=1, max_value=hi1_nod, value=[1,hi1_def_max])
                                hi_sel_tup = (hi1_sel_nod[1]-hi1_sel_nod[0]+1,)
                                # hidden layer 2
                                if int(model_full_results["ANN information"].loc['Layers']) >= 4:
                                    hi2_nod = int(ann_finalPara['hidden layer sizes'][0][1])
                                    if hi2_nod >= 10:
                                        hi2_def_max = 10
                                    else:
                                        hi2_def_max = hi2_nod
                                    hi2_sel_nod = sli_col2.slider('Hidden Layer 2', min_value=1, max_value=hi2_nod, value=[1,hi2_def_max])
                                    hi_sel_tup += (hi2_sel_nod[1]-hi2_sel_nod[0]+1,)
                                # hidden layer 3
                                if int(model_full_results["ANN information"].loc['Layers']) >= 5:
                                    hi3_nod = int(ann_finalPara['hidden layer sizes'][0][2])
                                    if hi3_nod >= 10:
                                        hi3_def_max = 10
                                    else:
                                        hi3_def_max = hi3_nod
                                    hi3_sel_nod = sli_col1.slider('Hidden Layer 3', min_value=1, max_value=hi3_nod, value=[1,hi3_def_max])
                                    hi_sel_tup += (hi3_sel_nod[1]-hi3_sel_nod[0]+1,)
                                
                                # ANN Visualization
                                st.write("")
                                st.warning("Very large artificial neural networks cannot be visualized clearly. Recommendation: display max. 20 neurons in one layer.")
                                
                                numb_output = len([response_var])
                                network_structure = np.hstack(([in_sel_nod[1]-in_sel_nod[0]+1], np.asarray(hi_sel_tup), [numb_output]))
                                
                                # seperate weight matrix
                                if int(model_full_results["ANN information"].loc['Layers']) == 3:
                                    in_hi_wei = coef_list[0]
                                    hi_out_wei = coef_list[1]
                                elif int(model_full_results["ANN information"].loc['Layers']) == 4:
                                    in_hi_wei = coef_list[0]
                                    hi1_hi2_wei = coef_list[1]
                                    hi_out_wei = coef_list[2]
                                elif int(model_full_results["ANN information"].loc['Layers']) == 5:
                                    in_hi_wei = coef_list[0]
                                    hi1_hi2_wei = coef_list[1]
                                    hi2_hi3_wei = coef_list[2]
                                    hi_out_wei = coef_list[3]

                                # weights for selected nodes
                                sel_in_hi_wei = in_hi_wei[in_sel_nod[0]-1:in_sel_nod[1], hi1_sel_nod[0]-1:hi1_sel_nod[1]]
                                sel_coef_list = [sel_in_hi_wei]
                                # 1 hidden layer
                                if int(model_full_results["ANN information"].loc['Layers']) == 3:
                                    sel_hi_out_wei = hi_out_wei[hi1_sel_nod[0]-1:hi1_sel_nod[1], 0:numb_output]
                                    sel_coef_list += [sel_hi_out_wei]
                                # 2 hidden layer
                                elif int(model_full_results["ANN information"].loc['Layers']) == 4:
                                    sel_hi1_hi2_wei = hi1_hi2_wei[hi1_sel_nod[0]-1:hi1_sel_nod[1], hi2_sel_nod[0]-1:hi2_sel_nod[1]]
                                    sel_hi_out_wei = hi_out_wei[hi2_sel_nod[0]-1:hi2_sel_nod[1], 0:numb_output]
                                    sel_coef_list += [sel_hi1_hi2_wei]
                                    sel_coef_list += [sel_hi_out_wei]
                                # 3 hidden layer
                                elif int(model_full_results["ANN information"].loc['Layers']) == 5:
                                    sel_hi1_hi2_wei = hi1_hi2_wei[hi1_sel_nod[0]-1:hi1_sel_nod[1], hi2_sel_nod[0]-1:hi2_sel_nod[1]]
                                    sel_hi2_hi3_wei = hi2_hi3_wei[hi2_sel_nod[0]-1:hi2_sel_nod[1], hi3_sel_nod[0]-1:hi3_sel_nod[1]]
                                    sel_hi_out_wei = hi_out_wei[hi3_sel_nod[0]-1:hi3_sel_nod[1], 0:numb_output]
                                    sel_coef_list += [sel_hi1_hi2_wei]
                                    sel_coef_list += [sel_hi2_hi3_wei]
                                    sel_coef_list += [sel_hi_out_wei]                                
                                
                                network=VisNN.DrawNN(network_structure, sel_coef_list)
                                st.set_option('deprecation.showPyplotGlobalUse', False)
                                st.write("")
                                st.pyplot(network.draw())
                            st.write("")
                            # Loss curve (loss vs. number of iterations (epochs))
                            if ann_finalPara["weight optimization solver"][0] != "lbfgs":
                                st.write("Loss curve:")
                                loss_curve = pd.DataFrame(index = range(len(model_full_results["ANN loss curve"])), columns = ["Loss"])
                                loss_curve["Loss"] = model_full_results["ANN loss curve"]
                                loss_curve["Iterations"] = loss_curve.index+1
                                loss_curve_plot = alt.Chart(loss_curve, height = 200).mark_line(color = "darkred").encode(
                                    x = alt.X("Iterations", title = "iterations (epochs)", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), scale = alt.Scale(domain = [loss_curve["Iterations"].min(), loss_curve["Iterations"].max()])),
                                    y = alt.Y("Loss", title = "loss", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Loss", "Iterations"]
                                )     
                                st.altair_chart(loss_curve_plot, use_container_width = True)    
                            st.write("") 
                            fm_ann_figs1_col1, fm_ann_figs1_col2 = st.columns(2)
                            # Variable importance (via permutation)
                            with fm_ann_figs1_col1:
                                st.write("Variable importance (via permutation):")
                                ann_varImp_table = model_full_results["ANN variable importance"]
                                st.table(ann_varImp_table.style.format(precision=user_precision))
                            with fm_ann_figs1_col2:
                                st.write("")
                                st.write("")
                                st.write("")
                                ann_varImp_plot_data = model_full_results["ANN variable importance"]
                                ann_varImp_plot_data["Variable"] = ann_varImp_plot_data.index
                                ann_varImp = alt.Chart(ann_varImp_plot_data, height = 200).mark_bar().encode(
                                    x = alt.X("mean", title = "variable importance", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Variable", title = None, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), sort = None),
                                    tooltip = ["Variable", "mean"]
                                )
                                st.altair_chart(ann_varImp, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_ANN_varImp_bin")))
                            st.write("") 
                            fm_ann_figs5_col1, fm_ann_figs5_col2 = st.columns(2)
                            # Observed vs. Probability of Occurrence 
                            with fm_ann_figs5_col1:
                                st.write("Observed vs. Probability of Occurrence:")
                                prob_data = pd.DataFrame(model_full_results["ANN fitted"])
                                prob_data["Observed"] = df[response_var]
                                prob_data["ProbabilityOfOccurrence"] = prob_data[1]
                                prob_data["Threshold"] = model_full_results["model comparison thres"].loc["threshold"]["Artificial Neural Networks"]
                                prob_data_plot = alt.Chart(prob_data, height = 200).mark_tick(size = 5, thickness = 1).encode(
                                    x = alt.X("ProbabilityOfOccurrence", title = "probability of occurrence", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Observed", title = "observed", scale = alt.Scale(domain = [min(prob_data["Observed"]), max(prob_data["Observed"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Observed", "ProbabilityOfOccurrence", "Threshold"]
                                )
                                thres = alt.Chart(prob_data, height = 200).mark_rule(size = 1.5, color = "darkred").encode(x = "Threshold", tooltip = ["Threshold"]) 
                                prob_plot = prob_data_plot + thres
                                st.altair_chart(prob_plot, use_container_width = True)
                            # ROC curve 
                            with fm_ann_figs5_col2:
                                st.write("ROC curve:")
                                AUC_ROC_data = pd.DataFrame()
                                AUC_ROC_data["FPR"] = model_full_results["ANN ROC curve"][0]
                                AUC_ROC_data["TPR"] = model_full_results["ANN ROC curve"][1]
                                AUC_ROC_data["AUC ROC"] = model_full_results["model comparison thresInd"].loc["AUC ROC"]["Artificial Neural Networks"]
                                AUC_ROC_data["Threshold"] = model_full_results["model comparison thres"].loc["threshold"]["Artificial Neural Networks"]
                                AUC_ROC_plot= alt.Chart(AUC_ROC_data, height = 200).mark_line().encode(
                                    x = alt.X("FPR", title = "1 - specificity (FPR)", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("TPR", title = "sensitivity (TPR)", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["TPR", "FPR", "AUC ROC"]
                                )
                                line = alt.Chart(
                                    pd.DataFrame({"FPR": [min(AUC_ROC_data["FPR"]), max(AUC_ROC_data["FPR"])], "TPR": [min(AUC_ROC_data["FPR"]), max(AUC_ROC_data["FPR"])]})).mark_line(size = 2, color = "darkred").encode(
                                            alt.X("FPR"),
                                            alt.Y("TPR"),
                                )
                                st.altair_chart(AUC_ROC_plot + line, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_ANN_thresAUC")))
                            st.write("") 

                            # Partial dependence plots
                            st.write("Partial dependence plots:")    
                            fm_ann_figs2_col1, fm_ann_figs2_col2 = st.columns(2)
                            for pd_var in expl_var:
                                pd_data_ann = pd.DataFrame(columns = [pd_var])
                                pd_data_ann[pd_var] = (model_full_results["ANN partial dependence"][pd_var]["values"][0]*(df[pd_var].std()))+df[pd_var].mean()
                                pd_data_ann["Partial dependence"] = model_full_results["ANN partial dependence"][pd_var]["average"][0]
                                pd_chart_ann = alt.Chart(pd_data_ann, height = 200).mark_line(color = "darkred").encode(
                                x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                y = alt.Y("Partial dependence", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["ANN partial dependence min/max"]["min"].min(), model_full_results["ANN partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                tooltip = ["Partial dependence"] + [pd_var]
                                )
                                pd_data_ticks_ann = pd.DataFrame(columns = [pd_var])
                                pd_data_ticks_ann[pd_var] = df[pd_var]
                                pd_data_ticks_ann["y"] = [model_full_results["ANN partial dependence min/max"]["min"].min()] * df.shape[0]
                                pd_ticks_ann = alt.Chart(pd_data_ticks_ann, height = 200).mark_tick(size = 5, thickness = 1).encode(
                                    x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), scale = alt.Scale(domain = [pd_data_ticks_ann[pd_var].min(), pd_data_ticks_ann[pd_var].max()])),
                                    y = alt.Y("y", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["ANN partial dependence min/max"]["min"].min(), model_full_results["ANN partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = [pd_var]
                                )
                                if expl_var.index(pd_var)%2 == 0:
                                    with fm_ann_figs2_col1:
                                        st.altair_chart(pd_ticks_ann + pd_chart_ann, use_container_width = True)
                                if expl_var.index(pd_var)%2 == 1:
                                    with fm_ann_figs2_col2:
                                        st.altair_chart(pd_ticks_ann + pd_chart_ann, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_ANN_partDep_bin")))

                            # Download link for ANN output
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            model_full_results["ANN information"].to_excel(excel_file, sheet_name="regression_information")
                            ann_error_est.to_excel(excel_file, sheet_name="regression_statistics")
                            ann_varImp_table.to_excel(excel_file, sheet_name="variable_importance")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "ANN full model output__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download Artificial Neural Networks full model output</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("") 

                        # Performance metrics across all models
                        if any(a for a in sb_ML_alg if a == "Logistic Regression" or a == "Random Forest" or a == "Generalized Additive Models" or a == "Boosted Regression Trees" or a == "Artificial Neural Networks"):
                            st.markdown("**Model comparison**")
                            st.write("Threshold-independent metrics:")
                            st.write((model_full_results["model comparison thresInd"]).transpose().style.format(precision=user_precision))
                            if len(sb_ML_alg) > 1:
                                if sett_hints:
                                    st.info(str(fc.learning_hints("mod_md_modCompThresInd")))
                            st.write("")
                            st.write("Thresholds:")
                            st.table(model_full_results["model comparison thres"].transpose().style.format(precision=user_precision))
                            st.write("")
                            st.write("Threshold-dependent metrics:")
                            st.write((model_full_results["model comparison thresDep"]).transpose().style.format(precision=user_precision))
                            if len(sb_ML_alg) > 1:
                                if sett_hints:
                                    st.info(str(fc.learning_hints("mod_md_modCompThresDep")))
                            st.write("")
                            
                            # Download link for model comparison output
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            model_full_results["model comparison thresInd"].transpose().to_excel(excel_file, sheet_name="thresh_independent_metrics")
                            model_full_results["model comparison thres"].to_excel(excel_file, sheet_name="thresholds")
                            model_full_results["model comparison thresDep"].transpose().to_excel(excel_file, sheet_name="thresh_dependent_metrics")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "Model comparison full model output__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download model comparison output</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("")
                
                else:
                    st.warning("Please run models!")
                st.write("")

            #--------------------------------------------------------------------------------------
            # FULL MODEL PREDICTIONS

            prediction_output = st.expander("Full model predictions", expanded = False)
            with prediction_output:
                
                if model_full_results is not None:

                    #-------------------------------------------------------------

                    # Continuous response variable
                    if response_var_type == "continuous":

                        # MLR specific output
                        if any(a for a in sb_ML_alg if a == "Multiple Linear Regression"):
                            st.markdown("**Multiple Linear Regression**")

                            pred_col1, pred_col2 = st.columns(2)
                            with pred_col1:
                                st.write("Predictions for original data:")
                                MLR_pred_orig = pd.DataFrame(columns = [response_var])
                                MLR_pred_orig[response_var] = model_full_results["MLR fitted"]
                                st.write(MLR_pred_orig.style.format(precision=user_precision))
                            with pred_col2:
                                if do_modprednew == "Yes":
                                    st.write("Predictions for new data:")
                                    MLR_pred_new = pd.DataFrame(columns = [response_var])
                                    MLR_pred_new[response_var] = model_full_results["MLR prediction"]
                                    st.write(MLR_pred_new.style.format(precision=user_precision))

                        # GAM specific output
                        if any(a for a in sb_ML_alg if a == "Generalized Additive Models"):
                            st.markdown("**Generalized Additive Models**")

                            pred_col1, pred_col2 = st.columns(2)
                            with pred_col1:
                                st.write("Predictions for original data:")
                                GAM_pred_orig = pd.DataFrame(columns = [response_var])
                                GAM_pred_orig[response_var] = model_full_results["GAM fitted"]
                                st.write(GAM_pred_orig.style.format(precision=user_precision))
                            with pred_col2:
                                if do_modprednew == "Yes":
                                    st.write("Predictions for new data:")
                                    GAM_pred_new = pd.DataFrame(columns = [response_var])
                                    GAM_pred_new[response_var] = model_full_results["GAM prediction"]
                                    st.write(GAM_pred_new.style.format(precision=user_precision))
                        
                        # RF specific output
                        if any(a for a in sb_ML_alg if a == "Random Forest"):
                            st.markdown("**Random Forest**")

                            pred_col1, pred_col2 = st.columns(2)
                            with pred_col1:
                                st.write("Predictions for original data:")
                                RF_pred_orig = pd.DataFrame(columns = [response_var])
                                RF_pred_orig[response_var] = model_full_results["RF fitted"]
                                st.write(RF_pred_orig.style.format(precision=user_precision))
                            with pred_col2:
                                if do_modprednew == "Yes":
                                    st.write("Predictions for new data:")
                                    RF_pred_new = pd.DataFrame(columns = [response_var])
                                    RF_pred_new[response_var] = model_full_results["RF prediction"]
                                    st.write(RF_pred_new.style.format(precision=user_precision))
                        
                        # BRT specific output
                        if any(a for a in sb_ML_alg if a == "Boosted Regression Trees"):
                            st.markdown("**Boosted Regression Trees**")

                            pred_col1, pred_col2 = st.columns(2)
                            with pred_col1:
                                st.write("Predictions for original data:")
                                BRT_pred_orig = pd.DataFrame(columns = [response_var])
                                BRT_pred_orig[response_var] = model_full_results["BRT fitted"]
                                st.write(BRT_pred_orig.style.format(precision=user_precision))
                            with pred_col2:
                                if do_modprednew == "Yes":
                                    st.write("Predictions for new data:")
                                    BRT_pred_new = pd.DataFrame(columns = [response_var])
                                    BRT_pred_new[response_var] = model_full_results["BRT prediction"]
                                    st.write(BRT_pred_new.style.format(precision=user_precision))
                        
                        # ANN specific output
                        if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                            st.markdown("**Artificial Neural Networks**")

                            pred_col1, pred_col2 = st.columns(2)
                            with pred_col1:
                                st.write("Predictions for original data:")
                                ANN_pred_orig = pd.DataFrame(columns = [response_var])
                                ANN_pred_orig[response_var] = model_full_results["ANN fitted"]
                                st.write(ANN_pred_orig.style.format(precision=user_precision))
                            with pred_col2:
                                if do_modprednew == "Yes":
                                    st.write("Predictions for new data:")
                                    ANN_pred_new = pd.DataFrame(columns = [response_var])
                                    ANN_pred_new[response_var] = model_full_results["ANN prediction"]
                                    st.write(ANN_pred_new.style.format(precision=user_precision))                        

                    #-------------------------------------------------------------

                    # Binary response variable
                    if response_var_type == "binary":

                        # MLR specific output
                        if any(a for a in sb_ML_alg if a == "Multiple Linear Regression"):
                            st.markdown("**Multiple Linear Regression**")

                            pred_col1, pred_col2 = st.columns(2)
                            with pred_col1:
                                st.write("Predictions for original data:")
                                MLR_pred_orig = pd.DataFrame(columns = [response_var])
                                MLR_pred_orig[response_var] = model_full_results["MLR fitted"]
                                st.write(MLR_pred_orig.style.format(precision=user_precision))
                            with pred_col2:
                                if do_modprednew == "Yes":
                                    st.write("Predictions for new data:")
                                    MLR_pred_new = pd.DataFrame(columns = [response_var])
                                    MLR_pred_new[response_var] = model_full_results["MLR prediction"]
                                    st.write(MLR_pred_new.style.format(precision=user_precision))
                            st.write("")

                        # LR specific output
                        if any(a for a in sb_ML_alg if a == "Logistic Regression"):
                            st.markdown("**Logistic Regression**")

                            pred_col1, pred_col2 = st.columns(2)
                            with pred_col1:
                                st.write("Predictions for original data:")
                                LR_pred_orig = pd.DataFrame(columns = [response_var])
                                LR_pred_orig[response_var] = model_full_results["LR fitted"][:, 1]
                                LR_pred_orig[response_var + "_binary"] = model_full_results["LR fitted binary"]
                                st.write(LR_pred_orig.style.format(precision=user_precision))
                            with pred_col2:
                                if do_modprednew == "Yes":
                                    st.write("Predictions for new data:")
                                    LR_pred_new = pd.DataFrame(columns = [response_var])
                                    LR_pred_new[response_var] = model_full_results["LR prediction"][:, 1]
                                    LR_pred_new[response_var + "_binary"] = model_full_results["LR prediction binary"]
                                    st.write(LR_pred_new.style.format(precision=user_precision))
                            st.write("")

                        # GAM specific output
                        if any(a for a in sb_ML_alg if a == "Generalized Additive Models"):
                            st.markdown("**Generalized Additive Models**")

                            pred_col1, pred_col2 = st.columns(2)
                            with pred_col1:
                                st.write("Predictions for original data:")
                                GAM_pred_orig = pd.DataFrame(columns = [response_var])
                                GAM_pred_orig[response_var] = model_full_results["GAM fitted"]
                                GAM_pred_orig[response_var + "_binary"] = model_full_results["GAM fitted binary"]
                                st.write(GAM_pred_orig.style.format(precision=user_precision))
                            with pred_col2:
                                if do_modprednew == "Yes":
                                    st.write("Predictions for new data:")
                                    GAM_pred_new = pd.DataFrame(columns = [response_var])
                                    GAM_pred_new[response_var] = model_full_results["GAM prediction"]
                                    GAM_pred_new[response_var + "_binary"] = model_full_results["GAM prediction binary"]
                                    st.write(GAM_pred_new.style.format(precision=user_precision))
                            st.write("")

                        # RF specific output
                        if any(a for a in sb_ML_alg if a == "Random Forest"):
                            st.markdown("**Random Forest**")

                            pred_col1, pred_col2 = st.columns(2)
                            with pred_col1:
                                st.write("Predictions for original data:")
                                RF_pred_orig = pd.DataFrame(columns = [response_var])
                                RF_pred_orig[response_var] = model_full_results["RF fitted"][:, 1]
                                RF_pred_orig[response_var + "_binary"] = model_full_results["RF fitted binary"]
                                st.write(RF_pred_orig.style.format(precision=user_precision))
                            with pred_col2:
                                if do_modprednew == "Yes":
                                    st.write("Predictions for new data:")
                                    RF_pred_new = pd.DataFrame(columns = [response_var])
                                    RF_pred_new[response_var] = model_full_results["RF prediction"][:, 1]
                                    RF_pred_new[response_var + "_binary"] = model_full_results["RF prediction binary"]
                                    st.write(RF_pred_new.style.format(precision=user_precision))
                            st.write("")
                        
                        # BRT specific output
                        if any(a for a in sb_ML_alg if a == "Boosted Regression Trees"):
                            st.markdown("**Boosted Regression Trees**")

                            pred_col1, pred_col2 = st.columns(2)
                            with pred_col1:
                                st.write("Predictions for original data:")
                                BRT_pred_orig = pd.DataFrame(columns = [response_var])
                                BRT_pred_orig[response_var] = model_full_results["BRT fitted"][:, 1]
                                BRT_pred_orig[response_var + "_binary"] = model_full_results["BRT fitted binary"]
                                st.write(BRT_pred_orig.style.format(precision=user_precision))
                            with pred_col2:
                                if do_modprednew == "Yes":
                                    st.write("Predictions for new data:")
                                    BRT_pred_new = pd.DataFrame(columns = [response_var])
                                    BRT_pred_new[response_var] = model_full_results["BRT prediction"][:, 1]
                                    BRT_pred_new[response_var + "_binary"] = model_full_results["BRT prediction binary"]
                                    st.write(BRT_pred_new.style.format(precision=user_precision))
                            st.write("")
                        
                        # ANN specific output
                        if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                            st.markdown("**Artificial Neural Networks**")

                            pred_col1, pred_col2 = st.columns(2)
                            with pred_col1:
                                st.write("Predictions for original data:")
                                ANN_pred_orig = pd.DataFrame(columns = [response_var])
                                ANN_pred_orig[response_var] = model_full_results["ANN fitted"][:, 1]
                                ANN_pred_orig[response_var + "_binary"] = model_full_results["ANN fitted binary"]
                                st.write(ANN_pred_orig.style.format(precision=user_precision))
                            with pred_col2:
                                if do_modprednew == "Yes":
                                    st.write("Predictions for new data:")
                                    ANN_pred_new = pd.DataFrame(columns = [response_var])
                                    ANN_pred_new[response_var] = model_full_results["ANN prediction"][:, 1]
                                    ANN_pred_new[response_var + "_binary"] = model_full_results["ANN prediction binary"]
                                    st.write(ANN_pred_new.style.format(precision=user_precision))
                            st.write("")
                    
                    #-------------------------------------------------------------
                    st.write("")
                    # Download links for prediction data
                    output = BytesIO()
                    predictions_excel = pd.ExcelWriter(output, engine="xlsxwriter")
                    if any(a for a in sb_ML_alg if a == "Multiple Linear Regression"):
                        MLR_pred_orig.to_excel(predictions_excel, sheet_name="MLR_pred_orig")
                        if do_modprednew == "Yes":
                            MLR_pred_new.to_excel(predictions_excel, sheet_name="MLR_pred_new")
                    if any(a for a in sb_ML_alg if a == "Logistic Regression"):
                        LR_pred_orig.to_excel(predictions_excel, sheet_name="LR_pred_orig")
                        if do_modprednew == "Yes":
                            LR_pred_new.to_excel(predictions_excel, sheet_name="LR_pred_new")
                    if any(a for a in sb_ML_alg if a == "Generalized Additive Models"):
                        GAM_pred_orig.to_excel(predictions_excel, sheet_name="GAM_pred_orig")
                        if do_modprednew == "Yes":
                            GAM_pred_new.to_excel(predictions_excel, sheet_name="GAM_pred_new")
                    if any(a for a in sb_ML_alg if a == "Random Forest"):
                        RF_pred_orig.to_excel(predictions_excel, sheet_name="RF_pred_orig")
                        if do_modprednew == "Yes":
                            RF_pred_new.to_excel(predictions_excel, sheet_name="RF_pred_new")
                    if any(a for a in sb_ML_alg if a == "Boosted Regression Trees"):
                        BRT_pred_orig.to_excel(predictions_excel, sheet_name="BRT_pred_orig")
                        if do_modprednew == "Yes":
                            BRT_pred_new.to_excel(predictions_excel, sheet_name="BRT_pred_new")
                    if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                        ANN_pred_orig.to_excel(predictions_excel, sheet_name="ANN_pred_orig")
                        if do_modprednew == "Yes":
                            ANN_pred_new.to_excel(predictions_excel, sheet_name="ANN_pred_new")
                    predictions_excel.close()
                    predictions_excel = output.getvalue()
                    b64 = base64.b64encode(predictions_excel)
                    dl_file_name = "Full model predictions__" + df_name + ".xlsx"
                    st.markdown(
                        f"""
                    <a href="data:file/predictions_excel;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download full model predictions</a>
                    """,
                    unsafe_allow_html=True)
                    st.write("")

            #--------------------------------------------------------------------------------------
            # VALIDATION OUTPUT
            
            if do_modval == "Yes":
                val_output = st.expander("Validation output", expanded = False)
                with val_output:
                    if model_val_results is not None:
                        
                        #------------------------------------
                        # Continuous response variable

                        if response_var_type == "continuous":

                            # Metrics means
                            st.write("Means of metrics across validation runs:")
                            st.write(model_val_results["mean"].transpose().style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_val_means")))
                            # Metrics sd
                            st.write("SDs of metrics across validation runs:")
                            st.write(model_val_results["sd"].transpose().style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_val_sds")))
                            st.write("")
                            st.write("")
                            val_col1, val_col2 = st.columns(2)
                            with val_col1:
                                # Residuals boxplot
                                if model_val_results["residuals"] is not None:
                                    st.write("Boxplot of residuals across validation runs:")
                                    residual_results = model_val_results["residuals"]
                                    residuals_bplot = pd.melt(residual_results, ignore_index = False, var_name = "Algorithm", value_name = "Residuals")
                                    residuals_boxchart = alt.Chart(residuals_bplot, height = 200).mark_boxplot(color = "#1f77b4", median = dict(color = "darkred")).encode(
                                        x = alt.X("Residuals", title = "residuals", scale = alt.Scale(zero = False)),
                                        y = alt.Y("Algorithm", title = None),
                                        color = alt.Color("Algorithm", legend = None)
                                    ).configure_axis(
                                        labelFontSize = 12,
                                        titleFontSize = 12
                                    )
                                    residuals_plot = residuals_boxchart #+ residuals_scatter
                                    st.altair_chart(residuals_plot, use_container_width=True)
                                    if sett_hints:
                                        st.info(str(fc.learning_hints("mod_md_val_resBoxplot")))
                            with val_col2:
                                # Variance explained boxplot
                                if model_val_results["variance explained"] is not None:
                                    st.write("Boxplot of % VE across validation runs:")
                                    ve_results = model_val_results["variance explained"]
                                    ve_bplot = pd.melt(ve_results, ignore_index = False, var_name = "Algorithm", value_name = "% VE")
                                    ve_boxchart = alt.Chart(ve_bplot, height = 200).mark_boxplot(color = "#1f77b4", median = dict(color = "darkred")).encode(
                                        x = alt.X("% VE", scale = alt.Scale(domain = [min(ve_bplot["% VE"]), max(ve_bplot["% VE"])])),
                                        y = alt.Y("Algorithm", title = None),
                                        color = alt.Color("Algorithm", legend = None)
                                    ).configure_axis(
                                        labelFontSize = 12,
                                        titleFontSize = 12
                                    )
                                    st.altair_chart(ve_boxchart, use_container_width = True)
                                    if sett_hints:
                                        st.info(str(fc.learning_hints("mod_md_val_VEBoxplot")))
                            st.write("") 
                            # Variable importance (via permutation)
                            st.write("Means of variable importances:")
                            varImp_table_mean = model_val_results["variable importance mean"]
                            st.write(varImp_table_mean.transpose().style.format(precision=user_precision))
                            st.write("SDs of variable importances:")
                            varImp_table_sd = model_val_results["variable importance sd"]
                            st.write(varImp_table_sd.transpose().style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_val_varImp")))
                            st.write("")
                            st.write("")
                            # Residuals
                            if model_val_results["residuals"] is not None:
                                model_val_res = pd.DataFrame(index = ["min", "25%-Q", "median", "75%-Q", "max"], columns = sb_ML_alg)
                                for m in sb_ML_alg: 
                                    model_val_res.loc["min"][m] = model_val_results["residuals"][m].min()
                                    model_val_res.loc["25%-Q"][m] = model_val_results["residuals"][m].quantile(q = 0.25)
                                    model_val_res.loc["median"][m] = model_val_results["residuals"][m].quantile(q = 0.5)
                                    model_val_res.loc["75%-Q"][m] = model_val_results["residuals"][m].quantile(q = 0.75)
                                    model_val_res.loc["max"][m] = model_val_results["residuals"][m].max()
                                st.write("Residuals distribution across all validation runs:")
                                st.write(model_val_res.transpose().style.format(precision=user_precision))
                                if sett_hints:
                                    st.info(str(fc.learning_hints("mod_md_val_res")))
                            st.write("")

                            # Download link for validation output
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            model_val_results["mean"].transpose().to_excel(excel_file, sheet_name="performance_metrics_mean")
                            model_val_results["sd"].transpose().to_excel(excel_file, sheet_name="performance_metrics_sd")
                            varImp_table_mean.to_excel(excel_file, sheet_name="variable_importance_mean")
                            varImp_table_sd.to_excel(excel_file, sheet_name="variable_importance_sd")
                            model_val_res.transpose().to_excel(excel_file, sheet_name="residuals_distribution")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "Validation output__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download validation output</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("")

                        #------------------------------------
                        # Binary response variable

                        if response_var_type == "binary":
                            if model_val_results["mean_ind"].empty:
                                st.warning("Please select an additional algorithm besides Multiple Linear Regression!")
                            
                            # Metrics (independent)
                            if model_val_results["mean_ind"].empty:
                                st.write("")
                            else:
                                st.write("Means of threshold-independent metrics across validation runs:")
                                st.write(model_val_results["mean_ind"].transpose().style.format(precision=user_precision))
                                if sett_hints:
                                    st.info(str(fc.learning_hints("mod_md_val_means_thresInd")))
                            # Metrics (independent)
                            if model_val_results["sd_ind"].empty:
                                st.write("")
                            else:
                                st.write("SDs of threshold-independent metrics across validation runs:")
                                st.write(model_val_results["sd_ind"].transpose().style.format(precision=user_precision))
                                if sett_hints:
                                    st.info(str(fc.learning_hints("mod_md_val_sds_thresInd")))
                                st.write("")
                                st.write("")

                            val_col1, val_col2 = st.columns(2)
                            with val_col1: 
                                # AUC ROC boxplot
                                if model_val_results["AUC ROC"].empty:
                                    st.write("")
                                else:
                                    st.write("Boxplot of AUC ROC across validation runs:")
                                    auc_results = model_val_results["AUC ROC"]
                                    auc_bplot = pd.melt(auc_results, ignore_index = False, var_name = "Algorithm", value_name = "Value")
                                    auc_boxchart = alt.Chart(auc_bplot, height = 200).mark_boxplot(color = "#1f77b4", median = dict(color = "darkred")).encode(
                                        x = alt.X("Value", title = "AUC ROC", scale = alt.Scale(domain = [min(auc_bplot["Value"]), max(auc_bplot["Value"])])),
                                        y = alt.Y("Algorithm", title = None),
                                        color = alt.Color("Algorithm", legend = None)
                                    ).configure_axis(
                                        labelFontSize = 12,
                                        titleFontSize = 12
                                    )
                                    st.altair_chart(auc_boxchart, use_container_width = True) 
                                    if sett_hints:
                                        st.info(str(fc.learning_hints("mod_md_val_AUCBoxplot")))
                            with val_col2: 
                                # TSS boxplot
                                if model_val_results["TSS"].empty:
                                    st.write("")
                                else:
                                    st.write("Boxplot of TSS across validation runs:")
                                    tss_results = model_val_results["TSS"]
                                    tss_bplot = pd.melt(tss_results, ignore_index = False, var_name = "Algorithm", value_name = "Value")
                                    tss_boxchart = alt.Chart(tss_bplot, height = 200).mark_boxplot(color = "#1f77b4", median = dict(color = "darkred")).encode(
                                        x = alt.X("Value", title = "TSS", scale = alt.Scale(domain = [min(tss_bplot["Value"]), max(tss_bplot["Value"])])),
                                        y = alt.Y("Algorithm", title = None),
                                        color = alt.Color("Algorithm", legend = None)
                                    ).configure_axis(
                                        labelFontSize = 12,
                                        titleFontSize = 12
                                    )
                                    st.altair_chart(tss_boxchart, use_container_width = True)
                                    if sett_hints:
                                        st.info(str(fc.learning_hints("mod_md_val_TSSBoxplot")))
                            st.write("") 

                            # Variable importance
                            st.write("Means of variable importances:")
                            varImp_table_mean = model_val_results["variable importance mean"]
                            st.write(varImp_table_mean.style.format(precision=user_precision))
                            st.write("SDs of variable importances:")
                            varImp_table_sd = model_val_results["variable importance sd"]
                            st.write(varImp_table_sd.style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_val_varImp_bin")))
                            st.write("")
                            st.write("")

                            # Metrics (dependent)
                            if model_val_results["mean_dep"].empty:
                                st.write("")
                            else:
                                st.write("Means of threshold-dependent metrics across validation runs:")
                                st.write(model_val_results["mean_dep"].transpose().style.format(precision=user_precision))
                                if sett_hints:
                                    st.info(str(fc.learning_hints("mod_md_val_means_thresDep")))
                            # Metrics (dependent)
                            if model_val_results["sd_dep"].empty:
                                st.write("")
                            else:
                                st.write("SDs of threshold-dependent metrics across validation runs:")
                                st.write(model_val_results["sd_dep"].transpose().style.format(precision=user_precision))
                                if sett_hints:
                                    st.info(str(fc.learning_hints("mod_md_val_sds_thresDep")))
                            st.write("")
                            
                            # Download link for validation output
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            model_val_results["mean_ind"].transpose().to_excel(excel_file, sheet_name="thresh_independent_metrics_mean")
                            model_val_results["sd_ind"].transpose().to_excel(excel_file, sheet_name="thresh_independent_metrics_sd")
                            varImp_table_mean.to_excel(excel_file, sheet_name="variable_importance_mean")
                            varImp_table_sd.to_excel(excel_file, sheet_name="variable_importance_sd")
                            model_val_results["mean_dep"].transpose().to_excel(excel_file, sheet_name="thresh_dependent_metrics_mean")
                            model_val_results["sd_dep"].transpose().to_excel(excel_file, sheet_name="thresh_dependent_metrics_sd")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "Validation output__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download validation output</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("")   

                    else:
                        st.warning("Please run models!")
                    st.write("")
            
            #--------------------------------------------------------------------------------------
            # HYPERPARAMETER-TUNING OUTPUT

            if any(a for a in sb_ML_alg if a == "Random Forest") or any(a for a in sb_ML_alg if a == "Boosted Regression Trees") or any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                if do_hypTune == "Yes":
                    hype_title = "Hyperparameter-tuning output"
                if do_hypTune != "Yes":
                    hype_title = "Hyperparameter output"
                hype_output = st.expander(hype_title, expanded = False)
                with hype_output:
                    
                    # Random Forest
                    if any(a for a in sb_ML_alg if a == "Random Forest"):
                        st.markdown("**Random Forest**")

                        # Final hyperparameters
                        if rf_finalPara is not None:
                            st.write("Final hyperparameters:")
                            st.table(rf_finalPara.transpose())
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_hypeTune_RF_finPara")))
                            st.write("")
                        else:
                            st.warning("Please run models!")
                        
                        # Tuning details
                        if do_hypTune == "Yes":
                            if rf_tuning_results is not None and rf_finalPara is not None:
                                st.write("Tuning details:")
                                rf_finalTuneMetrics = pd.DataFrame(index = ["value"], columns = ["scoring metric", "number of models", "mean cv score", "standard deviation cv score", "test data score"])
                                rf_finalTuneMetrics["scoring metric"] = [rf_tuning_results.loc["value"]["scoring"]]
                                rf_finalTuneMetrics["number of models"] = [rf_tuning_results.loc["value"]["number of models"]]
                                rf_finalTuneMetrics["mean cv score"] = [rf_tuning_results.loc["value"]["mean score"]]
                                rf_finalTuneMetrics["standard deviation cv score"] = [rf_tuning_results.loc["value"]["std score"]]
                                rf_finalTuneMetrics["test data score"] = [rf_tuning_results.loc["value"]["test score"]]
                                st.table(rf_finalTuneMetrics.transpose().style.format(precision=user_precision))
                                if sett_hints:
                                    st.info(str(fc.learning_hints("mod_md_hypeTune_RF_details")))
                                st.write("")

                    # Boosted Regression Trees
                    if any(a for a in sb_ML_alg if a == "Boosted Regression Trees"):
                        st.markdown("**Boosted Regression Trees**")

                        # Final hyperparameters
                        if brt_finalPara is not None:
                            st.write("Final hyperparameters:")
                            st.table(brt_finalPara.transpose())
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_hypeTune_BRT_finPara")))
                            st.write("")
                        else:
                            st.warning("Please run models!")
                        
                        # Tuning details
                        if do_hypTune == "Yes":
                            if brt_tuning_results is not None and brt_finalPara is not None:
                                st.write("Tuning details:")
                                brt_finalTuneMetrics = pd.DataFrame(index = ["value"], columns = ["scoring metric", "number of models", "mean cv score", "standard deviation cv score", "test data score"])
                                brt_finalTuneMetrics["scoring metric"] = [brt_tuning_results.loc["value"]["scoring"]]
                                brt_finalTuneMetrics["number of models"] = [brt_tuning_results.loc["value"]["number of models"]]
                                brt_finalTuneMetrics["mean cv score"] = [brt_tuning_results.loc["value"]["mean score"]]
                                brt_finalTuneMetrics["standard deviation cv score"] = [brt_tuning_results.loc["value"]["std score"]]
                                brt_finalTuneMetrics["test data score"] = [brt_tuning_results.loc["value"]["test score"]]
                                st.table(brt_finalTuneMetrics.transpose().style.format(precision=user_precision))
                                if sett_hints:
                                    st.info(str(fc.learning_hints("mod_md_hypeTune_BRT_details")))
                                st.write("")

                    # Artificial Neural Networks
                    if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                        st.markdown("**Artificial Neural Networks**")
                        
                        # Final hyperparameters
                        if ann_finalPara is not None:
                            st.write("Final hyperparameters:")
                            st.table(ann_finalPara.transpose().style.format({"L² regularization": "{:.5}"}))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_hypeTune_ANN_finPara")))
                            st.write("")
                        else:
                            st.warning("Please run models!")
                        
                        # Tuning details
                        if do_hypTune == "Yes":
                            if ann_tuning_results is not None and ann_finalPara is not None:
                                st.write("Tuning details:")
                                ann_finalTuneMetrics = pd.DataFrame(index = ["value"], columns = ["scoring metric", "number of models", "mean cv score", "standard deviation cv score", "test data score"])
                                ann_finalTuneMetrics["scoring metric"] = [ann_tuning_results.loc["value"]["scoring"]]
                                ann_finalTuneMetrics["number of models"] = [ann_tuning_results.loc["value"]["number of models"]]
                                ann_finalTuneMetrics["mean cv score"] = [ann_tuning_results.loc["value"]["mean score"]]
                                ann_finalTuneMetrics["standard deviation cv score"] = [ann_tuning_results.loc["value"]["std score"]]
                                ann_finalTuneMetrics["test data score"] = [ann_tuning_results.loc["value"]["test score"]]
                                st.table(ann_finalTuneMetrics.transpose().style.format(precision=user_precision))
                                if sett_hints:
                                    st.info(str(fc.learning_hints("mod_md_hypeTune_ANN_details")))
                                st.write("")

                    # Download link for hyperparameter output
                    output = BytesIO()
                    excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                    if any(a for a in sb_ML_alg if a == "Random Forest"):
                        rf_finalPara.to_excel(excel_file, sheet_name="RF_final_hyperparameters")
                        if do_hypTune == "Yes":
                            rf_finalTuneMetrics.to_excel(excel_file, sheet_name="RF_tuning_details")
                    if any(a for a in sb_ML_alg if a == "Boosted Regression Trees"):
                        brt_finalPara.to_excel(excel_file, sheet_name="BRT_final_hyperparameters")
                        if do_hypTune == "Yes":
                            brt_finalTuneMetrics.to_excel(excel_file, sheet_name="BRT_tuning_details")
                    if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                        ann_finalPara.to_excel(excel_file, sheet_name="ANN_final_hyperparameters")
                        if do_hypTune == "Yes":
                            ann_finalTuneMetrics.to_excel(excel_file, sheet_name="ANN_tuning_details")
                    excel_file.close()
                    excel_file = output.getvalue()
                    b64 = base64.b64encode(excel_file)
                    if do_hypTune == "Yes":
                        dl_file_name = "Hyperparameter-tuning output__" + df_name + ".xlsx"
                        st.markdown(
                            f"""
                        <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download hyperparameter-tuning output</a>
                        """,
                        unsafe_allow_html=True)
                    if do_hypTune != "Yes":
                        dl_file_name = "Hyperparameter output__" + df_name + ".xlsx"
                        st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download hyperparameter output</a>
                            """,
                            unsafe_allow_html=True)
                        st.write("")

    #------------------------------------------------------------------------------------------

    # MULTI-CLASS CLASSIFICATION
    
    if analysis_type == "Multi-class classification":

        #++++++++++++++++++++++++++++++++++++++++++++
        # MACHINE LEARNING (PREDICTIVE DATA ANALYSIS)

        st.write("")
        st.write("")
        
        data_machinelearning_container2 = st.container()
        with data_machinelearning_container2:
            st.header("**Multi-class classification**")
            st.markdown("Go for creating predictive models of your data using machine learning techniques!  STATY will take care of the modelling for you, so you can put your focus on results interpretation and communication! ")

            ml_settings = st.expander("Specify models ", expanded = False)
            with ml_settings:
                
                # Initial status for running models (same as for regression, bc same functions are used)
                run_models = False
                sb_ML_alg = "NA"
                do_hypTune = "No"
                do_modval = "No"
                do_hypTune_no = "No hyperparameter tuning"
                final_hyPara_values="None"
                model_val_results = None
                model_full_results = None
                gam_finalPara = None
                brt_finalPara = None
                brt_tuning_results = None
                rf_finalPara = None
                rf_tuning_results = None
                ann_finalPara = None
                ann_tuning_results = None
                MLR_intercept = None
                MLR_cov_type = None
                MLR_finalPara = None
                MLR_model = "OLS"
                LR_cov_type = None
                LR_finalPara = None
                LR_finalPara = None

                if df.shape[1] > 0 and df.shape[0] > 0:
                    #--------------------------------------------------------------------------------------
                    # GENERAL SETTINGS
                
                    st.markdown("**Variable selection**")
                    
                    # Variable categories
                    df_summary_model = fc.data_summary(df)
                    var_cat = df_summary_model["Variable types"].loc["category"]
                    
                    # Response variable
                    response_var_type = "multi-class"
                    response_var_options = df.columns
                    response_var = st.selectbox("Select response variable", response_var_options, on_change=in_wid_change)
                    
                    # Check how many classes the response variable has (max: 10 classes)
                    if len(pd.unique(df[response_var])) > 10:
                        st.error("ERROR: Your response variable has more than 10 classes. Please select a variable with less classes!")
                        return
                    
                    # Check if response variable is numeric and has no NAs
                    response_var_message_num = False
                    response_var_message_na = False
                    response_var_message_cat = False

                    if var_cat.loc[response_var] == "string/binary" or var_cat.loc[response_var] == "bool/binary":
                        response_var_message_num = "ERROR: Please select a numeric multi-class response variable!"
                    elif var_cat.loc[response_var] == "string/categorical" or var_cat.loc[response_var] == "other" or var_cat.loc[response_var] == "string/single":
                        response_var_message_num = "ERROR: Please select a numeric multi-class response variable!"
                    elif var_cat.loc[response_var] == "numeric" and df[response_var].dtypes == "float64":
                        response_var_message_num = "ERROR: Please select a multi-class response variable!"
                    elif var_cat.loc[response_var] == "binary":
                        response_var_message_num = "ERROR: Please select a multi-class response variable!"
                    elif var_cat.loc[response_var] == "numeric" and df[response_var].dtypes == "int64":
                        response_var_message_cat = "WARNING: Please check whether your response variable is indeed a multi-class variable!"

                    if response_var_message_num != False:
                        st.error(response_var_message_num)
                    if response_var_message_cat != False:
                        st.warning(response_var_message_cat)

                    # Continue if everything is clean for response variable
                    if response_var_message_num == False and response_var_message_na == False:
                        # Select explanatory variables
                        expl_var_options = df.columns
                        expl_var_options = expl_var_options[expl_var_options.isin(df.drop(response_var, axis = 1).columns)]
                        expl_var = st.multiselect("Select explanatory variables", expl_var_options, on_change=in_wid_change)
                        var_list = list([response_var]) + list(expl_var)

                        # Check if explanatory variables are numeric
                        expl_var_message_num = False
                        expl_var_message_na = False
                        if any(a for a in df[expl_var].dtypes if a != "float64" and a != "float32" and a != "int64" and a != "int64" and a != "bool" and a != "int32"): 
                            expl_var_not_num = df[expl_var].select_dtypes(exclude=["int64", "int32", "float64", "float32","bool"]).columns
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

                            st.markdown("**Specify modelling algorithms**")

                            # Select algorithms 
                            algorithms = ["Random Forest", "Artificial Neural Networks"]

                            alg_list = list(algorithms)
                            sb_ML_alg = st.multiselect("Select modelling techniques", alg_list, alg_list)

                            st.markdown("**Model-specific settings**")

                            # Logistic Regression settings
                            # if any(a for a in sb_ML_alg if a == "Logistic Regression"):
                            #     LR_finalPara = pd.DataFrame(index = ["value"], columns = ["intercept", "covType"])
                            #     LR_intercept = "Yes"
                            #     LR_cov_type = "non-robust"
                            #     LR_finalPara["intercept"] = LR_intercept
                            #     LR_finalPara["covType"] = LR_cov_type
                            #     if st.checkbox("Adjust settings for Logistic Regression"):
                            #         col1, col2 = st.columns(2)
                            #         with col1:
                            #             LR_intercept = st.selectbox("Include intercept   ", ["Yes", "No"])
                            #         with col2:
                            #             LR_cov_type = st.selectbox("Covariance type", ["non-robust", "HC0"])
                            #         LR_finalPara["intercept"] = LR_intercept
                            #         LR_finalPara["covType"] = LR_cov_type
                            #         st.write("") 

                            # Save hyperparameter values for machine learning methods
                            final_hyPara_values = {}

                            # Random Forest settings
                            if any(a for a in sb_ML_alg if a == "Random Forest"):
                                rf_finalPara = pd.DataFrame(index = ["value"], columns = ["number of trees", "maximum tree depth", "maximum number of features", "sample rate"])
                                rf_finalPara["number of trees"] = [100]
                                rf_finalPara["maximum tree depth"] = [None]
                                rf_finalPara["maximum number of features"] = [len(expl_var)]
                                rf_finalPara["sample rate"] = [0.99]
                                final_hyPara_values["rf"] = rf_finalPara
                                if st.checkbox("Adjust settings for Random Forest ", on_change=in_wid_change):  
                                    col1, col2 = st.columns(2)
                                    col3, col4 = st.columns(2)
                                    with col1:
                                        rf_finalPara["number of trees"] = st.number_input("Number of trees", value=100, step=1, min_value=1, on_change=in_wid_change) 
                                    with col3:
                                        rf_mtd_sel = st.selectbox("Specify maximum tree depth ", ["No", "Yes"], on_change=in_wid_change)
                                        if rf_mtd_sel == "No":
                                            rf_finalPara["maximum tree depth"] = [None]
                                        if rf_mtd_sel == "Yes":
                                            rf_finalPara["maximum tree depth"] = st.slider("Maximum tree depth ", value=20, step=1, min_value=1, max_value=50, on_change=in_wid_change)
                                    if len(expl_var) >1:
                                        with col4:
                                            rf_finalPara["maximum number of features"] = st.slider("Maximum number of features ", value=len(expl_var), step=1, min_value=1, max_value=len(expl_var), on_change=in_wid_change)
                                        with col2:
                                            rf_finalPara["sample rate"] = st.slider("Sample rate ", value=0.99, step=0.01, min_value=0.5, max_value=0.99, on_change=in_wid_change)
                                    else:
                                        with col2:
                                            rf_finalPara["sample rate"] = st.slider("Sample rate ", value=0.99, step=0.01, min_value=0.5, max_value=0.99, on_change=in_wid_change)
                                    final_hyPara_values["rf"] = rf_finalPara 
                                    st.write("") 

                            # Artificial Neural Networks settings 
                            if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                                ann_finalPara = pd.DataFrame(index = ["value"], columns = ["weight optimization solver", "maximum number of iterations", "activation function", "hidden layer sizes", "learning rate", "L² regularization"])
                                ann_finalPara["weight optimization solver"] = ["adam"]
                                ann_finalPara["maximum number of iterations"] = [200]
                                ann_finalPara["activation function"] = ["relu"]
                                ann_finalPara["hidden layer sizes"] = [(100,)]
                                ann_finalPara["learning rate"] = [0.001]
                                ann_finalPara["L² regularization"] = [0.0001]
                                final_hyPara_values["ann"] = ann_finalPara
                                if st.checkbox("Adjust settings for Artificial Neural Networks "): 
                                    col1, col2 = st.columns(2)
                                    col3, col4 = st.columns(2)
                                    col5, col6 = st.columns(2)
                                    with col1:
                                        ann_finalPara["weight optimization solver"] = st.selectbox("Weight optimization solver ", ["adam"], on_change=in_wid_change)
                                    with col2:
                                        ann_finalPara["activation function"] = st.selectbox("Activation function ", ["relu", "identity", "logistic", "tanh"], on_change=in_wid_change)
                                    with col3:
                                        ann_finalPara["maximum number of iterations"] = st.slider("Maximum number of iterations ", value=200, step=1, min_value=10, max_value=1000, on_change=in_wid_change) 
                                    with col4:
                                        ann_finalPara["learning rate"] = st.slider("Learning rate  ", min_value=0.0001, max_value=0.01, value=0.001, step=1e-4, format="%.4f", on_change=in_wid_change)
                                    with col5:
                                        number_hidden_layers = st.selectbox("Number of hidden layers", [1, 2, 3], on_change=in_wid_change)
                                        if number_hidden_layers == 1:
                                            number_nodes1 = st.slider("Number of nodes in hidden layer", 5, 500, 100, on_change=in_wid_change)
                                            ann_finalPara["hidden layer sizes"] = [(number_nodes1,)]
                                        if number_hidden_layers == 2:
                                            number_nodes1 = st.slider("Number of neurons in first hidden layer", 5, 500, 100, on_change=in_wid_change)
                                            number_nodes2 = st.slider("Number of neurons in second hidden layer", 5, 500, 100, on_change=in_wid_change)
                                            ann_finalPara["hidden layer sizes"] = [(number_nodes1,number_nodes2,)]
                                        if number_hidden_layers == 3:
                                            number_nodes1 = st.slider("Number of neurons in first hidden layer", 5, 500, 100, on_change=in_wid_change)
                                            number_nodes2 = st.slider("Number of neurons in second hidden layer", 5, 500, 100, on_change=in_wid_change)
                                            number_nodes3 = st.slider("Number of neurons in third hidden layer", 5, 500, 100, on_change=in_wid_change)
                                            ann_finalPara["hidden layer sizes"] = [(number_nodes1,number_nodes2,number_nodes3,)]
                                    with col6:
                                        ann_finalPara["L² regularization"] = st.slider("L² regularization  ", min_value=0.00001, max_value=0.001, value=0.0001, step=1e-5, format="%.5f", on_change=in_wid_change)                                

                            #--------------------------------------------------------------------------------------
                            # HYPERPARAMETER TUNING SETTINGS
                            
                            if len(sb_ML_alg) >= 1:

                                # Depending on algorithm selection different hyperparameter settings are shown
                                if any(a for a in sb_ML_alg if a == "Random Forest") or any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                                    # General settings
                                    st.markdown("**Hyperparameter-tuning settings**")
                                    do_hypTune = st.selectbox("Use hyperparameter-tuning", ["No", "Yes"], on_change=in_wid_change)
                                
                                    # Save hyperparameter values for all algorithms
                                    hyPara_values = {}
                                    
                                    # No hyperparameter-tuning
                                    if do_hypTune == "No":
                                        do_hypTune_no = "Default hyperparameter values are used!"

                                    # Hyperparameter-tuning 
                                    elif do_hypTune == "Yes":
                                        st.warning("WARNING: Hyperparameter-tuning can take a lot of time! For tips, please [contact us](mailto:staty@quant-works.de?subject=Staty-App).")
                                        
                                        # Further general settings
                                        hypTune_method = st.selectbox("Hyperparameter-search method", ["random grid-search", "grid-search"], on_change=in_wid_change)
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            hypTune_nCV = st.slider("Select number for n-fold cross-validation", 2, 10, 5, on_change=in_wid_change)

                                        if hypTune_method == "random grid-search" or hypTune_method == "Bayes optimization" or hypTune_method == "sequential model-based optimization":
                                            with col2:
                                                hypTune_iter = st.slider("Select number of iterations for search", 20, 1000, 20, on_change=in_wid_change)
                                        else:
                                            hypTune_iter = False

                                        st.markdown("**Model-specific tuning settings**")
                                        # Random Forest settings
                                        if any(a for a in sb_ML_alg if a == "Random Forest"):
                                            rf_tunePara = pd.DataFrame(index = ["min", "max"], columns = ["number of trees", "maximum tree depth", "maximum number of features", "sample rate"])
                                            rf_tunePara["number of trees"] = [50, 500]
                                            rf_tunePara["maximum tree depth"] = [None, None]
                                            rf_tunePara["maximum number of features"] = [1, len(expl_var)]
                                            rf_tunePara["sample rate"] = [0.8, 0.99]
                                            hyPara_values["rf"] = rf_tunePara
                                            if st.checkbox("Adjust tuning settings for Random Forest"):
                                                col1, col2 = st.columns(2)
                                                col3, col4 = st.columns(2)
                                                with col1:
                                                    rf_tunePara["number of trees"] = st.slider("Range for number of trees ", 50, 1000, [50, 500], on_change=in_wid_change)
                                                with col3:
                                                    rf_mtd_choice = st.selectbox("Specify maximum tree depth", ["No", "Yes"], on_change=in_wid_change)
                                                    if rf_mtd_choice == "Yes":
                                                        rf_tunePara["maximum tree depth"] = st.slider("Range for maximum tree depth ", 1, 50, [2, 10], on_change=in_wid_change)
                                                    else:
                                                        rf_tunePara["maximum tree depth"] = [None, None]
                                                with col4:
                                                    if len(expl_var) > 1:
                                                        rf_tunePara["maximum number of features"] = st.slider("Range for maximum number of features", 1, len(expl_var), [1, len(expl_var)], on_change=in_wid_change)
                                                    else:
                                                        rf_tunePara["maximum number of features"] = [1,1]
                                                with col2:
                                                    rf_tunePara["sample rate"] = st.slider("Range for sample rate ", 0.5, 0.99, [0.8, 0.99], on_change=in_wid_change)
                                                hyPara_values["rf"] = rf_tunePara

                                        # Artificial Neural Networks settings
                                        if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                                            ann_tunePara = pd.DataFrame(index = ["min", "max"], columns = ["weight optimization solver", "maximum number of iterations", "activation function", "number of hidden layers", "nodes per hidden layer", "learning rate","L² regularization"])# "learning rate schedule", "momentum", "epsilon"])
                                            ann_tunePara["weight optimization solver"] = list([["adam"], "NA"])
                                            ann_tunePara["maximum number of iterations"] = [100, 200]
                                            ann_tunePara["activation function"] = list([["relu"], "NA"])
                                            ann_tunePara["number of hidden layers"] = list([1, "NA"])
                                            ann_tunePara["nodes per hidden layer"] = [50, 100]
                                            ann_tunePara["learning rate"] = [0.0001, 0.002]
                                            ann_tunePara["L² regularization"] = [0.00001, 0.0002]
                                            hyPara_values["ann"] = ann_tunePara
                                            if st.checkbox("Adjust tuning settings for Artificial Neural Networks", on_change=in_wid_change):
                                                col1, col2 = st.columns(2)
                                                col3, col4 = st.columns(2)
                                                col5, col6 = st.columns(2)
                                                with col1:
                                                    weight_opt_list = st.selectbox("Weight optimization solver  ", ["adam"], on_change=in_wid_change)
                                                    if len(weight_opt_list) == 0:
                                                        weight_opt_list = ["adam"]
                                                        st.warning("WARNING: Default value used 'adam'")
                                                    ann_tunePara["weight optimization solver"] = list([[weight_opt_list], "NA"])
                                                with col2:
                                                    ann_tunePara["maximum number of iterations"] = st.slider("Maximum number of iterations (epochs) ", 10, 1000, [100, 200], on_change=in_wid_change)
                                                with col3:
                                                    act_func_list = st.multiselect("Activation function ", ["identity", "logistic", "tanh", "relu"], ["relu"], on_change=in_wid_change)
                                                    if len(act_func_list) == 0:
                                                        act_func_list = ["relu"]
                                                        st.warning("WARNING: Default value used 'relu'")
                                                    ann_tunePara["activation function"] = list([act_func_list, "NA"])
                                                with col5:
                                                    number_hidden_layers = st.selectbox("Number of hidden layers ", [1, 2, 3], on_change=in_wid_change)
                                                    ann_tunePara["number of hidden layers"]  = list([number_hidden_layers, "NA"])
                                                    # Cases for hidden layers
                                                    if number_hidden_layers == 1:
                                                        ann_tunePara["nodes per hidden layer"] = st.slider("Number of nodes in hidden layer ", 5, 500, [50, 100], on_change=in_wid_change)
                                                    if number_hidden_layers == 2:
                                                        number_nodes1 = st.slider("Number of neurons in first hidden layer ", 5, 500, [50, 100], on_change=in_wid_change)
                                                        number_nodes2 = st.slider("Number of neurons in second hidden layer ", 5, 500, [50, 100], on_change=in_wid_change)
                                                        min_nodes = list([number_nodes1[0], number_nodes2[0]])
                                                        max_nodes = list([number_nodes1[1], number_nodes2[1]])
                                                        ann_tunePara["nodes per hidden layer"] = list([min_nodes, max_nodes])
                                                    if number_hidden_layers == 3:
                                                        number_nodes1 = st.slider("Number of neurons in first hidden layer ", 5, 500, [50, 100], on_change=in_wid_change)
                                                        number_nodes2 = st.slider("Number of neurons in second hidden layer ", 5, 500, [50, 100], on_change=in_wid_change)
                                                        number_nodes3 = st.slider("Number of neurons in third hidden layer ", 5, 500, [50, 100], on_change=in_wid_change)
                                                        min_nodes = list([number_nodes1[0], number_nodes2[0], number_nodes3[0]])
                                                        max_nodes = list([number_nodes1[1], number_nodes2[1], number_nodes3[1]])
                                                        ann_tunePara["nodes per hidden layer"] = list([min_nodes, max_nodes])
                                                with col6:
                                                    if weight_opt_list == "adam": 
                                                        ann_tunePara["learning rate"] = st.slider("Range for learning rate ", 0.0001, 0.01, [0.0001, 0.002], step=1e-4, format="%.4f", on_change=in_wid_change)
                                                with col4:
                                                    ann_tunePara["L² regularization"] = st.slider("L² regularization parameter ", 0.0, 0.001, [0.00001, 0.0002], step=1e-5, format="%.5f", on_change=in_wid_change)
                                                hyPara_values["ann"] = ann_tunePara
                                        
                                #--------------------------------------------------------------------------------------
                                # VALIDATION SETTINGS

                                st.markdown("**Validation settings**")
                                do_modval= st.selectbox("Use model validation", ["No", "Yes"], on_change=in_wid_change)

                                if do_modval == "Yes":
                                    col1, col2 = st.columns(2)
                                    # Select training/ test ratio
                                    with col1: 
                                        train_frac = st.slider("Select training data size", 0.5, 0.95, 0.8, on_change=in_wid_change)

                                    # Select number for validation runs
                                    with col2:
                                        val_runs = st.slider("Select number for validation runs", 5, 100, 10, on_change=in_wid_change)

                                #--------------------------------------------------------------------------------------
                                # PREDICTION SETTINGS

                                st.markdown("**Model predictions**")
                                do_modprednew = st.selectbox("Use model prediction for new data", ["No", "Yes"], on_change=in_wid_change)

                                if do_modprednew == "Yes":
                                    # Upload new data
                                    new_data_pred = st.file_uploader("  ", type=["csv", "txt"], on_change=in_wid_change)

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
                                        
                                        # Check if explanatory variables are available as columns
                                        expl_list = []
                                        for expl_incl in expl_var:
                                            if expl_incl not in df_new.columns:
                                                expl_list.append(expl_incl)
                                        if expl_list:
                                            st.error("ERROR: Some variables are missing in new data: "+ ', '.join(expl_list))
                                            return
                                        else:
                                            st.info("All variables are available for predictions!")
                                        
                                        # Check if NAs are present and delete them automatically
                                        if df_new.iloc[list(pd.unique(np.where(df_new.isnull())[0]))].shape[0] == 0:
                                            st.empty()
                                        else:
                                            df_new = df_new[expl_var].dropna()
                                            st.warning("WARNING: Your new data set includes NAs. Rows with NAs are automatically deleted!")
                                        df_new = df_new[expl_var]
                                    
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
                                    excel_file.close()
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
                                            excel_file.close()
                                            excel_file = output.getvalue()
                                            b64 = base64.b64encode(excel_file)
                                            dl_file_name= "New data for predictions__" + df_name + ".xlsx"
                                            st.markdown(
                                                f"""
                                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download new data for predictions</a>
                                            """,
                                            unsafe_allow_html=True)
                                        st.write("")

                                # Show machine learning summary
                                if st.checkbox('Show a summary of machine learning settings', value = False): 
                                    
                                    #--------------------------------------------------------------------------------------
                                    # ALGORITHMS
                                    
                                    st.write("Algorithms summary:")
                                    st.write("- Models:",  ', '.join(sb_ML_alg))
                                    # if any(a for a in sb_ML_alg if a == "Logistic Regression"):
                                    #     st.write("- Logistic Regression including intercept: ", LR_intercept)
                                    #     st.write("- Logistic Regression covariance type: ", LR_cov_type)
                                    if any(a for a in sb_ML_alg if a == "Random Forest") and do_hypTune == "No":
                                        st.write("- Random Forest parameters: ")
                                        st.write(rf_finalPara)
                                    if any(a for a in sb_ML_alg if a == "Artificial Neural Networks") and do_hypTune == "No":
                                        st.write("- Artificial Neural Networks parameters: ")
                                        st.write(ann_finalPara)
                                    st.write("")

                                    #--------------------------------------------------------------------------------------
                                    # SETTINGS

                                    # Hyperparameter settings summary
                                    if any(a for a in sb_ML_alg if a == "Artificial Neural Networks" or a == "Boosted Regression Trees" or a == "Random Forest"):
                                        st.write("Hyperparameter-tuning settings summary:")
                                        if do_hypTune == "No":
                                            st.write("- ", do_hypTune_no)
                                            st.write("")
                                        if do_hypTune == "Yes":
                                            st.write("- Search method:", hypTune_method)
                                            st.write("- ", hypTune_nCV, "-fold cross-validation")
                                            if hypTune_method == "random grid-search" or hypTune_method == "Bayes optimization" or hypTune_method == "sequential model-based optimization":
                                                st.write("- ", hypTune_iter, "iterations in search")
                                                st.write("")
                                            # Random Forest summary
                                            if any(a for a in sb_ML_alg if a == "Random Forest"):
                                                st.write("Random Forest tuning settings summary:")
                                                st.write(rf_tunePara)
                                            # Artificial Neural Networks summary
                                            if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                                                st.write("Artificial Neural Networks tuning settings summary:")
                                                st.write(ann_tunePara.style.format({"L² regularization": "{:.5}"}))
                                                #st.caption("** Learning rate is only used in adam")
                                            st.write("")

                                    # General settings summary
                                    st.write("General settings summary:")
                                    st.write("- Response variable type: ", response_var_type)
                                    # Modelling formula
                                    if expl_var != False:
                                        st.write("- Modelling formula:", response_var, "~",  ' + '.join(expl_var))
                                    if do_modval == "Yes":
                                        # Train/ test ratio
                                        if train_frac != False:
                                            st.write("- Train/ test ratio:", str(round(train_frac*100)), "% / ", str(round(100-train_frac*100)), "%")
                                        # Validation runs
                                        if val_runs != False:
                                            st.write("- Validation runs:", str(val_runs))
                                    st.write("")
                            
                                #--------------------------------------------------------------------------------------
                                # RUN MODELS

                                # Models are run on button click
                                st.write("")
                                run_models = st.button("Run models")
                                st.write("")
                                
                                if run_models:

                                    # Check if new data available
                                    if do_modprednew == "Yes":
                                        if new_data_pred is None:
                                            st.error("ERROR: Please upload new data for additional model predictions or select 'No'!")
                                            return

                                    #Hyperparameter   
                                    if do_hypTune == "Yes":

                                        # Tuning
                                        model_tuning_results = ml.model_tuning(df, sb_ML_alg, hypTune_method, hypTune_iter, hypTune_nCV, hyPara_values, response_var_type, response_var, expl_var)
                                    
                                        # Save final hyperparameters
                                        # Random Forest
                                        if any(a for a in sb_ML_alg if a == "Random Forest"):
                                            rf_tuning_results = model_tuning_results["rf tuning"]
                                            rf_finalPara = pd.DataFrame(index = ["value"], columns = ["number of trees", "maximum tree depth", "maximum number of features", "sample rate"])
                                            rf_finalPara["number of trees"] = [rf_tuning_results.loc["value"]["number of trees"]]
                                            if [rf_tuning_results.loc["value"]["maximum tree depth"]][0] == "None":
                                                rf_finalPara["maximum tree depth"] = None
                                            else:
                                                rf_finalPara["maximum tree depth"] = [rf_tuning_results.loc["value"]["maximum tree depth"]]
                                            rf_finalPara["maximum number of features"] = [rf_tuning_results.loc["value"]["maximum number of features"]]
                                            rf_finalPara["sample rate"] = [rf_tuning_results.loc["value"]["sample rate"]]
                                            final_hyPara_values["rf"] = rf_finalPara
                                        # Artificial Neural Networks
                                        if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                                            ann_tuning_results = model_tuning_results["ann tuning"]
                                            ann_finalPara = pd.DataFrame(index = ["value"], columns = ["weight optimization solver", "maximum number of iterations", "activation function", "hidden layer sizes", "learning rate", "L² regularization"]) #"learning rate schedule", "momentum", "epsilon"])
                                            ann_finalPara["weight optimization solver"] = [ann_tuning_results.loc["value"]["weight optimization solver"]]
                                            ann_finalPara["maximum number of iterations"] = [ann_tuning_results.loc["value"]["maximum number of iterations"]]
                                            ann_finalPara["activation function"] = [ann_tuning_results.loc["value"]["activation function"]]
                                            ann_finalPara["hidden layer sizes"] = [ann_tuning_results.loc["value"]["hidden layer sizes"]]
                                            ann_finalPara["learning rate"] = [ann_tuning_results.loc["value"]["learning rate"]]
                                            #ann_finalPara["learning rate schedule"] = [ann_tuning_results.loc["value"]["learning rate schedule"]]
                                            #ann_finalPara["momentum"] = [ann_tuning_results.loc["value"]["momentum"]]
                                            ann_finalPara["L² regularization"] = [ann_tuning_results.loc["value"]["L² regularization"]]
                                            #ann_finalPara["epsilon"] = [ann_tuning_results.loc["value"]["epsilon"]]
                                            final_hyPara_values["ann"] = ann_finalPara
                                        
                                    # Model validation
                                    if do_modval == "Yes":
                                        model_val_results = ml.model_val(df, sb_ML_alg, MLR_model, train_frac, val_runs, response_var_type, response_var, expl_var, final_hyPara_values, gam_finalPara, MLR_finalPara, LR_finalPara)
                                    
                                    # Full model (depending on prediction for new data)
                                    if do_modprednew == "Yes":
                                        if new_data_pred is not None:
                                            if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                                                model_full_results, full_model_ann_sk = ml.model_full(df, df_new, sb_ML_alg, MLR_model, MLR_finalPara, LR_finalPara, response_var_type, response_var, expl_var, final_hyPara_values, gam_finalPara)
                                            else:
                                                model_full_results = ml.model_full(df, df_new, sb_ML_alg, MLR_model, MLR_finalPara, LR_finalPara, response_var_type, response_var, expl_var, final_hyPara_values, gam_finalPara)
                                    if do_modprednew == "No":
                                        df_new = pd.DataFrame()
                                        if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                                            model_full_results, full_model_ann_sk = ml.model_full(df, df_new, sb_ML_alg, MLR_model, MLR_finalPara, LR_finalPara, response_var_type, response_var, expl_var, final_hyPara_values, gam_finalPara)
                                        else:
                                            model_full_results = ml.model_full(df, df_new, sb_ML_alg, MLR_model, MLR_finalPara, LR_finalPara, response_var_type, response_var, expl_var, final_hyPara_values, gam_finalPara)
                                    # Success message
                                    st.success('Models run successfully!')
                                    if do_hypTune == "Yes":
                                        st.session_state['model_tuning_results'] = model_tuning_results  
                                    st.session_state['model_full_results'] = model_full_results 
                                    if do_modval == "Yes":
                                        st.session_state['model_val_results'] = model_val_results  
                                    if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                                        st.session_state['full_model_ann_sk'] = full_model_ann_sk
                                        st.session_state['ann_finalPara'] = ann_finalPara
                                        if do_hypTune == "Yes":
                                            st.session_state['ann_tuning_results'] = ann_tuning_results
                else: st.error("ERROR: No data available for Modelling!") 

        #++++++++++++++++++++++
        # ML OUTPUT

        # Show only if models were run
        if st.session_state['model_full_results'] is not None and 'expl_var' in locals():

            model_tuning_results = st.session_state['model_tuning_results']
            model_full_results = st.session_state['model_full_results']
            model_val_results = st.session_state['model_val_results']
            if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                full_model_ann_sk = st.session_state['full_model_ann_sk']
                ann_finalPara = st.session_state['ann_finalPara']
                ann_tuning_results = st.session_state['ann_tuning_results']

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
                    corr_plot1 = (corr_plot + text).properties(width = 400, height = 400)
                    correlation_plot = correlation_plot.properties(padding = {"left": 50, "top": 5, "right": 5, "bottom": 50})
                    # hist_2d_plot = scat_plot.properties(height = 350)
                    st.altair_chart(correlation_plot, use_container_width = True)
                    if sett_hints:
                        st.info(str(fc.learning_hints("mod_cor")))
                    st.write("")

                    #-------------------------------------------------------------

                    # Multi-class response variable
                    if response_var_type == "multi-class":

                        # RF specific output
                        if any(a for a in sb_ML_alg if a == "Random Forest"):
                            st.markdown("**Random Forest**")
                            
                            fm_rf_reg_col1, fm_rf_reg_col2 = st.columns(2)
                            # Regression information
                            with fm_rf_reg_col1:
                                st.write("Regression information:")
                                st.table(model_full_results["RF information"].style.format(precision=user_precision))
                            # Regression statistics
                            with fm_rf_reg_col2:
                                st.write("Regression statistics:")
                                rf_error_est = pd.DataFrame(index = ["ACC", "BAL ACC"], columns = ["Value"])
                                rf_error_est.loc["ACC"] = model_full_results["model comparison"].loc["ACC"]["Random Forest"]
                                rf_error_est.loc["BAL ACC"] = model_full_results["model comparison"].loc["BAL ACC"]["Random Forest"]
                                st.table(rf_error_est.style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_RF_regStat_mult"))) 
                            st.write("")
                            fm_rf_figs1_col1, fm_rf_figs1_col2 = st.columns(2)
                            # Variable importance (via permutation)
                            with fm_rf_figs1_col1:
                                st.write("Variable importance (via permutation):")
                                rf_varImp_table = model_full_results["RF variable importance"]
                                st.table(rf_varImp_table.style.format(precision=user_precision))
                                st.write("")
                            with fm_rf_figs1_col2:
                                st.write("")
                                st.write("")
                                st.write("")
                                rf_varImp_plot_data = model_full_results["RF variable importance"]
                                rf_varImp_plot_data["Variable"] = rf_varImp_plot_data.index
                                rf_varImp = alt.Chart(rf_varImp_plot_data, height = 200).mark_bar().encode(
                                    x = alt.X("mean", title = "variable importance", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Variable", title = None, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), sort = None),
                                    tooltip = ["Variable", "mean"]
                                )
                                st.altair_chart(rf_varImp, use_container_width = True) 
                            st.write("") 
                            fm_rf_figs2_col1, fm_rf_figs2_col2 = st.columns(2)
                            # Feature importance
                            with fm_rf_figs2_col1:
                                st.write("Feature importance (impurity-based):")
                                rf_featImp_table = model_full_results["RF feature importance"]
                                st.table(rf_featImp_table.style.format(precision=user_precision))
                            with fm_rf_figs2_col2:
                                st.write("")
                                st.write("")
                                st.write("")
                                rf_featImp_plot_data = model_full_results["RF feature importance"]
                                rf_featImp_plot_data["Variable"] = rf_featImp_plot_data.index
                                rf_featImp = alt.Chart(rf_featImp_plot_data, height = 200).mark_bar().encode(
                                    x = alt.X("Value", title = "feature importance", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Variable", title = None, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), sort = None),
                                    tooltip = ["Variable", "Value"]
                                )
                                st.altair_chart(rf_featImp, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_RF_varImp_mult"))) 
                            st.write("") 
                            # Partial dependence plots
                            # st.write("Partial dependence plots:")    
                            # fm_rf_figs3_col1, fm_rf_figs3_col2 = st.columns(2)
                            # for pd_var in expl_var:
                            #     pd_data_rf = pd.DataFrame(columns = [pd_var])
                            #     pd_data_rf[pd_var] = model_full_results["RF partial dependence"][pd_var][1][0]
                            #     pd_data_rf["Partial dependence"] = model_full_results["RF partial dependence"][pd_var][0][0]
                            #     pd_chart_rf = alt.Chart(pd_data_rf, height = 200).mark_line(color = "darkred").encode(
                            #         x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            #         y = alt.Y("Partial dependence", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["RF partial dependence min/max"]["min"].min(), model_full_results["RF partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            #         tooltip = ["Partial dependence"] + [pd_var]
                            #     )
                            #     pd_data_ticks_rf = pd.DataFrame(columns = [pd_var])
                            #     pd_data_ticks_rf[pd_var] = df[pd_var]
                            #     pd_data_ticks_rf["y"] = [model_full_results["RF partial dependence min/max"]["min"].min()] * df.shape[0]
                            #     pd_ticks_rf = alt.Chart(pd_data_ticks_rf, height = 200).mark_tick(size = 5, thickness = 1).encode(
                            #         x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), scale = alt.Scale(domain = [pd_data_ticks_rf[pd_var].min(), pd_data_ticks_rf[pd_var].max()])),
                            #         y = alt.Y("y", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["RF partial dependence min/max"]["min"].min(), model_full_results["RF partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            #         tooltip = [pd_var]
                            #     )
                            #     if expl_var.index(pd_var)%2 == 0:
                            #         with fm_rf_figs3_col1:
                            #             st.altair_chart(pd_ticks_rf + pd_chart_rf, use_container_width = True)
                            #     if expl_var.index(pd_var)%2 == 1:
                            #         with fm_rf_figs3_col2:
                            #             st.altair_chart(pd_ticks_rf + pd_chart_rf, use_container_width = True)
                            # if sett_hints:
                            #     st.info(str(fc.learning_hints("mod_md_RF_partDep_bin")))
                            # Confusion matrix
                            st.write("Confusion matrix (columns correspond to predictions):")
                            st.table(model_full_results["RF confusion"])
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_RF_confu_mult"))) 
                            st.write("") 
                            # Classification report
                            st.write("Classification report:")
                            st.table(model_full_results["RF classification report"].style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_RF_classRep_mult"))) 
                            st.write("") 

                            # Download link for RF output
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            model_full_results["RF information"].to_excel(excel_file, sheet_name="classification_information")
                            rf_error_est.to_excel(excel_file, sheet_name="classification_statistics")
                            rf_varImp_table.to_excel(excel_file, sheet_name="variable_importance")
                            rf_featImp_table.to_excel(excel_file, sheet_name="feature_importance")
                            model_full_results["RF confusion"].to_excel(excel_file, sheet_name="confusion_matrix")
                            model_full_results["RF classification report"].to_excel(excel_file, sheet_name="classification_report")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "RF full model output__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download Random Forest full model output</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("")
            
                        # ANN specific output
                        if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                            st.markdown("**Artificial Neural Networks**")

                            fm_ann_reg_col1, fm_ann_reg_col2 = st.columns(2)
                            # Regression information
                            with fm_ann_reg_col1:
                                st.write("Regression information:")
                                st.table(model_full_results["ANN information"].style.format(precision=user_precision))
                            # Regression statistics
                            with fm_ann_reg_col2:
                                st.write("Regression statistics:")
                                ann_error_est = pd.DataFrame(index = ["ACC", "BAL ACC", "Best loss"], columns = ["Value"])
                                ann_error_est.loc["ACC"] = model_full_results["model comparison"].loc["ACC"]["Artificial Neural Networks"]
                                ann_error_est.loc["BAL ACC"] = model_full_results["model comparison"].loc["BAL ACC"]["Artificial Neural Networks"]
                                if ann_finalPara["weight optimization solver"][0] != "lbfgs":
                                    ann_error_est.loc["Best loss"] =  model_full_results["ANN loss"]
                                st.table(ann_error_est.style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_ANN_regStat_mult")))
                            st.write("")
                            # ANN architecture
                            st.write("Artificial Neural Network architecture:")
                            coef_list = full_model_ann_sk.coefs_
                            # weight matrix
                            annviz_weights = st.checkbox("Show weight matrix")
                            if annviz_weights:
                                layer_options = ["Input Layer <-> Hidden Layer 1"]
                                if int(model_full_results["ANN information"].loc['Layers']) == 3:
                                    layer_options += ["Hidden Layer 1 <-> Output Layer"]
                                elif int(model_full_results["ANN information"].loc['Layers']) == 4:
                                    layer_options += ["Hidden Layer 1 <-> Hidden Layer 2"]
                                    layer_options += ["Hidden Layer 2 <-> Output Layer"]
                                elif int(model_full_results["ANN information"].loc['Layers']) == 5:
                                    layer_options += ["Hidden Layer 1 <-> Hidden Layer 2"]
                                    layer_options += ["Hidden Layer 2 <-> Hidden Layer 3"]
                                    layer_options += ["Hidden Layer 3 <-> Output Layer"]
                                wei_matrix = st.selectbox('Weight matrix for following Layer', layer_options)
                                output = np.unique(df[response_var])
                                df_weights = ml.weight_matrix_func(output, expl_var, wei_matrix, coef_list)
                                st.write(df_weights)
                            annviz_output = st.checkbox("Show Artificial Neural Network Visualization")
                            if annviz_output:
                                st.write("Select which neurons of the hidden layer should be visualized:")
                                sli_col1, sli_col2 = st.columns(2)
                                # input layer
                                in_sel_nod = (1,len(expl_var))
                                # hidden layer 1
                                hi1_nod = int(ann_finalPara['hidden layer sizes'][0][0])
                                if hi1_nod >= 10:
                                    hi1_def_max = 10
                                else:
                                    hi1_def_max = hi1_nod
                                hi1_sel_nod = sli_col1.slider('Hidden Layer 1', min_value=1, max_value=hi1_nod, value=[1,hi1_def_max])
                                hi_sel_tup = (hi1_sel_nod[1]-hi1_sel_nod[0]+1,)
                                # hidden layer 2
                                if int(model_full_results["ANN information"].loc['Layers']) >= 4:
                                    hi2_nod = int(ann_finalPara['hidden layer sizes'][0][1])
                                    if hi2_nod >= 10:
                                        hi2_def_max = 10
                                    else:
                                        hi2_def_max = hi2_nod
                                    hi2_sel_nod = sli_col2.slider('Hidden Layer 2', min_value=1, max_value=hi2_nod, value=[1,hi2_def_max])
                                    hi_sel_tup += (hi2_sel_nod[1]-hi2_sel_nod[0]+1,)
                                # hidden layer 3
                                if int(model_full_results["ANN information"].loc['Layers']) >= 5:
                                    hi3_nod = int(ann_finalPara['hidden layer sizes'][0][2])
                                    if hi3_nod >= 10:
                                        hi3_def_max = 10
                                    else:
                                        hi3_def_max = hi3_nod
                                    hi3_sel_nod = sli_col1.slider('Hidden Layer 3', min_value=1, max_value=hi3_nod, value=[1,hi3_def_max])
                                    hi_sel_tup += (hi3_sel_nod[1]-hi3_sel_nod[0]+1,)
                                
                                # ANN Visualization
                                st.write("")
                                st.warning("Very large artificial neural networks cannot be visualized clearly. Recommendation: display max. 20 neurons in one layer.")
                                numb_output = len(np.unique(df[response_var]))
                                network_structure = np.hstack(([in_sel_nod[1]-in_sel_nod[0]+1], np.asarray(hi_sel_tup), [numb_output]))
                                
                                # seperate weight matrix
                                if int(model_full_results["ANN information"].loc['Layers']) == 3:
                                    in_hi_wei = coef_list[0]
                                    hi_out_wei = coef_list[1]
                                elif int(model_full_results["ANN information"].loc['Layers']) == 4:
                                    in_hi_wei = coef_list[0]
                                    hi1_hi2_wei = coef_list[1]
                                    hi_out_wei = coef_list[2]
                                elif int(model_full_results["ANN information"].loc['Layers']) == 5:
                                    in_hi_wei = coef_list[0]
                                    hi1_hi2_wei = coef_list[1]
                                    hi2_hi3_wei = coef_list[2]
                                    hi_out_wei = coef_list[3]

                                # weights for selected nodes
                                sel_in_hi_wei = in_hi_wei[in_sel_nod[0]-1:in_sel_nod[1], hi1_sel_nod[0]-1:hi1_sel_nod[1]]
                                sel_coef_list = [sel_in_hi_wei]
                                # 1 hidden layer
                                if int(model_full_results["ANN information"].loc['Layers']) == 3:
                                    sel_hi_out_wei = hi_out_wei[hi1_sel_nod[0]-1:hi1_sel_nod[1], 0:numb_output]
                                    sel_coef_list += [sel_hi_out_wei]
                                # 2 hidden layer
                                elif int(model_full_results["ANN information"].loc['Layers']) == 4:
                                    sel_hi1_hi2_wei = hi1_hi2_wei[hi1_sel_nod[0]-1:hi1_sel_nod[1], hi2_sel_nod[0]-1:hi2_sel_nod[1]]
                                    sel_hi_out_wei = hi_out_wei[hi2_sel_nod[0]-1:hi2_sel_nod[1], 0:numb_output]
                                    sel_coef_list += [sel_hi1_hi2_wei]
                                    sel_coef_list += [sel_hi_out_wei]
                                # 3 hidden layer
                                elif int(model_full_results["ANN information"].loc['Layers']) == 5:
                                    sel_hi1_hi2_wei = hi1_hi2_wei[hi1_sel_nod[0]-1:hi1_sel_nod[1], hi2_sel_nod[0]-1:hi2_sel_nod[1]]
                                    sel_hi2_hi3_wei = hi2_hi3_wei[hi2_sel_nod[0]-1:hi2_sel_nod[1], hi3_sel_nod[0]-1:hi3_sel_nod[1]]
                                    sel_hi_out_wei = hi_out_wei[hi3_sel_nod[0]-1:hi3_sel_nod[1], 0:numb_output]
                                    sel_coef_list += [sel_hi1_hi2_wei]
                                    sel_coef_list += [sel_hi2_hi3_wei]
                                    sel_coef_list += [sel_hi_out_wei]                                
                                
                                network=VisNN.DrawNN(network_structure, sel_coef_list)
                                st.set_option('deprecation.showPyplotGlobalUse', False)
                                st.write("")
                                st.pyplot(network.draw())
                            st.write("")
                            # Loss curve (loss vs. number of iterations (epochs))
                            if ann_finalPara["weight optimization solver"][0] != "lbfgs":
                                st.write("Loss curve:")
                                loss_curve = pd.DataFrame(index = range(len(model_full_results["ANN loss curve"])), columns = ["Loss"])
                                loss_curve["Loss"] = model_full_results["ANN loss curve"]
                                loss_curve["Iterations"] = loss_curve.index+1
                                loss_curve_plot = alt.Chart(loss_curve, height = 200).mark_line(color = "darkred").encode(
                                    x = alt.X("Iterations", title = "iterations (epochs)", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), scale = alt.Scale(domain = [loss_curve["Iterations"].min(), loss_curve["Iterations"].max()])),
                                    y = alt.Y("Loss", title = "loss", scale = alt.Scale(zero = False), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["Loss", "Iterations"]
                                )     
                                st.altair_chart(loss_curve_plot, use_container_width = True)    
                            st.write("") 
                            fm_ann_figs1_col1, fm_ann_figs1_col2 = st.columns(2)
                            # Variable importance (via permutation)
                            with fm_ann_figs1_col1:
                                st.write("Variable importance (via permutation):")
                                ann_varImp_table = model_full_results["ANN variable importance"]
                                st.table(ann_varImp_table.style.format(precision=user_precision))
                            with fm_ann_figs1_col2:
                                st.write("")
                                st.write("")
                                st.write("")
                                ann_varImp_plot_data = model_full_results["ANN variable importance"]
                                ann_varImp_plot_data["Variable"] = ann_varImp_plot_data.index
                                ann_varImp = alt.Chart(ann_varImp_plot_data, height = 200).mark_bar().encode(
                                    x = alt.X("mean", title = "variable importance", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("Variable", title = None, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), sort = None),
                                    tooltip = ["Variable", "mean"]
                                )
                                st.altair_chart(ann_varImp, use_container_width = True)
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_ANN_varImp_bin")))
                            st.write("")
                            # Partial dependence plots
                            # st.write("Partial dependence plots:")    
                            # fm_ann_figs2_col1, fm_ann_figs2_col2 = st.columns(2)
                            # for pd_var in expl_var:
                            #     pd_data_ann = pd.DataFrame(columns = [pd_var])
                            #     pd_data_ann[pd_var] = (model_full_results["ANN partial dependence"][pd_var][1][0]*(df[pd_var].std()))+df[pd_var].mean()
                            #     pd_data_ann["Partial dependence"] = model_full_results["ANN partial dependence"][pd_var][0][0]
                            #     pd_chart_ann = alt.Chart(pd_data_ann, height = 200).mark_line(color = "darkred").encode(
                            #     x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            #     y = alt.Y("Partial dependence", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["ANN partial dependence min/max"]["min"].min(), model_full_results["ANN partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            #     tooltip = ["Partial dependence"] + [pd_var]
                            #     )
                            #     pd_data_ticks_ann = pd.DataFrame(columns = [pd_var])
                            #     pd_data_ticks_ann[pd_var] = df[pd_var]
                            #     pd_data_ticks_ann["y"] = [model_full_results["ANN partial dependence min/max"]["min"].min()] * df.shape[0]
                            #     pd_ticks_ann = alt.Chart(pd_data_ticks_ann, height = 200).mark_tick(size = 5, thickness = 1).encode(
                            #         x = alt.X(pd_var, axis = alt.Axis(titleFontSize = 12, labelFontSize = 11), scale = alt.Scale(domain = [pd_data_ticks_ann[pd_var].min(), pd_data_ticks_ann[pd_var].max()])),
                            #         y = alt.Y("y", title = "partial dependence", scale = alt.Scale(domain = [model_full_results["ANN partial dependence min/max"]["min"].min(), model_full_results["ANN partial dependence min/max"]["max"].max()]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            #         tooltip = [pd_var]
                            #     )
                            #     if expl_var.index(pd_var)%2 == 0:
                            #         with fm_ann_figs2_col1:
                            #             st.altair_chart(pd_ticks_ann + pd_chart_ann, use_container_width = True)
                            #     if expl_var.index(pd_var)%2 == 1:
                            #         with fm_ann_figs2_col2:
                            #             st.altair_chart(pd_ticks_ann + pd_chart_ann, use_container_width = True)
                            # if sett_hints:
                            #     st.info(str(fc.learning_hints("mod_md_ANN_partDep_bin")))
                            # Confusion matrix
                            st.write("Confusion matrix (columns correspond to predictions):")
                            st.table(model_full_results["ANN confusion"])
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_ANN_confu_mult"))) 
                            st.write("") 
                            # Classification report
                            st.write("Classification report:")
                            st.table(model_full_results["ANN classification report"].style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_ANN_classRep_mult"))) 
                            st.write("") 

                            # Download link for ANN output
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            model_full_results["ANN information"].to_excel(excel_file, sheet_name="classification_information")
                            ann_error_est.to_excel(excel_file, sheet_name="classification_statistics")
                            ann_varImp_table.to_excel(excel_file, sheet_name="variable_importance")
                            model_full_results["ANN confusion"].to_excel(excel_file, sheet_name="confusion_matrix")
                            model_full_results["ANN classification report"].to_excel(excel_file, sheet_name="classification_report")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "ANN full model output__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download Artificial Neural Networks full model output</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("") 

                        # Performance metrics across all models
                        if any(a for a in sb_ML_alg if a == "Random Forest" or a == "Artificial Neural Networks"):
                            st.markdown("**Model comparison**")
                            st.write((model_full_results["model comparison"]).transpose().style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_modComp_mult"))) 
                            st.write("")
                            
                            # Download link for model comparison output
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            model_full_results["model comparison"].transpose().to_excel(excel_file, sheet_name="model_comparison")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "Model comparison full model output__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download model comparison output</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("")
                
                else:
                    st.warning("Please run models!")
                st.write("")

            #--------------------------------------------------------------------------------------
            # FULL MODEL PREDICTIONS

            prediction_output = st.expander("Full model predictions", expanded = False)
            with prediction_output:
                
                if model_full_results is not None:

                     #-------------------------------------------------------------

                    # Multi-class response variable
                    if response_var_type == "multi-class":

                        # RF specific output
                        if any(a for a in sb_ML_alg if a == "Random Forest"):
                            st.markdown("**Random Forest**")

                            pred_col1, pred_col2 = st.columns(2)
                            with pred_col1:
                                st.write("Predictions for original data:")
                                RF_pred_orig = pd.DataFrame(columns = [response_var])
                                RF_pred_orig[response_var] = model_full_results["RF fitted"]
                                RF_pred_orig = RF_pred_orig.join(pd.DataFrame(model_full_results["RF fitted proba"]))
                                st.write(RF_pred_orig.style.format(precision=user_precision))
                            with pred_col2:
                                if do_modprednew == "Yes":
                                    st.write("Predictions for new data:")
                                    RF_pred_new = pd.DataFrame(columns = [response_var])
                                    RF_pred_new[response_var] = model_full_results["RF prediction"]
                                    RF_pred_new = RF_pred_new.join(pd.DataFrame(model_full_results["RF prediction proba"]))
                                    st.write(RF_pred_new.style.format(precision=user_precision))
                            st.write("")
                        
                        # ANN specific output
                        if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                            st.markdown("**Artificial Neural Networks**")

                            pred_col1, pred_col2 = st.columns(2)
                            with pred_col1:
                                st.write("Predictions for original data:")
                                ANN_pred_orig = pd.DataFrame(columns = [response_var])
                                ANN_pred_orig[response_var] = model_full_results["ANN fitted"]
                                ANN_pred_orig = ANN_pred_orig.join(pd.DataFrame(model_full_results["ANN fitted proba"]))
                                st.write(ANN_pred_orig.style.format(precision=user_precision))
                            with pred_col2:
                                if do_modprednew == "Yes":
                                    st.write("Predictions for new data:")
                                    ANN_pred_new = pd.DataFrame(columns = [response_var])
                                    ANN_pred_new[response_var] = model_full_results["ANN prediction"]
                                    ANN_pred_new = ANN_pred_new.join(pd.DataFrame(model_full_results["ANN prediction proba"]))
                                    st.write(ANN_pred_new)
                            st.write("")
                    
                    #-------------------------------------------------------------
                    st.write("")
                    # Download links for prediction data
                    output = BytesIO()
                    predictions_excel = pd.ExcelWriter(output, engine="xlsxwriter")
                    if any(a for a in sb_ML_alg if a == "Random Forest"):
                        RF_pred_orig.to_excel(predictions_excel, sheet_name="RF_pred_orig")
                        if do_modprednew == "Yes":
                            RF_pred_new.to_excel(predictions_excel, sheet_name="RF_pred_new")
                    if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                        ANN_pred_orig.to_excel(predictions_excel, sheet_name="ANN_pred_orig")
                        if do_modprednew == "Yes":
                            ANN_pred_new.to_excel(predictions_excel, sheet_name="ANN_pred_new")
                    predictions_excel.close()
                    predictions_excel = output.getvalue()
                    b64 = base64.b64encode(predictions_excel)
                    dl_file_name = "Full model predictions__" + df_name + ".xlsx"
                    st.markdown(
                        f"""
                    <a href="data:file/predictions_excel;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download full model predictions</a>
                    """,
                    unsafe_allow_html=True)
                    st.write("")
            
            #--------------------------------------------------------------------------------------
            # VALIDATION OUTPUT
            
            if do_modval == "Yes":
                val_output = st.expander("Validation output", expanded = False)
                with val_output:
                    if model_val_results is not None:
                    
                        #------------------------------------
                        # Multi-class response variable

                        if response_var_type == "multi-class":
                            
                            # Metric
                            col1, col2 = st.columns(2)
                            with col1:
                                if model_val_results["mean"].empty:
                                    st.write("")
                                else:
                                    st.write("Means across validation runs:")
                                    st.write(model_val_results["mean"].transpose().style.format(precision=user_precision))
                                    if sett_hints:
                                        st.info(str(fc.learning_hints("mod_md_val_means_mult")))
                                    st.write("")
                            with col2:
                                if model_val_results["sd"].empty:
                                    st.write("")
                                else:
                                    st.write("SDs across validation runs:")
                                    st.write(model_val_results["sd"].transpose().style.format(precision=user_precision))
                                    if sett_hints:
                                        st.info(str(fc.learning_hints("mod_md_val_sds_mult")))
                                    st.write("")

                            val_col1, val_col2 = st.columns(2)
                            with val_col1: 
                                # ACC boxplot
                                if model_val_results["ACC"].empty:
                                    st.write("")
                                else:
                                    st.write("Boxplot of ACC across validation runs:")
                                    acc_results = model_val_results["ACC"]
                                    acc_bplot = pd.melt(acc_results, ignore_index = False, var_name = "Algorithm", value_name = "Value")
                                    acc_boxchart = alt.Chart(acc_bplot, height = 200).mark_boxplot(color = "#1f77b4", median = dict(color = "darkred")).encode(
                                        x = alt.X("Value", title = "ACC", scale = alt.Scale(domain = [min(acc_bplot["Value"]), max(acc_bplot["Value"])])),
                                        y = alt.Y("Algorithm", title = None),
                                        color = alt.Color("Algorithm", legend = None)
                                    ).configure_axis(
                                        labelFontSize = 12,
                                        titleFontSize = 12
                                    )
                                    st.altair_chart(acc_boxchart, use_container_width = True) 
                                    if sett_hints:
                                        st.info(str(fc.learning_hints("mod_md_val_ACCBoxplot")))
                            with val_col2: 
                                # BAL ACC boxplot
                                if model_val_results["BAL ACC"].empty:
                                    st.write("")
                                else:
                                    st.write("Boxplot of BAL ACC across validation runs:")
                                    bal_acc_results = model_val_results["BAL ACC"]
                                    bal_acc_bplot = pd.melt(bal_acc_results, ignore_index = False, var_name = "Algorithm", value_name = "Value")
                                    bal_acc_boxchart = alt.Chart(bal_acc_bplot, height = 200).mark_boxplot(color = "#1f77b4", median = dict(color = "darkred")).encode(
                                        x = alt.X("Value", title = "BAL ACC", scale = alt.Scale(domain = [min(bal_acc_bplot["Value"]), max(bal_acc_bplot["Value"])])),
                                        y = alt.Y("Algorithm", title = None),
                                        color = alt.Color("Algorithm", legend = None)
                                    ).configure_axis(
                                        labelFontSize = 12,
                                        titleFontSize = 12
                                    )
                                    st.altair_chart(bal_acc_boxchart, use_container_width = True)
                                    if sett_hints:
                                        st.info(str(fc.learning_hints("mod_md_val_BALACCBoxplot")))
                            st.write("") 

                            # Variable importance
                            st.write("Means of variable importances:")
                            varImp_table_mean = model_val_results["variable importance mean"]
                            st.write(varImp_table_mean.style.format(precision=user_precision))
                            st.write("SDs of variable importances:")
                            varImp_table_sd = model_val_results["variable importance sd"]
                            st.write(varImp_table_sd.style.format(precision=user_precision))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_val_varImp_mult")))
                            st.write("")
                            st.write("")
                            
                            # Download link for validation output
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            model_val_results["mean"].transpose().to_excel(excel_file, sheet_name="metrics_mean")
                            model_val_results["sd"].transpose().to_excel(excel_file, sheet_name="metrics_sd")
                            varImp_table_mean.to_excel(excel_file, sheet_name="variable_importance_mean")
                            varImp_table_sd.to_excel(excel_file, sheet_name="variable_importance_sd")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "Validation output__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download validation output</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("")   

                    else:
                        st.warning("Please run models!")
                    st.write("")
            
            #--------------------------------------------------------------------------------------
            # HYPERPARAMETER-TUNING OUTPUT

            if any(a for a in sb_ML_alg if a == "Random Forest") or any(a for a in sb_ML_alg if a == "Boosted Regression Trees") or any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                if do_hypTune == "Yes":
                    hype_title = "Hyperparameter-tuning output"
                if do_hypTune != "Yes":
                    hype_title = "Hyperparameter output"
                hype_output = st.expander(hype_title, expanded = False)
                with hype_output:
                    
                    # Random Forest
                    if any(a for a in sb_ML_alg if a == "Random Forest"):
                        st.markdown("**Random Forest**")

                        # Final hyperparameters
                        if rf_finalPara is not None:
                            st.write("Final hyperparameters:")
                            st.table(rf_finalPara.transpose())
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_hypeTune_RF_finPara")))
                            st.write("")
                        else:
                            st.warning("Please run models!")
                        
                        # Tuning details
                        if do_hypTune == "Yes":
                            if rf_tuning_results is not None and rf_finalPara is not None:
                                st.write("Tuning details:")
                                rf_finalTuneMetrics = pd.DataFrame(index = ["value"], columns = ["scoring metric", "number of models", "mean cv score", "standard deviation cv score", "test data score"])
                                rf_finalTuneMetrics["scoring metric"] = [rf_tuning_results.loc["value"]["scoring"]]
                                rf_finalTuneMetrics["number of models"] = [rf_tuning_results.loc["value"]["number of models"]]
                                rf_finalTuneMetrics["mean cv score"] = [rf_tuning_results.loc["value"]["mean score"]]
                                rf_finalTuneMetrics["standard deviation cv score"] = [rf_tuning_results.loc["value"]["std score"]]
                                rf_finalTuneMetrics["test data score"] = [rf_tuning_results.loc["value"]["test score"]]
                                st.table(rf_finalTuneMetrics.transpose().style.format(precision=user_precision))
                                if sett_hints:
                                    st.info(str(fc.learning_hints("mod_md_hypeTune_RF_details")))
                                st.write("")

                    # Boosted Regression Trees
                    if any(a for a in sb_ML_alg if a == "Boosted Regression Trees"):
                        st.markdown("**Boosted Regression Trees**")

                        # Final hyperparameters
                        if brt_finalPara is not None:
                            st.write("Final hyperparameters:")
                            st.table(brt_finalPara.transpose())
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_hypeTune_BRT_finPara")))
                            st.write("")
                        else:
                            st.warning("Please run models!")
                        
                        # Tuning details
                        if do_hypTune == "Yes":
                            if brt_tuning_results is not None and brt_finalPara is not None:
                                st.write("Tuning details:")
                                brt_finalTuneMetrics = pd.DataFrame(index = ["value"], columns = ["scoring metric", "number of models", "mean cv score", "standard deviation cv score", "test data score"])
                                brt_finalTuneMetrics["scoring metric"] = [brt_tuning_results.loc["value"]["scoring"]]
                                brt_finalTuneMetrics["number of models"] = [brt_tuning_results.loc["value"]["number of models"]]
                                brt_finalTuneMetrics["mean cv score"] = [brt_tuning_results.loc["value"]["mean score"]]
                                brt_finalTuneMetrics["standard deviation cv score"] = [brt_tuning_results.loc["value"]["std score"]]
                                brt_finalTuneMetrics["test data score"] = [brt_tuning_results.loc["value"]["test score"]]
                                st.table(brt_finalTuneMetrics.transpose().style.format(precision=user_precision))
                                if sett_hints:
                                    st.info(str(fc.learning_hints("mod_md_hypeTune_BRT_details")))
                                st.write("")

                    # Artificial Neural Networks
                    if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                        st.markdown("**Artificial Neural Networks**")
                        
                        # Final hyperparameters
                        if ann_finalPara is not None:
                            st.write("Final hyperparameters:")
                            st.table(ann_finalPara.transpose().style.format({"L² regularization": "{:.5}"}))
                            if sett_hints:
                                st.info(str(fc.learning_hints("mod_md_hypeTune_ANN_finPara")))
                            st.write("")
                        else:
                            st.warning("Please run models!")
                        
                        # Tuning details
                        if do_hypTune == "Yes":
                            if ann_tuning_results is not None and ann_finalPara is not None:
                                st.write("Tuning details:")
                                ann_finalTuneMetrics = pd.DataFrame(index = ["value"], columns = ["scoring metric", "number of models", "mean cv score", "standard deviation cv score", "test data score"])
                                ann_finalTuneMetrics["scoring metric"] = [ann_tuning_results.loc["value"]["scoring"]]
                                ann_finalTuneMetrics["number of models"] = [ann_tuning_results.loc["value"]["number of models"]]
                                ann_finalTuneMetrics["mean cv score"] = [ann_tuning_results.loc["value"]["mean score"]]
                                ann_finalTuneMetrics["standard deviation cv score"] = [ann_tuning_results.loc["value"]["std score"]]
                                ann_finalTuneMetrics["test data score"] = [ann_tuning_results.loc["value"]["test score"]]
                                st.table(ann_finalTuneMetrics.transpose().style.format(precision=user_precision))
                                if sett_hints:
                                    st.info(str(fc.learning_hints("mod_md_hypeTune_ANN_details")))
                                st.write("")

                    # Download link for hyperparameter output
                    output = BytesIO()
                    excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                    if any(a for a in sb_ML_alg if a == "Random Forest"):
                        rf_finalPara.to_excel(excel_file, sheet_name="RF_final_hyperparameters")
                        if do_hypTune == "Yes":
                            rf_finalTuneMetrics.to_excel(excel_file, sheet_name="RF_tuning_details")
                    if any(a for a in sb_ML_alg if a == "Boosted Regression Trees"):
                        brt_finalPara.to_excel(excel_file, sheet_name="BRT_final_hyperparameters")
                        if do_hypTune == "Yes":
                            brt_finalTuneMetrics.to_excel(excel_file, sheet_name="BRT_tuning_details")
                    if any(a for a in sb_ML_alg if a == "Artificial Neural Networks"):
                        ann_finalPara.to_excel(excel_file, sheet_name="ANN_final_hyperparameters")
                        if do_hypTune == "Yes":
                            ann_finalTuneMetrics.to_excel(excel_file, sheet_name="ANN_tuning_details")
                    excel_file.close()
                    excel_file = output.getvalue()
                    b64 = base64.b64encode(excel_file)
                    if do_hypTune == "Yes":
                        dl_file_name = "Hyperparameter-tuning output__" + df_name + ".xlsx"
                        st.markdown(
                            f"""
                        <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download hyperparameter-tuning output</a>
                        """,
                        unsafe_allow_html=True)
                    if do_hypTune != "Yes":
                        dl_file_name = "Hyperparameter output__" + df_name + ".xlsx"
                        st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download hyperparameter output</a>
                            """,
                            unsafe_allow_html=True)
                        st.write("")

    #--------------------------------------------------------------------------------------

    # DATA DECOMPOSITION
    
    if analysis_type == "Data decomposition":

        #++++++++++++++++++++++++++++++++++++++++++++
        # DIMENSIONALITY REDUCTION

        st.write("")
        st.write("")
        
        data_decomposition_container = st.container()
        with data_decomposition_container:
            st.header("**Data decomposition**")
            st.markdown("STATY will take care of the decomposition for you, so you can put your focus on results interpretation and communication! ")

            dd_settings = st.expander("Specify method", expanded = False)
            with dd_settings:

                if df.shape[1] > 0 and df.shape[0] > 0:
                    #--------------------------------------------------------------------------------------
                    # GENERAL SETTINGS
                
                    st.markdown("**Variable selection**")

                    # Select variables
                    var_options = list(df.select_dtypes(['number']).columns)
                    if len(var_options)>0:
                        decomp_var = st.multiselect("Select variables for decomposition", var_options, var_options, on_change=in_wid_change)
                    else:
                        st.error("ERROR: No numeric variables in dataset!")
                        return
                    
                    # Include response variable in output?
                    resp_var_dec = st.selectbox("Include response variable in transformed data output", ["No", "Yes"], on_change=in_wid_change)
                    if resp_var_dec == "Yes":
                        resp_var_options = df.columns
                        resp_var_options = resp_var_options[resp_var_options.isin(df.drop(decomp_var, axis = 1).columns)]
                        resp_var = st.selectbox("Select response variable for transformed data output", resp_var_options, on_change=in_wid_change)

                    # Filter data according to selected variables
                    if len(decomp_var) < 2:
                        st.error("ERROR: Please select more than 1 variable!")
                        return
                    else:
                        # Decomposition data set (and response variable)
                        if resp_var_dec == "Yes":
                            df = df[list([resp_var]) + list(decomp_var)]
                        else:
                            df = df[decomp_var]

                        # Check if NAs are present and delete them automatically 
                        if np.where(df[decomp_var].isnull())[0].size > 0:
                            st.warning("WARNING: Your data set includes NAs. Rows with NAs are automatically deleted!")
                            df = df.dropna()

                        #--------------------------------------------------------------------------------------
                        # ALGORITHMS

                        st.markdown("**Specify algorithm**")
                        DEC_alg = st.selectbox("Select decomposition technique", ["Principal Component Analysis", "Factor Analysis"])

                        if DEC_alg == "Factor Analysis":
                            
                            # Defaul settings
                            nfactors = len(decomp_var)
                            farotation = None
                            famethod = "ml"

                            # Adjust settings
                            if st.checkbox("Adjust Factor Analysis settings"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    nfactors = st.number_input("Number of factors", min_value=2, max_value=len(decomp_var), value=len(decomp_var))
                                    farotation = st.selectbox("Rotation", [None, "varimax", "promax", "oblimin", "oblimax", "quartimin", "quartimax", "equamax"])
                                    if farotation == "None":
                                        farotation = None
                                with col2:
                                    famethod = st.selectbox("Fitting method", ["Maximum Likelihood", "MINRES", "Principal Factor"])           
                                    if famethod == "Maximum Likelihood":
                                        famethod = "ml"
                                    elif famethod == "MINRES":
                                        famethod = "minres"
                                    elif famethod == "Principal Factor":
                                        famethod = "principal"  

                        #--------------------------------------------------------------------------------------
                        # DECOMPOSITION DATA
                        
                        st.write("")

                        # Show data
                        if resp_var_dec == "Yes":
                            show_data_text = "Show data for response variable and decomposition"
                        else:
                            show_data_text = "Show data for decomposition"
                        if st.checkbox(show_data_text):
                            st.write(df)
                            st.write("Data shape: ", df.shape[0],  " rows and ", df.shape[1], " columns")
                            
                            # Download link for decomposition data
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            df.to_excel(excel_file, sheet_name="decomposition_data")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name= "Decomposition data__" + df_name + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download decomposition data</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("")

                        #--------------------------------------------------------------------------------------
                        # RUN DECOMPOSITION

                        # Decomposition is run on button click
                        st.write("")
                        run_decomposition = st.button("Run decomposition")
                        st.write("")
                        
                        if run_decomposition:

                            decomp_results = {}

                            # Check algorithm
                            # Principal Component Analysis
                            if DEC_alg == "Principal Component Analysis":
                               
                               # Standardize data
                                X_pca = df[decomp_var]
                                scaler = StandardScaler()
                                scaler.fit(X_pca)
                                X_pca = scaler.transform(X_pca)

                                # Create components names
                                components_names = []
                                for i in range(len(decomp_var)):
                                    components_names.append("pc_" + str(i+1))
                                
                                # Adjust index and column names
                                X_pca = pd.DataFrame(X_pca, index = df[decomp_var].index, columns = df[decomp_var].columns)

                                # Fit PCA
                                pca = decomposition.PCA()
                                pca.fit(X_pca)

                                # Transform data 
                                X_pca_transform = pca.transform(X_pca)
                                
                                # Adjust index and column names
                                X_pca_transform = pd.DataFrame(X_pca_transform, index = df[decomp_var].index, columns = components_names)

                                # Add response variable if wanted
                                if resp_var_dec == "Yes":
                                    X_pca_transform[resp_var] = df[resp_var]

                                # Save results
                                EVEV = pd.DataFrame(pca.explained_variance_, index = components_names, columns = ["eigenvalue"])
                                EVEV["explained variance ratio"] = pca.explained_variance_ratio_
                                EVEV["cumulative explained variance"] = np.cumsum(pca.explained_variance_ratio_)
                                
                                decomp_results["transformed data"] = X_pca_transform 
                                decomp_results["eigenvalues and explained variance"] = EVEV
                                decomp_results["eigenvectors"] = pd.DataFrame(pca.components_.T, index = decomp_var, columns = components_names)

                            # Factor Analysis
                            if DEC_alg == "Factor Analysis":
                            
                                # Standardize data
                                X_fa = df[decomp_var]
                                scaler = StandardScaler()
                                scaler.fit(X_fa)
                                X_fa = scaler.transform(X_fa)

                                # Create components names
                                components_names1 = []
                                for i in range(len(decomp_var)):
                                    components_names1.append("factor_" + str(i+1))
                                components_names2 = []
                                for i in range(nfactors):
                                    components_names2.append("factor_" + str(i+1))
                                
                                # Adjust index and column names
                                X_fa = pd.DataFrame(X_fa, index = df[decomp_var].index, columns = df[decomp_var].columns)

                                # Fit FA
                                fa = FactorAnalyzer(n_factors=nfactors, rotation= farotation, method= famethod)
                                fa.fit(X_fa)

                                # Transform data  
                                X_fa_transform = fa.transform(X_fa)
                                
                                # Adjust index and column names
                                X_fa_transform = pd.DataFrame(X_fa_transform, index = df[decomp_var].index, columns = components_names2)

                                # Add response variable if wanted
                                if resp_var_dec == "Yes":
                                    X_fa_transform[resp_var] = df[resp_var]

                                # Save results                                 
                                BST = pd.DataFrame(index = ["statistic", "dof", "p-value"], columns = ["value"])
                                BST.loc["statistic"] = -np.log(np.linalg.det(X_fa.corr()))* (X_fa.shape[0] - 1 - (2 * X_fa.shape[1] + 5) / 6)
                                BST.loc["dof"] = X_fa.shape[1] * (X_fa.shape[1]  - 1) / 2
                                BST.loc["p-value"] = stats.chi2.sf(BST.loc["statistic"][0], BST.loc["dof"][0])
                                KMO = pd.DataFrame(calculate_kmo(X_fa)[1], index = ["KMO"], columns = ["value"])
                                EV = pd.DataFrame(fa.get_eigenvalues()[0], index = components_names1, columns = ["eigenvalue"])
                                EV["explained variance ratio"] = EV["eigenvalue"]/sum(EV["eigenvalue"])
                                EV["cumulative explained variance"] = np.cumsum(EV["explained variance ratio"])
                                EV["common factor eigenvalue"] = fa.get_eigenvalues()[1]
                                LEV =  pd.DataFrame(fa.get_factor_variance()[0], index = components_names2, columns = ["SS loadings"])
                                LEV["explained variance ratio"] = fa.get_factor_variance()[1]
                                LEV["cumulative explained variance"] = fa.get_factor_variance()[2]
                                CU = pd.DataFrame(fa.get_communalities(), index = df[decomp_var].columns, columns = ["communality"])
                                CU["uniqueness"] = fa.get_uniquenesses()

                                decomp_results["transformed data"] = X_fa_transform
                                decomp_results["BST"] = BST
                                decomp_results["KMO"] = KMO 
                                decomp_results["eigenvalues"] = EV
                                decomp_results["loadings and explained variance"] = LEV
                                decomp_results["communalities and uniqueness"] = CU
                                decomp_results["loadings"] = pd.DataFrame(fa.loadings_, index = df[decomp_var].columns, columns = components_names2)
                else: 
                    st.error("ERROR: No data available for Modelling!") 
                    return
                
        #----------------------------------------------------------------------------------------------

        if run_decomposition:
            st.write("")
            st.write("")
            st.header("**Decomposition outputs**")

            if DEC_alg == "Principal Component Analysis":
                expander_text = "Principal Component Analysis results"
            if DEC_alg == "Factor Analysis":
                expander_text = "Factor Analysis results"
            decomp_res1 = st.expander(expander_text, expanded = False)
            with decomp_res1:
                
                corr_matrix = df[decomp_var].corr()
                if len(decomp_var) <= 10:
                    st.write("Correlation Matrix & 2D-Histogram")
                    # Define variable selector
                    var_sel_cor = alt.selection_single(fields=['variable', 'variable2'], clear=False, 
                                        init={'variable': decomp_var[0], 'variable2': decomp_var[0]})
                    # Calculate correlation data
                    corr_data = df[decomp_var].corr().stack().reset_index().rename(columns={0: "correlation", 'level_0': "variable", 'level_1': "variable2"})
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
                    value_columns = df[decomp_var]
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
                    corr_plot1 = (corr_plot + text).properties(width = 400, height = 400)
                    correlation_plot = correlation_plot.properties(padding = {"left": 50, "top": 5, "right": 5, "bottom": 50})
                    # hist_2d_plot = scat_plot.properties(height = 350)
                    st.altair_chart(correlation_plot, use_container_width = True)
                    if sett_hints:
                        st.info(str(fc.learning_hints("decomp_cor")))
                    st.write("")
                else:
                    st.write("Correlation matrix:")
                    st.write(corr_matrix.style.format(precision=user_precision))
                    st.write("")

                # Principal Component Analysis Output
                if DEC_alg == "Principal Component Analysis":

                    st.write("Eigenvalues and explained variance:")
                    st.table(decomp_results["eigenvalues and explained variance"].style.format(precision=user_precision))
                    if sett_hints:
                        st.info(str(fc.learning_hints("decomp_pca_eigval")))
                    st.write("")
                    st.write("Eigenvectors:")
                    st.table(decomp_results["eigenvectors"].style.format(precision=user_precision))
                    if sett_hints:
                        st.info(str(fc.learning_hints("decomp_pca_eigvec")))
                    st.write("")

                    # Scree plot
                    st.write("Scree plot:")
                    scree_plot_data = decomp_results["eigenvalues and explained variance"].copy()
                    scree_plot_data["component"] = decomp_results["eigenvalues and explained variance"].index
                    scree_plot1 = alt.Chart(scree_plot_data, height = 200).mark_line(point = True).encode(
                        x = alt.X("component", sort = list(scree_plot_data["component"]), title = "princial component", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                        y = alt.Y("explained variance ratio", title = "proportion of variance", scale = alt.Scale(domain = [0, 1]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                        tooltip = ["cumulative explained variance", "explained variance ratio", "component",]
                    )
                    scree_plot2 = alt.Chart(scree_plot_data, height = 200).mark_line(color = "darkred", point = True).encode(
                        x = alt.X("component", sort = list(scree_plot_data["component"]), title = "princial component", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                        y = alt.Y("cumulative explained variance", title = "proportion of variance", scale = alt.Scale(domain = [0, 1]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                        tooltip = ["cumulative explained variance", "explained variance ratio", "component",]
                    )
                    st.altair_chart(scree_plot1 + scree_plot2, use_container_width = True)
                    st.write("")
                    
                    # 2D principal component plot
                    if resp_var_dec == "Yes":
                        st.write("2D principal component plot:")
                        pc_plot_data = decomp_results["transformed data"].copy()
                        pc_plot_data["index"] = decomp_results["transformed data"].index
                        pc_plot = alt.Chart(pc_plot_data, height = 200).mark_circle(point = True).encode(
                            x = alt.X("pc_1", title = "princial component 1", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            y = alt.Y("pc_2", title = "princial component 2", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            color = resp_var,
                            tooltip = ["pc_1", "pc_2", "index",]
                        )
                        st.altair_chart(pc_plot, use_container_width = True)
                        st.write("") 
                    else:
                        st.write("2D principal component plot:")
                        pc_plot_data = decomp_results["transformed data"].copy()
                        pc_plot_data["index"] = decomp_results["transformed data"].index
                        pc_plot = alt.Chart(pc_plot_data, height = 200).mark_circle(point = True).encode(
                            x = alt.X("pc_1", title = "princial component 1", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            y = alt.Y("pc_2", title = "princial component 2", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            tooltip = ["pc_1", "pc_2", "index",]
                        )
                        st.altair_chart(pc_plot, use_container_width = True)
                        st.write("") 
                    
                    # Download link for decomposition results
                    output = BytesIO()
                    excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                    corr_matrix.to_excel(excel_file, sheet_name="correlation_matrix")
                    decomp_results["eigenvalues and explained variance"].to_excel(excel_file, sheet_name="eigval_and_explained_variance")
                    decomp_results["eigenvectors"].to_excel(excel_file, sheet_name="eigenvectors")
                    excel_file.close()
                    excel_file = output.getvalue()
                    b64 = base64.b64encode(excel_file)
                    dl_file_name= "Decomposition output__PCA__" + df_name + ".xlsx"
                    st.markdown(
                        f"""
                    <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download decomposition output</a>
                    """,
                    unsafe_allow_html=True)
                    st.write("")

                # Factor Analysis
                if DEC_alg == "Factor Analysis":

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Bartlett's Sphericity test:")
                        st.table(decomp_results["BST"].style.format(precision=user_precision))
                    with col2:
                        st.write("Kaiser-Meyer-Olkin criterion:")
                        st.table(decomp_results["KMO"].style.format(precision=user_precision))
                    if sett_hints:
                        st.info(str(fc.learning_hints("decomp_fa_adeqtests")))
                    st.write("")

                    st.write("Eigenvalues:")
                    st.table(decomp_results["eigenvalues"].style.format(precision=user_precision))
                    if sett_hints:
                        st.info(str(fc.learning_hints("decomp_fa_eigval")))
                    st.write("")
                    st.write("Explained variance:")
                    st.table(decomp_results["loadings and explained variance"].style.format(precision=user_precision))
                    if sett_hints:
                        st.info(str(fc.learning_hints("decomp_fa_explvar")))
                    st.write("")
                    st.write("Communalities and uniquenesses:")
                    st.table(decomp_results["communalities and uniqueness"].style.format(precision=user_precision))
                    if sett_hints:
                        st.info(str(fc.learning_hints("decomp_fa_comuniq")))
                    st.write("")
                    st.write("Loadings:")
                    st.table(decomp_results["loadings"].style.format(precision=user_precision))
                    if sett_hints:
                        st.info(str(fc.learning_hints("decomp_fa_loadings")))
                    st.write("")                 

                    # Scree plot
                    st.write("Scree plot:")
                    scree_plot_data = decomp_results["eigenvalues"].copy()
                    scree_plot_data["component"] = [str(i+1) for i in range(len(decomp_var))]
                    scree_plot1 = alt.Chart(scree_plot_data, height = 200).mark_line(point = True).encode(
                        x = alt.X("component", sort = list(scree_plot_data["component"]), title = "component", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11, labelAngle = 0)),
                        y = alt.Y("eigenvalue", title = "eigenvalue", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                        tooltip = ["eigenvalue", "component",]
                    )
                    Kaiser_criterion = alt.Chart(pd.DataFrame({'y': [1]}), height = 200).mark_rule(size = 2, color = "darkred").encode(y='y') 
                    st.altair_chart(scree_plot1+ Kaiser_criterion, use_container_width = True)
                    #if sett_hints:
                        #st.info(str(fc.learning_hints("mod_md_BRT_thresAUC")))
                    st.write("")
                    
                    # 2D factor loadings plot
                    if nfactors >= 2:
                        st.write("2D factor loadings plot:")
                        comp_plot_data = decomp_results["loadings"].copy()
                        comp_plot_data["variable"] = decomp_results["loadings"].index
                        comp_plot = alt.Chart(comp_plot_data, height = 200).mark_circle(point = True).encode(
                            x = alt.X("factor_1", title = "factor 1", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            y = alt.Y("factor_2", title = "factor 2", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                            tooltip = ["factor_1", "factor_2", "variable",]
                        )
                        yaxis = alt.Chart(pd.DataFrame({'y': [0]}), height = 200).mark_rule(size = 2, color = "darkred").encode(y='y') 
                        xaxis = alt.Chart(pd.DataFrame({'x': [0]}), height = 200).mark_rule(size = 2, color = "darkred").encode(x='x') 
                        st.altair_chart(comp_plot + yaxis + xaxis, use_container_width = True)
                        st.write("") 

                    # Download link for decomposition results
                    output = BytesIO()
                    excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                    corr_matrix.to_excel(excel_file, sheet_name="correlation_matrix")
                    decomp_results["BST"].to_excel(excel_file, sheet_name="Bartlett's_sphericity_test")
                    decomp_results["KMO"].to_excel(excel_file, sheet_name="Kaiser-Meyer-Olkin_criterion")
                    decomp_results["eigenvalues"].to_excel(excel_file, sheet_name="eigenvalues")
                    decomp_results["loadings and explained variance"].to_excel(excel_file, sheet_name="explained_variance")
                    decomp_results["communalities and uniqueness"].to_excel(excel_file, sheet_name="communalities_uniqueness")
                    decomp_results["loadings"].to_excel(excel_file, sheet_name="loadings")
                    excel_file.close()
                    excel_file = output.getvalue()
                    b64 = base64.b64encode(excel_file)
                    if farotation is not None:
                        dl_file_name= "Decomposition output__FA(" + str(famethod) + ", " + str(farotation) + ")__" + df_name + ".xlsx"
                    else:
                        dl_file_name= "Decomposition output__FA(" + str(famethod) + ")__" + df_name + ".xlsx"
                    st.markdown(
                        f"""
                    <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download decomposition output</a>
                    """,
                    unsafe_allow_html=True)
                    st.write("")

            decomp_res2 = st.expander("Transformed data", expanded = False)
            with decomp_res2:

                # Principal Component Analysis Output
                if DEC_alg == "Principal Component Analysis":

                    st.write(decomp_results["transformed data"].style.format(precision=user_precision))
                    st.write("Data shape: ", decomp_results["transformed data"].shape[0],  " rows and ", decomp_results["transformed data"].shape[1], " columns")
                    
                    # Download link for transformed data
                    output = BytesIO()
                    excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                    decomp_results["transformed data"].to_excel(excel_file, sheet_name="transformed_data")
                    excel_file.close()
                    excel_file = output.getvalue()
                    b64 = base64.b64encode(excel_file)
                    dl_file_name= "Transformed data__PCA__" + df_name + ".xlsx"
                    st.markdown(
                        f"""
                    <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download transformed data</a>
                    """,
                    unsafe_allow_html=True)
                    st.write("")

                # Factor Analysis Output
                if DEC_alg == "Factor Analysis":

                    st.write(decomp_results["transformed data"].style.format(precision=user_precision))
                    st.write("Data shape: ", decomp_results["transformed data"].shape[0],  " rows and ", decomp_results["transformed data"].shape[1], " columns")
                    
                    # Download link for transformed data
                    output = BytesIO()
                    excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                    decomp_results["transformed data"].to_excel(excel_file, sheet_name="transformed_data")
                    excel_file.close()
                    excel_file = output.getvalue()
                    b64 = base64.b64encode(excel_file)
                    if farotation is not None:
                        dl_file_name= "Transformed data__FA(" + str(famethod) + ", " + str(farotation) + ")__" + df_name + ".xlsx"
                    else:
                        dl_file_name= "Transformed data__FA(" + str(famethod) + ")__" + df_name + ".xlsx"
                    st.markdown(
                        f"""
                    <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download transformed data</a>
                    """,
                    unsafe_allow_html=True)
                    st.write("")

#--------------------------------------------------------------------------------------
