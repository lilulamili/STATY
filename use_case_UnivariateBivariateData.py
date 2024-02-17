#----------------------------------------------------------------------------------------------
from operator import index
import streamlit as st
import pandas as pd
import numpy as np
from streamlit.proto.DataFrame_pb2 import Index
from streamlit.proto.RootContainer_pb2 import SIDEBAR
import plotly as dd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager
import plotly.graph_objects as go
import functions as fc
import elements as el
import altair as alt
import scipy
import math
import os
import sys
import platform
import base64
from io import BytesIO
from streamlit_js_eval import streamlit_js_eval
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
from scipy.stats import t
from scipy.stats import norm


#----------------------------------------------------------------------------------------------

def app():

    # Clear cache
    st.runtime.legacy_caching.clear_cache()

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

    #++++++++++++++++++++++++++++++++++++++++++++
    # RESET INPUT
    
    #Session state
    if 'key' not in st.session_state:
        st.session_state['key'] = 0
    reset_clicked = st.sidebar.button("Reset all your input")
    if reset_clicked:
        #st.session_state['key'] = st.session_state['key'] + 1
        streamlit_js_eval(js_expressions="parent.window.location.reload()")
    st.sidebar.markdown("")
    def in_wid_change():
        multiv_session=None  
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
                col_sep=a5.selectbox("Column sep.",[';',  ','  , '|', '\s+','\t','other'])
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
           df = pd.read_csv("default data/savings.csv", sep = ";|,|\t",engine='python')
           df_name="Savings" 
    else:
        df = pd.read_csv("default data/savings.csv", sep = ";|,|\t",engine='python') 
        df_name="Savings" 
    st.sidebar.markdown("")
     
    #Basic data info
    n_rows = df.shape[0]
    n_cols = df.shape[1]  

    #------------------------------------------------------------------------------------------
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

    #++++++++++++++++++++++++++++++++++++++++++++
    # DATA PREPROCESSING & VISUALIZATION

    data_title_container = st.container()
    with data_title_container:
        st.header("**Uni- and bivariate data**")
        st.markdown("Let STATY do the data cleaning, variable transformations, visualizations and deliver you the stats you need. Specify your data processing preferences and start exploring your data stories right below... ")

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

        dev_expander_raw = st.expander("Explore raw data info and stats", expanded = False)
        with dev_expander_raw:
            # Default data description:
            if uploaded_data == None:
                if st.checkbox("Show data description", value = False):          
                    st.markdown("**Data source:**")
                    st.markdown("The data come from the World Bank's study on [financial intermediation and growth](https://datacatalog.worldbank.org/dataset/wps2059-finance-and-sources-growth-and-financial-intermediation-and-growth). ")
                    
                    st.markdown("**Citation:**")
                    st.markdown(" Levine, Ross; Loayza, Norman; Beck, Thorsten.1999. Financial intermediation and growth : Causality and causes (English). Policy, Research working paper ; no. WPS 2059 Washington, D.C. : World Bank Group. ")
                    
                    st.markdown("**Variables in the dataset:**")
                    st.write("*The term 'period average' relates to the period 1970-1995.*")
                    col1,col2=st.columns(2) 
                    col1.write("UrbanPopulation")
                    col2.write("Share of urban population")
                    
                    col1,col2=st.columns(2)
                    col1.write("PrivateSaving")
                    col2.write("Private saving rate as the ratio of gross private savings and GPDI")
                    
                    col1,col2=st.columns(2)	
                    col1.write("logGDPI")
                    col2.write("Log of real per capita GPDI")

                    col1,col2=st.columns(2)
                    col1.write("GrowtRate")
                    col2.write("Growth rate of real GPDI")

                    col1,col2=st.columns(2)
                    col1.write("GovermentSaving")
                    col2.write("Government saving as share of real GDP")

                    col1,col2=st.columns(2)
                    col1.write("LogTermsTrade")
                    col2.write("Log of terms of trade")

                    col1,col2=st.columns(2)
                    col1.write("OlderThan65")
                    col2.write("Share of population over 65 in total population")

                    col1,col2=st.columns(2)
                    col1.write("Under15")
                    col2.write("Share of population under 15 in total population")

                    col1,col2=st.columns(2)
                    col1.write("CommercialCentralBank")
                    col2.write("Commercial-Central Bank, period average")

                    col1,col2=st.columns(2)
                    col1.write("LiquidLiabilities")
                    col2.write("Liquid Liabilities, period average")

                    col1,col2=st.columns(2)
                    col1.write("PrivateCredit")
                    col2.write("Private Credit, period average")

                    col1,col2=st.columns(2)
                    col1.write("BankCredit")
                    col2.write("Bank credit, period average")

                    col1,col2=st.columns(2)
                    col1.write("LegalOrigin")
                    col2.write("English, French, German or Scandinavian")


                    st.markdown("")

            # Show raw data & data info
            df_summary = fc.data_summary(df) 
            if st.checkbox("Show raw data", value = False):      
            # st.dataframe(df.style.apply(lambda x: ["background-color: #ffe5e5" if (not pd.isna(df_summary_mq_full.loc["1%-Q"][i]) and df_summary_vt_cat[i] == "numeric" and (v <= df_summary_mq_full.loc["1%-Q"][i] or v >= df_summary_mq_full.loc["99%-Q"][i]) or pd.isna(v)) else "" for i, v in enumerate(x)], axis = 1))
                
                st.write(df)
                 # Download link for data
                output = BytesIO()
                excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                df.to_excel(excel_file, sheet_name="data",index=False)    
                excel_file.close()
                excel_file = output.getvalue()
                b64 = base64.b64encode(excel_file)
                dl_file_name = df_name + "_staty.xlsx"
                st.markdown(
                    f"""
                <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Show data in Excel</a>
                """,
                unsafe_allow_html=True)
                st.write("") 
                  
                
                st.write("Data shape: ", n_rows,  " rows and ", n_cols, " columns")
                st.write("") 
                st.write("") 
                #st.info("** Note that NAs and numerical values below/ above the 1%/ 99% quantile are highlighted.") 
            if df[df.duplicated()].shape[0] > 0 or df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0] > 0:
                check_nasAnddupl=st.checkbox("Show duplicates and NAs info", value = False) 
                if check_nasAnddupl:      
                    if df[df.duplicated()].shape[0] > 0:
                        st.write("Number of duplicates: ", df[df.duplicated()].shape[0])
                        st.write("Duplicate row index: ", ', '.join(map(str,list(df.index[df.duplicated()]))))
                    if df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0] > 0:
                        st.write("Number of rows with NAs: ", df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0])
                        st.write("Rows with NAs: ", ', '.join(map(str,list(pd.unique(np.where(df.isnull())[0])))))
                
            # Show variable info 
            if st.checkbox('Show variable info', value = False): 
                #st.write(df_summary["Variable types"])
                               
                st.write(df_summary["Variable types"])
            # Show summary statistics (raw data)
            if st.checkbox('Show summary statistics (raw data)', value = False): 
                #st.write(df_summary["ALL"])
                df_datasumstat=df_summary["ALL"]
                #dfStyler = df_datasumstat.style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector = 'th', props=[('text-align', 'left')])]) 
                
                st.write(df_datasumstat.style.format(precision=user_precision))
                if fc.get_mode(df).loc["n_unique"].any():
                    st.caption("** Mode is not unique.")
                if sett_hints:
                    st.info(str(fc.learning_hints("de_summary_statistics")))

                # Download link for summary statistics
                        
                output = BytesIO()
                excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                df_summary["Variable types"].to_excel(excel_file, sheet_name="variable_info")
                df_summary["ALL"].to_excel(excel_file, sheet_name="summary_statistics")
                excel_file.close()
                excel_file = output.getvalue()
                b64 = base64.b64encode(excel_file)
                dl_file_name = "Summary statistics_univariate_" + df_name + ".xlsx"
                st.markdown(
                    f"""
                <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download summary statistics</a>
                """,
                unsafe_allow_html=True)
        
        
        #---------------------------------
        # DATA PROCESSING       
        #---------------------------------
        (df,sb_DM_dTrans_ohe, sb_DM_dTrans_log, sb_DM_dTrans_sqrt, sb_DM_dTrans_square, sb_DM_dTrans_cent, sb_DM_dTrans_stand, sb_DM_dTrans_norm, sb_DM_dTrans_numCat, sb_DM_dTrans_mult, sb_DM_dTrans_div)=el.data_processing(df_name,df, n_rows,n_cols,sett_hints, user_precision,in_wid_change)
        
        
                    
    #------------------------------------------------------------------------------------------
    
    data_visualization_container = st.container()
    with data_visualization_container:
        #---------------------------------
        # DATA VISUALIZATION
        #---------------------------------
        st.write("")
        st.write("")
        st.header("**Data visualization**")
        
        dev_expander_datavis = st.expander("Check some data charts", expanded = False)
        with dev_expander_datavis:
            
            a4, a5= st.columns(2)   
            with a4:
                # Scatterplot
                st.subheader("Scatterplot") 
                x_var = st.selectbox('Select x variable for your scatterplot', df.columns)    
                y_var = st.selectbox('Select y variable for your scatterplot', df.columns)
                
                if x_var==y_var:
                    df_to_plot=df[[x_var]]
                else: 
                    df_to_plot=df[[x_var,y_var]]
                df_to_plot.loc[:,"Index"]=df.index                             
                fig = px.scatter(data_frame=df_to_plot, x=x_var, y=y_var,hover_data=[x_var, y_var, "Index"], color_discrete_sequence=['rgba(77, 121, 169, 0.7)'])
                fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
                fig.update_layout(xaxis=dict(title=x_var, titlefont_size=14, tickfont_size=14,),)
                fig.update_layout(yaxis=dict(title=y_var, titlefont_size=14, tickfont_size=14,),)
                fig.update_layout(hoverlabel=dict(bgcolor="white", ))
                fig.update_layout(height=400,width=400)
                st.plotly_chart(fig) 
                
                if sett_hints:
                    st.info(str(fc.learning_hints("dv_scatterplot")))
            # st.altair_chart(s_chart, use_container_width=True)
                
            with a5:
                #Boxplot
                st.subheader("Boxplot")  
                bx_var = st.selectbox('Draw a boxplot for...?', df.columns)    
                st.markdown("") 
                st.markdown("") 
                st.markdown("")  
                st.markdown("") 
                st.markdown("") 
                st.markdown("") 
                                
                df_to_plot=df[[bx_var]]                
                df_to_plot.loc[:,"Index"]=df.index                                         
                fig = go.Figure()
                fig.add_trace(go.Box( 
                    y=df[bx_var],name=bx_var,
                    boxpoints='all', jitter=0,whiskerwidth=0.2,
                    marker_color = 'indianred', customdata=df_to_plot["Index"], marker_size=2, line_width=1)
                )
                fig.update_traces(hovertemplate=bx_var+': %{y} <br> Index: %{customdata}') 
                #fig = px.box(df_to_plot, y=bx_var,points='all',labels={bx_var:bx_var}, color_discrete_sequence =['indianred'], notched=False, hover_data=[bx_var,"Index"]   )
                #fig.update_traces(marker=dict(size=2))
                #fig.update_traces(line=dict(width=1))
                

                fig.update_layout(font=dict(size=14,),)
                fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',})  
                fig.update_layout(hoverlabel=dict(bgcolor="white",align="left"))
                fig.update_layout(height=400,width=400)
                st.plotly_chart(fig)

                if sett_hints:
                    st.info(str(fc.learning_hints("dv_boxplot")))

    #---------------------------------
    # METHODS
    #---------------------------------
    
    data_analyses_container = st.container()
    with data_analyses_container:
        
        #---------------------------------
        # Frequency analysis
        #---------------------------------
        st.write("")
        st.write("")
        st.header('**Data analyses**')
        dev_expander_fa = st.expander("Univariate frequency analysis", expanded = False)
        
        #initialisation     
        fa_low=None
        fa_up=None

        with dev_expander_fa:
                
            feature = st.selectbox('Which variable would you like to analyse?', df.columns)
            
            if len(df[feature])!=len(df[feature].dropna()):
                st.warning("WARNING: Your data set includes NAs! To ensure comparability of your results, we recommend you to go through the data cleaning process within 'Data screening and processing' section.")
            user_order=[] 
            
            
            if df[feature].dtypes=="int64" or df[feature].dtypes=="float64":
                fa_uniqueLim=30
            else:
                fa_uniqueLim=((df[feature]).unique()).size

            #-----------------------------------------------
            # Identify the plot type (bar/hist) & key chart properties
            if ((df[feature]).unique()).size>fa_uniqueLim:               
                default_BinNo=min(10,math.ceil(np.sqrt(df.size)))
                default_bins=default_BinNo
                plot_type="hist" # i.e. count freq. for value ranges                  
            elif df[feature].dtypes=="object" or df[feature].dtypes=="bool":
                default_BinNo=((df[feature]).unique()).size
                default_bins=sorted(pd.Series(df[feature]).unique())
                plot_type="bars" # i.e. count freq. for every unique sample values
            elif df[feature].dtypes=="int64" or df[feature].dtypes=="float64": 
                freq_type_sel='classes'
                freq_type_sel=st.selectbox("Please specify if you would like to group the values into classes (option 'classes'), or to analyse frequency of every single unique value (option 'unique values')", ['classes','unique values'])
                if freq_type_sel=='classes':
                    default_BinNo=min(10,math.ceil(np.sqrt(df.size)))
                    default_bins=default_BinNo
                    plot_type="hist" # i.e. count freq. for value ranges
                else:    
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

                
            fa_data_output=st.checkbox("Include data for frequency analysis in the output file", value = False)     
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
                        fa_data=fa_data[fa_data.between(fa_low,fa_up, inclusive='both')] 
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
                    #df_freqanal['Rel. freq.']=pd.Series(["{0:.2f}%".format(fa_val * 100) for fa_val in df_freqanal['Rel. freq.']], index = df_freqanal.index)
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
                    fig.update_layout(yaxis=dict(title='Relative frequency', titlefont_size=14, tickfont_size=14,),)
                    fig.update_layout(xaxis=dict(title=feature, titlefont_size=14, tickfont_size=14,),)
                    fig.update_layout(hoverlabel=dict(bgcolor="white",align="left"))
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
                    fig.update_layout(yaxis=dict(title='Relative frequency', titlefont_size=14, tickfont_size=14,),)
                    fig.update_layout(xaxis=dict(title=feature, titlefont_size=14, tickfont_size=14,),)
                    fig.update_layout(hoverlabel=dict(bgcolor="white",align="left"))
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
                st.subheader("Frequency table")  #st.table(df_freqanal.style.format(precision=user_precision))
                st.write(df_freqanal.style.format(precision=user_precision))

                # xls-File download link:
                output = BytesIO()
                excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                if fa_data_output==True:
                    df[feature].to_excel(excel_file, sheet_name="data")
                df_freqanal.to_excel(excel_file, sheet_name="frequency analysis")
                excel_file.close()
                excel_file = output.getvalue()
                b64 = base64.b64encode(excel_file)
                dl_file_name = "Frequency analysis_univariate_" + df_name +"_" + feature+ ".xlsx"
                st.markdown(
                    f"""
                <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download frequency table</a>
                """,
                unsafe_allow_html=True)
                st.write("")  


                st.write("") 
                st.write("") 
        # -------------------
        # Anova
        #----------------------

        dev_expander_anova = st.expander("ANOVA", expanded = False)
        with dev_expander_anova:
                
                
            # Target variable
            target_var = st.selectbox('Select the target variable', df.columns)
            if df[target_var].dtypes=="int64" or df[target_var].dtypes=="float64": 
                class_var_options = df.columns
                class_var_options = class_var_options[class_var_options.isin(df.drop(target_var, axis = 1).columns)]
                clas_var=st.selectbox('Select the classifier variable', class_var_options) 

                st.write("")
                anova_data_output=st.checkbox("Include data from ANOVA in the output file", value=False)
                st.write("")

                if len((df[clas_var]).unique())>=len(df[clas_var]):
                    st.error("ERROR: The variable you selected is not suitable as a classifier!")
                else:
                    anova_data=df[[clas_var,target_var]]
                    if len(anova_data)!=len(anova_data.dropna()):
                        anova_data=anova_data.dropna()
                        st.warning("WARNING: Your data set includes NAs. Rows with NAs are automatically deleted!")          
                        st.write("")    
                    run_anova = st.button("Press to perform one-way ANOVA") 
                    st.write("") 
                    st.write("")

                    if run_anova:         
                        
                        # Boxplot
                        anova_data['Index']=anova_data.index
                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            x=anova_data[clas_var],y=anova_data[target_var],customdata=anova_data['Index'],
                            name='', boxpoints='all', jitter=0,
                            whiskerwidth=0.2, marker_color = 'indianred',
                            marker_size=2, line_width=1))
                        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',})  
                        fig.update_layout(font=dict(size=14,),)
                        fig.update_layout(hoverlabel=dict(bgcolor="white", ))
                        fig.update_traces(hovertemplate=target_var+': %{y} <br> Index: %{customdata}') 
                        fig.update_layout(height=400,width=400)

                        # Histogram
                        fig1 = px.histogram(anova_data, height=400, x=target_var, color=clas_var)     
                        fig1.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
                        fig1.update_traces(opacity=0.35) 
                        fig1.update_layout(yaxis=dict(titlefont_size=14, tickfont_size=14,),)
                        fig1.update_layout(xaxis=dict(titlefont_size=14, tickfont_size=14,),)
                        fig1.update_layout(height=400,width=400)

                        a4,a5=st.columns(2)
                        with a4:
                            st.subheader("Boxplots")
                            st.plotly_chart(fig, use_container_width=True)
                        with a5:
                            st.subheader("Histograms")
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        # ANOVA calculation & plots
                        df_grouped=anova_data.groupby(clas_var)
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
                        
                        anova_summary=pd.DataFrame(index=df_grouped.mean().index,columns=["count", "mean" , "variance"])
                        anova_summary["count"]=df_grouped.count()
                        anova_summary["mean"]=df_grouped.mean()
                        anova_summary["variance"]=df_grouped.var()

                        st.subheader("Groups summary:")
                        st.table(anova_summary.style.format(precision=user_precision))
                        
                        
                        anova_table=pd.DataFrame({
                            "SS": [sqe_stat.values[0], sqr_stat.values[0], sq_tot.values[0]],
                            "DOF": [dof1, dof2.values[0], dof_tot.values[0]],
                            "MS": [mqe_stat.values[0], mqe_stat.values[0], ""],
                            "F": [F_stat.values[0], "", ""],
                            "p-value": [p_value[0], "", ""]},
                            index=["Between groups", "Within groups", "Total"],)
                        
                        st.subheader("ANOVA")
                        st.table(anova_table.style.format(precision=user_precision))

                        #Anova (OLS)
                        codes, uniques = pd.factorize(anova_data[clas_var])
                        ano_ols = sm.OLS(anova_data[target_var], sm.add_constant(codes))
                        ano_ols_output= ano_ols.fit()
                        residuals=anova_data[target_var]-ano_ols_output.fittedvalues

                        # Q-Q plot residuals
                        qq_plot_data = pd.DataFrame()
                        qq_plot_data["Theoretical quantiles"] = stats.probplot(residuals, dist="norm")[0][0]
                        qq_plot_data["StandResiduals"] = sorted((residuals - residuals.mean())/residuals.std())
                        qq_plot_data["Index"]=qq_plot_data.index                                                
                        qq_plot = px.scatter(data_frame=qq_plot_data, x="Theoretical quantiles", y="StandResiduals",
                            hover_data=["Theoretical quantiles", "StandResiduals", 'Index'], color_discrete_sequence=['rgba(77, 121, 169, 0.7)'])
                        qq_plot.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
                        qq_plot.update_layout(xaxis=dict(title="theoretical quantiles", titlefont_size=14, tickfont_size=14,),)
                        qq_plot.update_layout(yaxis=dict(title="stand. residuals", titlefont_size=14, tickfont_size=14,),)
                        qq_plot.update_layout(hoverlabel=dict(bgcolor="white", ))
                        qq_plot.update_layout(height=400,width=400)


                        # histogram - residuals
                        res_hist = px.histogram(residuals, histnorm='probability density',opacity=0.5,color_discrete_sequence=['indianred'] ,height=400)                    
                        res_hist.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
                        res_hist.layout.showlegend = False 
                        res_hist.update_layout(hoverlabel=dict(bgcolor="white", ))   
                        res_hist.update_layout(xaxis=dict(title="residuals", titlefont_size=14, tickfont_size=14,),)  
                        res_hist.update_layout(height=400,width=400)          

                        a4,a5=st.columns(2)
                        with a4:
                            st.subheader("Q-Q plot")
                            st.plotly_chart(qq_plot, use_container_width=True)
                        with a5:
                            st.subheader("Histogram")
                            st.plotly_chart(res_hist,use_container_width=True)
                    
                        output = BytesIO()
                        excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                        if anova_data_output==True:
                            anova_data.to_excel(excel_file, sheet_name="data")
                        anova_summary.to_excel(excel_file, sheet_name="Groups summary")
                        anova_table.to_excel(excel_file, sheet_name="ANOVA")
                        excel_file.close()
                        excel_file = output.getvalue()
                        b64 = base64.b64encode(excel_file)
                        dl_file_name = "ANOVA_univariate_" + df_name + "_"+ target_var + clas_var + ".xlsx"
                        st.markdown(
                            f"""
                        <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download ANOVA results</a>
                        """,
                        unsafe_allow_html=True)
                        st.write("")                       
                        st.write("")
                        st.write("")

                        # xls-File download link:
              
                
            else:
                st.error("ERROR: The target variable must be a numerical one!")


        # -------------------
        # Hypothesis testing
        #----------------------

        dev_expander_test = st.expander("Hypothesis testing", expanded = False)
        with dev_expander_test:

            #initialisation
            test_check=False
                        
            # Test selection
            test_sel = st.selectbox('Select the test', ['z-test','One sample location t-test', 'Two sample location t-test'])     

            #Variance assumption two sample t-test
            if test_sel=='Two sample location t-test':
                    test_variance_ass=st.selectbox('Select variance assumption', ['equal variance', 'unequal variance']) 
            
            # Test variable                       
            test_var = st.selectbox('Select the test variable', df.columns)
            
                        
            if df[test_var].dtypes=="int64" or df[test_var].dtypes=="float64":                    
                
                a4,a5=st.columns(2)                
                with a4:
                    if test_sel in ['z-test','One sample location t-test']:
                        test_sollwert=st.number_input('Enter sollwert/population mean',format="%.8f") 
                        test_check=True
                    if test_sel=='z-test':
                        with a5:
                            test_variance=st.number_input('Enter the population variance',format="%.8f") 
                    
            else:
                st.error("ERROR: The test variable must be a numerical one!")       
            
            if test_sel =='Two sample location t-test':                 
                
                group_var = st.selectbox('Select the sample info variable', df.columns)
               
                if df[group_var].unique().size>2:  
                    a4,a5=st.columns(2)    
                    with a4:  
                        st.write("")                      
                        st.warning('The variable ' + group_var + ' has more than two realisations! Choose another variable or reclassify ' + group_var+ '!')            
                        test_check=False

                    with a5: 
                        st.write("")       
                        st.write("") 
                        st.write("")            
                        test_group_var_reclas=st.checkbox('Reclassify '+group_var, value=False )        
                        test_check=False

                    if test_group_var_reclas==True:
                        test_check=True
                        
                        
                        if df[group_var].dtypes=="int64" or df[group_var].dtypes=="float64":                           

                            a4,a5=st.columns(2)
                            with a4: 
                                test_recl=st.selectbox('First group is where values are ...', options=['greater','greater or equal','smaller','smaller or equal', 'equal','between'])
                                   
                            with a5:                                
                                if test_recl=='between':
                                    test_recl_thr_1=st.number_input('Lower limit for the classification',format="%.8f")
                                    test_recl_thr_2=st.number_input('Upper limit for the classification',format="%.8f")
                                    #reclassify values:
                                    df['test_class'] = np.where(((df[group_var] > test_recl_thr_1) & (df[group_var] < test_recl_thr_2)),1,0)  
                                   
                                else:
                                    test_recl_thr_1=st.number_input('Specify the value',format="%.8f")
                                    
                                     #reclassify values:
                                    if test_recl=='greater':
                                       df['test_class'] = np.where((df[group_var] > test_recl_thr_1),1,0) 
                                        
                                    elif test_recl=='greater or equal':
                                        df['test_class'] = np.where((df[group_var] >= test_recl_thr_1),1,0) 
                                        
                                    elif test_recl=='smaller':
                                        df['test_class'] = np.where((df[group_var] < test_recl_thr_1),1,0) 
                                        
                                    elif test_recl=='smaller or equal':
                                        df['test_class'] = np.where((df[group_var] <= test_recl_thr_1),1,0) 
                                       
                                    elif test_recl=='equal':
                                        df['test_class'] = np.where((df[group_var] == test_recl_thr_1),1,0) 
                                        
                                         

                        else:
                            a4,a5=st.columns(2)
                            with a4:
                                test_recl=st.selectbox('First group is where values are ...', options=(df[group_var]).unique())
                                df['test_class'] = np.where((df[group_var] == test_recl),1,0) 
                                test_check=True
             
            st.write("")    
            if test_check==True:
                st.write("")            
                test_data_output=st.checkbox("Include data from hypothesis testing in the output file", value=False)
                st.write("") 

                run_test = st.button("Press to perform hypothesis testing") 
                st.write("") 
                st.write("")

                if run_test:
                                        
                    #'z-test'
                    if test_sel=='z-test':
                        test_dataset=df[test_var].dropna()
                        
                        st.subheader("**z-test**")
                        st.write("")
                        st.write('Investigated variable: ' + test_var)
                        st.write('Sollwert/population mean = ' + str(test_sollwert))
                        st.write('Population variance = ' + str(test_variance))
                        st.write('Sample size = ' + str(len(test_dataset)))
                        st.write('Sample mean = ' + str(test_dataset.mean().round(user_precision)))
                        st.write('Sample std. = ' + str(test_dataset.std().round(user_precision)))
                       
                        z_val=(test_dataset.mean()-test_sollwert)/(np.sqrt(test_variance)/np.sqrt(len(test_dataset)))
                        p_value_one_sided = scipy.stats.norm.sf(abs(z_val)) 
                        p_value_two_sided = scipy.stats.norm.sf(abs(z_val))*2 
                        
                        a4,a5=st.columns(2)
                        with a4:
                            st.write("")
                            st.write('**Test statistics**')
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write('z = ' + str(z_val.round(user_precision)))
                            st.write('p-value (one sided) = ' + str(p_value_one_sided.round(user_precision)))
                            st.write('p-value (two sided) = ' + str(p_value_two_sided.round(user_precision))) 

                        with a5:
                            st.write("")
                            # Distribution plot
                            st.write('**St.normal distribution**') 
                            
                            x_var = np.linspace(norm.ppf(0.00001),norm.ppf(0.99999), 100)
                            y_var=norm.pdf(x_var)
                            
                            fig = px.line(x=x_var, y=y_var, color_discrete_sequence=['rgba(55, 126, 184, 0.7)'], height=300)
                            fig.add_vline(x=z_val, line_width=3, line_dash="dash", line_color='indianred')
                            fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
                            fig.update_layout(xaxis=dict(title='z', titlefont_size=12, tickfont_size=14,),)
                            fig.update_layout(yaxis=dict(title='pdf', titlefont_size=12, tickfont_size=14,),)
                            st.plotly_chart(fig,use_container_width=True)  
                        
                        
                        
                        z_stats = pd.DataFrame(index = ["Tested variable", "Sollwert", "Population variance", "n", "sample mean", "sample std", "z", "p-value - one sided","p-value - two sided"], columns = ["z-test"])
                        z_stats.loc["Tested variable"]=test_var
                        z_stats.loc["Sollwert"]=test_sollwert
                        z_stats.loc["Population variance"]=test_variance
                        z_stats.loc["n"]=len(test_dataset)
                        z_stats.loc["sample mean"]=test_dataset.mean()
                        z_stats.loc["sample std"]=test_dataset.std()
                        z_stats.loc["z"]=z_val
                        z_stats.loc["p-value - one sided"]=p_value_one_sided
                        z_stats.loc["p-value - two sided"]=p_value_two_sided

                        st.write("")                 
                        st.write("")
                        # Download link 
                        output = BytesIO()
                        excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                        if test_data_output==True:
                            test_dataset.to_excel(excel_file, sheet_name="data")    
                        z_stats.to_excel(excel_file, sheet_name="z-test")
                        excel_file.close()
                        excel_file = output.getvalue()
                        b64 = base64.b64encode(excel_file)
                        dl_file_name = "z-test_" + df_name + test_var+ ".xlsx"
                        st.markdown(
                            f"""
                        <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download z-test results</a>
                        """,
                        unsafe_allow_html=True)
                        st.write("")    
                        st.write("") 
                        st.write("") 



                    # 'One sample location t-test'
                    elif test_sel=='One sample location t-test':

                        test_dataset=df[test_var].dropna() 

                        st.subheader("One sample location t-test")
                        st.write("")

                        st.write('Investigated variable: ' + test_var)
                        st.write('Sollwert/population mean = ' + str(test_sollwert))                        
                        st.write('Sample size = ' + str(len(test_dataset)))
                        st.write('Sample mean = ' + str(test_dataset.mean().round(user_precision)))
                        st.write('Sample std. = ' + str(test_dataset.std().round(user_precision)))
                        n_sqr=np.sqrt(len(test_dataset))
                            
                        a4,a5=st.columns(2)
                        with a4:
                            st.write("")
                            st.write("**Test statistics**")
                            st.write("")
                            st.write("")
                            st.write("")
                            t_val=(test_dataset.mean()-test_sollwert)/(test_dataset.std()/n_sqr)
                            st.write('t = ' + str(t_val.round(user_precision)))
                            test_dof=len(test_dataset)-1
                            st.write('DOF = ' + str(test_dof))
                            p_value_one_sided = scipy.stats.t.sf(abs(t_val),len(test_dataset)-1) 
                            st.write('p-value (one sided) = ' + str(p_value_one_sided.round(user_precision)))
                            p_value_two_sided = scipy.stats.t.sf(abs(t_val),len(test_dataset)-1)*2                         
                            st.write('p-value (two sided) = ' + str(p_value_two_sided.round(user_precision))) 

                            t_stats = pd.DataFrame(index = ["Tested variable", "Sollwert",  "n", "sample mean","Sample std", "t", "DOF","p-value - one sided","p-value - two sided"], columns = ["t-test"])
                            t_stats.loc["Tested variable"]=test_var
                            t_stats.loc["Sollwert"]=test_sollwert
                            t_stats.loc["Sample std"]=test_dataset.std()
                            t_stats.loc["n"]=len(test_dataset)
                            t_stats.loc["sample mean"]=test_dataset.mean()
                            t_stats.loc["t"]=t_val
                            t_stats.loc["DOF"]=test_dof
                            t_stats.loc["p-value - one sided"]=p_value_one_sided
                            t_stats.loc["p-value - two sided"]=p_value_two_sided

                        with a5:
                            st.write("")
                            # Distribution plot
                            st.write('**t-distribution** '+ '(DOF = ' + str(test_dof)+')') 
                            
                            x_var = np.linspace(t.ppf(0.00001, test_dof),t.ppf(0.99999, test_dof), 100)
                            y_var=t.pdf(x_var, test_dof)
                            
                            fig = px.line(x=x_var, y=y_var, color_discrete_sequence=['rgba(55, 126, 184, 0.7)'], height=300)
                            fig.add_vline(x=t_val, line_width=3, line_dash="dash", line_color='indianred')
                            fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
                            fig.update_layout(xaxis=dict(title='t', titlefont_size=12, tickfont_size=14,),)
                            fig.update_layout(yaxis=dict(title='pdf', titlefont_size=12, tickfont_size=14,),)
                            st.plotly_chart(fig,use_container_width=True)     


                        st.write("")                 
                        st.write("")
                        # Download link
                        output = BytesIO()
                        excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                        if test_data_output==True:
                            test_dataset.to_excel(excel_file, sheet_name="data")    
                        t_stats.to_excel(excel_file, sheet_name="t-test")
                        excel_file.close()
                        excel_file = output.getvalue()
                        b64 = base64.b64encode(excel_file)
                        dl_file_name = "t-test_" + df_name + test_var + ".xlsx"
                        st.markdown(
                            f"""
                        <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download t-test results</a>
                        """,
                        unsafe_allow_html=True)
                        st.write("")    
                        st.write("") 
                        st.write("") 



                    #  'Two sample location t-test'
                    elif test_sel=='Two sample location t-test':
                        st.subheader("Two sample location t-test")
                        st.write("")                      
                       
                        if test_var==group_var:
                            test_dataset=df[[test_var,'test_class']].dropna()                            
                        else:                            
                            test_dataset=df[[test_var,group_var,'test_class']].dropna()   
                              
                        #identifying two data samples
                        test_sample_0 =test_dataset[test_dataset['test_class']==1]
                        test_sample_1 =test_dataset[test_dataset['test_class']==0]
                       
                        st.write('Investigated variable: ' + test_var)
                        st.write('Group info: ' + group_var)                                              
                        st.write('Variance assumption = ' + test_variance_ass )

                        st.write("")
                        st.write("")                    
                        a4,a5=st.columns(2)
                        with a4:
                            st.write('**1. sample**')
                            st.write('Sample size = ' + str(len(test_sample_0)))
                            st.write('Mean = ' + str(test_sample_0[test_var].mean().round(user_precision)))
                            st.write('Std. = ' + str(test_sample_0[test_var].std().round(user_precision)))
                        with a5:
                            st.write('**2. sample**')
                            st.write('Sample size = ' + str(len(test_sample_1)))
                            st.write('Mean = ' + str(test_sample_1[test_var].mean().round(user_precision)))
                            st.write('Std. = ' + str(test_sample_1[test_var].std().round(user_precision)))
                        
                        st.write("") 
                        st.write("") 

                        a4,a5=st.columns(2)
                        with a4:
                            st.write("")
                            st.write('**Test statistics**')
                            if test_variance_ass=="equal variance":
                                equal_var_bool=True
                            else:
                                equal_var_bool=False 

                            (t_val, p_value_less)=scipy.stats.ttest_ind(test_sample_0.loc[:,test_var].values,test_sample_1[test_var],equal_var=equal_var_bool, alternative='less')
                            (t_val, p_value_greater)=scipy.stats.ttest_ind(test_sample_0.loc[:,test_var].values,test_sample_1[test_var],equal_var=equal_var_bool, alternative='greater')
                            (t_val, p_value_two_sided)=scipy.stats.ttest_ind(test_sample_0.loc[:,test_var].values,test_sample_1[test_var],equal_var=equal_var_bool, alternative='two-sided')
                            
                            st.write('t = ' + str(t_val.round(user_precision))) 
                            if test_variance_ass=="equal variance":                      
                                test_dof=test_sample_0[test_var].size+test_sample_1[test_var].size-2    
                                
                            else:    
                                test_dof = (test_sample_0[test_var].var()/test_sample_0[test_var].size + test_sample_1[test_var].var()/test_sample_1[test_var].size)**2 / ((test_sample_0[test_var].var()/test_sample_0[test_var].size)**2 / (test_sample_0[test_var].size-1) + (test_sample_1[test_var].var()/test_sample_1[test_var].size)**2 / (test_sample_1[test_var].size-1))
                                test_dof =test_dof.round(user_precision)
                            st.write('DOF = ' + str(test_dof))
                            st.write('$H_{1}: \mu < \mu_{0}$ p-value = ' + str(p_value_less.round(user_precision)))
                            st.write('$H_{1}: \mu > \mu_{0}$ p-value = ' + str(p_value_greater.round(user_precision))) 
                            st.write('$H_{1}: \mu\, {=}\mathllap{/\,}\, \mu_{0}$ p-value = ' + str(p_value_two_sided.round(user_precision))) 

                        # plot pdf
                                             
                        with a5:
                            # Distribution plot
                            st.write('**t-distribution** '+ '(DOF = ' + str(test_dof)+')') 
                            x_var = np.linspace(t.ppf(0.00001, test_dof),t.ppf(0.99999, test_dof), 100)
                            y_var=t.pdf(x_var, test_dof)
                            
                            fig = px.line(x=x_var, y=y_var, color_discrete_sequence=['rgba(55, 126, 184, 0.7)'], height=300)
                            fig.add_vline(x=t_val, line_width=3, line_dash="dash", line_color='indianred')
                            fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
                            fig.update_layout(xaxis=dict(title='t', titlefont_size=12, tickfont_size=14,),)
                            fig.update_layout(yaxis=dict(title='pdf', titlefont_size=12, tickfont_size=14,),)
                            st.plotly_chart(fig,use_container_width=True)     
                            

                        t_stats = pd.DataFrame(index = ["Tested variable", "Group info", 'Variance assumption', "Sample 1 n", "Sample 1 mean","Sample 1 std", "Sample 2 n", "Sample 2 mean","Sample 2 std","t", "DOF" ,"p-value - less","p-value - greater","p-value - two sided"], columns = ["t-test"])
                        t_stats.loc["Tested variable"]=test_var
                        t_stats.loc["Group info"]=group_var
                        #t_stats.loc["Sollwert"]=test_sollwert
                        t_stats.loc["Variance assumption"]=test_variance_ass
                        
                        
                        t_stats.loc["Sample 1 n"]=len(test_sample_0[test_var])
                        t_stats.loc["Sample 1 mean"]=test_sample_0[test_var].mean()
                        t_stats.loc["Sample 1 std"]=test_sample_0[test_var].std()
                        t_stats.loc["Sample 2 n"]=len(test_sample_1[test_var])
                        t_stats.loc["Sample 2 mean"]=test_sample_1[test_var].mean()
                        t_stats.loc["Sample 2 std"]=test_sample_1[test_var].std()
                        t_stats.loc["t"]=t_val
                        t_stats.loc["DOF"]=test_dof
                        t_stats.loc["p-value - less"]=p_value_less
                        t_stats.loc["p-value - greater"]=p_value_greater
                        t_stats.loc["p-value - two sided"]=p_value_two_sided

                        #clean dataset
                        df=df.drop(['test_class'], axis=1)

                        st.write("")                 
                        st.write("")
                        # Download link 
                        output = BytesIO()
                        excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                        if test_data_output==True:
                            test_dataset.to_excel(excel_file, sheet_name="data")    
                        t_stats.to_excel(excel_file, sheet_name="t-test")
                        excel_file.close()
                        excel_file = output.getvalue()
                        b64 = base64.b64encode(excel_file)
                        dl_file_name = "t-test_two sample" + df_name + test_var + group_var + ".xlsx"
                        st.markdown(
                            f"""
                        <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download t-test results</a>
                        """,
                        unsafe_allow_html=True)
                        st.write("")    
                        st.write("") 
                        st.write("")



                    


        # --------------------
        # Fit theoretical dist
        #----------------------

        dev_expander_ft = st.expander("Distribution fitting", expanded = False)
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
                ft_var=st.selectbox("Select a variabe for dist. fitting",num_columns)
                ft_data=df[ft_var]
                
                #remove NAs if present   
                if np.where(ft_data.isnull())[0].size > 0:
                    ft_data = ft_data.dropna()
                    st.warning("WARNING: Your data set includes NAs. Rows with NAs are automatically deleted!")
           

                ft_selection=st.radio("Please choose if you would like to fit all distributions or a selection of distributions:", ["all (Note, this may take a while!)", "selection"], index=1)
                
                if ft_selection=='selection':
                    # Theoretical distribution:
                    ft_dist=st.multiselect("Select theoretical distributions for distribution fitting", dist_names, ['Normal','Lognormal','Weibull minimum'])
                else:
                    ft_dist=dist_names
                if len(ft_data)>4 and len(ft_data)<500:
                    iniBins=10*math.ceil(((len(ft_data))**0.5)/10)
                
                
                # Number of classes in the empirical distribution
                Nobins =iniBins
                ft_showOptions=st.checkbox("Additional settings", value = False)
                if ft_showOptions:                         
                    st.info('Sample value range for the variable "' +str(ft_var) + '":  \n min=' + str(min(df[ft_var]))+ '  \n max=' + str(max(df[ft_var])))
                    ft_low=st.number_input('Start fitting from the "' + str(ft_var) + '" value?')
                    ft_up=st.number_input('End fitting at the "' + str(ft_var) + '" value?')
                    Nobins = st.number_input('Number of classes for your histogram?',value=iniBins)
                
                ft_data_output=st.checkbox("Include data from distribution fitting in the output file", value=False)                    
        
                st.write("")
                run_fitting = st.button("Press to start the distribution fitting...")  
                st.write("") 
                st.write("")
                if run_fitting:

                    # status bar
                    st.info("Distribution fitting progress")

                    # Fit theoretical distribution to empirical data 
                    if(ft_low !=None and ft_up !=None):
                        ft_data=ft_data[ft_data.between(ft_low,ft_up, inclusive='both')] 
                    results, x_mid,xlower,xupper, min_ssd, max_p, clas_prob_res, best_name, best_dist, best_params = fc.fit_scipy_dist(ft_data, Nobins, ft_dist, ft_low, ft_up)
                    y, x = np.histogram(np.array(ft_data), bins=Nobins)
                    rel_freq=y/len(ft_data)
                    best_prob=clas_prob_res[min_ssd]
                    
                    #table output
                    st.success('Distribution fitting done!')
                    st.subheader("Goodness-of-fit results")
                    ft_results=results[['SSD', 'Chi-squared','DOF', 'p-value','Distribution']]
                    st.table(ft_results.style.format(precision=user_precision))
                    
                    
                    if sum(results['p-value']>0.05)>0:
                        st.info('The sum of squared diferences (SSD) is smallest for the "' +best_name + ' distribution"  \n The p-value of the $\chi^2$ statistics is largest for the "'+ max_p +' distribution"')
                    else:
                        st.info('The sum of squared diferences (SSD) is smallest for the "' +best_name + ' distribution"  \n The p-value of the $\chi^2$ statistics is for none of the distributions above 5%')
                    st.subheader("A comparison of relative frequencies")
                    rel_freq_comp=pd.DataFrame(xlower,columns=['Lower limit'])
                    rel_freq_comp['Upper limit']= xupper
                    rel_freq_comp['Class mean']=x_mid
                    rel_freq_comp['Rel. freq. (emp.)']= rel_freq
                    #rel_freq_comp['Rel. freq. (emp.)'] = pd.Series(["{0:.2f}%".format(val * 100) for val in rel_freq_comp['Rel. freq. (emp.)']], index = rel_freq_comp.index)
                    
                    rel_freq_comp['Rel. freq. ('+best_name +')']= best_prob
                    #rel_freq_comp['Rel. freq. ('+best_name +')'] = pd.Series(["{0:.2f}%".format(val * 100) for val in rel_freq_comp['Rel. freq. ('+best_name +')']], index = rel_freq_comp.index)
                
                    st.table(rel_freq_comp.style.format(precision=user_precision))
                            
                    # Plot results:
                    st.subheader("A comparison of relative frequencies (empirical vs. theoretical)")
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(x=x_mid, y=rel_freq, name='empirical',marker_color = 'indianred',opacity=0.5))
                    fig.add_trace(go.Bar(x=x_mid, y=best_prob, name=best_name,marker_color = 'rgb(26, 118, 255)',opacity=0.5))
                    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',})  
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("")
                    if sett_hints:
                        st.info(str(fc.learning_hints("fit_hints")))
                        st.write("")

                    # xls-File download link:
                    output = BytesIO()
                    excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                    if ft_data_output==True:
                        ft_data.to_excel(excel_file, sheet_name="data")
                    results[['SSD', 'Chi-squared','DOF', 'p-value','Distribution']].to_excel(excel_file, sheet_name="Goodness-of-fit results")
                    rel_freq_comp.to_excel(excel_file, sheet_name="frequency comparison")
                    excel_file.close()
                    excel_file = output.getvalue()
                    b64 = base64.b64encode(excel_file)
                    dl_file_name = "Distribution fitting_" + df_name +"_" + ft_var + ".xlsx"
                    st.markdown(
                        f"""
                    <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download fitting results</a>
                    """,
                    unsafe_allow_html=True)
                    st.write("")    
                    st.write("") 
                    st.write("")     
        # ---------------------
        # Corr. analysis
        #----------------------

        dev_expander_qa = st.expander("Correlation analysis", expanded = False)
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
                if transf_cols !=None:
                    #st.info('Please note, non-numerical variables are factorized to enable regression analysis! The factorized variables are: ' + str(transf_cols)) 
                    st.info('There are non-numerical variables in your dataset: ' + str(transf_cols)+ '. These will NOT be considered in the correlation analysis!') 
            #listOfAllColumns = df_cor.columns.to_list()   
            listOfAllColumns = non_trans_cols.to_list()      
            cor_sel_var=st.multiselect("Select variabes for correlation analysis",listOfAllColumns, listOfAllColumns)
            
            #remove NAs if present   
            if np.where(df_cor[cor_sel_var].isnull())[0].size > 0:
                df_cor = df_cor[cor_sel_var].dropna()
                st.warning("WARNING: Your data set includes NAs. Rows with NAs are automatically deleted!")
          
            cor_methods=['Pearson', 'Kendall', 'Spearman']
            cor_method=st.selectbox("Select the method",cor_methods)
            if cor_method=='Pearson':
                sel_method='pearson'
            elif cor_method=='Kendall':
                sel_method='kendall'
            else:
                sel_method='spearman'    

            if st.checkbox("Show data for correlation analysis", value = False):        
                st.write(df_cor[cor_sel_var].dropna())
            st.write("")
            st.write("") 

            #remove NAs if present   
            if np.where(df_cor[cor_sel_var].isnull())[0].size > 0:
                df_cor = df_cor[cor_sel_var].dropna()
                st.warning("WARNING: Your data set includes NAs. Rows with NAs are automatically deleted!")
        
            
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
                st.table((df_cor.corr(method=sel_method)).style.format(precision=user_precision))

                # xls-File download link:
                output = BytesIO()
                excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                df_cor.corr(method=sel_method).to_excel(excel_file, sheet_name="Correlation matrix")
                excel_file.close()
                excel_file = output.getvalue()
                b64 = base64.b64encode(excel_file)
                dl_file_name = "Correlation_" + df_name +"_" + sel_method + ".xlsx"
                st.markdown(
                    f"""
                <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download correlation table</a>
                """,
                unsafe_allow_html=True)
                st.write("")    
                st.write("") 
                st.write("")  

        #--------------------------------------
        # Regression analysis
        #--------------------------------------

        dev_expander_reg = st.expander("Regression techniques", expanded = False) 
        with dev_expander_reg:
            
            # initialisation:
            ra_techniques_names=['Simple Linear Regression', 'Linear-Log Regression', 'Log-Linear Regression', 'Log-Log Regression','Polynomial Regression']
            poly_order=2 # default for the polynomial regression
            num_columns=df.columns
            
            df_ra = df.copy()                       
            
            # check variable type
            for column in df_ra:            
                if not df_ra[column].dtypes in ('float', 'float64', 'int','int64'): 
                    num_columns=num_columns.drop(column) 
                    
            if len(num_columns)<2:
                st.error("ERROR: Your data is not suitable for regression analysis! Please try another dataset!")    
            else:
                # Variable selection:
                ra_Xvar=st.selectbox("Select the X variabe for regression analysis", num_columns)
                ra_Yvar=st.selectbox("Select the Y variabe for regression analysis",num_columns,index=1)

                #reduce the dataset to selected variables only
                df_ra=df_ra[[ra_Xvar,ra_Yvar]]
                #remove NAs if present   
                if np.where(df_ra.isnull())[0].size > 0:
                    df_ra = df_ra.dropna()
                    st.warning("WARNING: Your data set includes NAs. Rows with NAs are automatically deleted!")
          
                
                if ra_Xvar==ra_Yvar:
                    st.error("ERROR: Regressing a variable against itself doesn't make much sense!")             
                else:      
                    # regression techniques selection:
                    ra_selection=st.radio("Please choose if you would like to apply all regression techniques or a selection of techniques:", ["all", "selection"], index=1)
                    if ra_selection=='selection':
                        ra_tech=st.multiselect("Select regression techniques ", ra_techniques_names, ['Simple Linear Regression'])
                    else:
                        ra_tech=ra_techniques_names

                    
                    # Additional settings
                    #ra_showOptions=st.checkbox("Additional regression analysis settings?", value = False)
                    
                    if 'Polynomial Regression' in ra_tech:
                        poly_order=st.number_input('Specify the polynomial order for the polynomial regression',value=2, step=1)    
                    
                    ra_detailed_output=st.checkbox("Show detailed output per technique?", value = False)
                    ra_data_output=st.checkbox("Include data in the output file", value = False)

                    # specify the dataset for the regression analysis:
                    
                    X_ini=df_ra[ra_Xvar]
                    Y_ini=df_ra[ra_Yvar]
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
                            # xls-File download link:
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            if ra_data_output==True:
                                #write data to xls file
                                df_ra[[ra_Xvar,ra_Yvar]].to_excel(excel_file, sheet_name="data")

                            for reg_technique in ra_tech:                            
                                mlr_reg_inf, mlr_reg_stats, mlr_reg_anova, mlr_reg_coef,X_data, Y_data, Y_pred = fc.regression_models(X_ini, Y_ini, expl_var,reg_technique,poly_order)
                                # Model comparison
                                model_comparison.loc["R²"][reg_technique] = np.round_((mlr_reg_stats.loc["R²"]).Value,decimals=user_precision)
                                model_comparison.loc["Adj. R²"][reg_technique] = np.round_((mlr_reg_stats.loc["Adj. R²"]).Value,decimals=user_precision) 
                                model_comparison.loc["Log-likelihood"][reg_technique] = np.round_((mlr_reg_stats.loc["Log-likelihood"]).Value,decimals=user_precision) 
                                #model_comparison.loc["% VE"][reg_technique] =  r2_score(Y_data, Y_pred)
                                model_comparison.loc["MSE"][reg_technique] = np.round_(mean_squared_error(Y_data, Y_pred, squared = True),decimals=user_precision)
                                model_comparison.loc["RMSE"][reg_technique] = np.round_(mean_squared_error(Y_data, Y_pred, squared = False),decimals=user_precision)
                                model_comparison.loc["MAE"][reg_technique] = np.round_(mean_absolute_error(Y_data, Y_pred),decimals=user_precision)
                                model_comparison.loc["MaxErr"][reg_technique] = np.round_(max_error(Y_data, Y_pred),decimals=user_precision)
                                model_comparison.loc["EVRS"][reg_technique] = np.round_(explained_variance_score(Y_data, Y_pred),decimals=user_precision)
                                model_comparison.loc["SSR"][reg_technique] = np.round_(((Y_data-Y_pred)**2).sum(),decimals=user_precision)


                                # scatterplot with the initial data:
                                x_label=str(X_label_prefix[reg_technique])+str(ra_Xvar)
                                y_label=str(Y_label_prefix[reg_technique])+str(ra_Yvar)
                                reg_plot_data = pd.DataFrame()
                                
                                if reg_technique=='Polynomial Regression':
                                    reg_plot_data[ra_Xvar] = X_ini 
                                    
                                else:                                         
                                    reg_plot_data[ra_Xvar] = X_data
                                    
                                reg_plot_data["Index"] = reg_plot_data[ra_Xvar].index
                                reg_plot_data[ra_Yvar] = Y_data
                                reg_plot = alt.Chart(reg_plot_data,height=400).mark_circle(size=20).encode(
                                    x = alt.X(ra_Xvar, scale = alt.Scale(domain = [min(reg_plot_data[ra_Xvar]), max(reg_plot_data[ra_Xvar])]), axis = alt.Axis(title=x_label, titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y(ra_Yvar, scale = alt.Scale(domain = [min(min(reg_plot_data[ra_Yvar]),min(Y_pred)), max(max(reg_plot_data[ra_Yvar]),max(Y_pred))]), axis = alt.Axis(title=y_label,titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = [ra_Xvar,ra_Yvar,"Index"]
                                )
                                
                                # model fit plot 
                                line_plot_data = pd.DataFrame()
                                if reg_technique=='Polynomial Regression':
                                    line_plot_data[ra_Xvar] = X_ini                                     
                                else:       
                                    line_plot_data[ra_Xvar] = X_data
                                reg_technique
                                line_plot_data[reg_technique+'_'+ra_Yvar] = Y_pred

                                line_plot_data.to_excel(excel_file, sheet_name="Fit_"+reg_technique)
                                
                                line_plot_data[ra_Yvar] = Y_pred
                                line_plot_data["Index"]=line_plot_data[ra_Xvar].index    
                                

                                line_plot_0 = alt.Chart(line_plot_data,height=400).mark_line(size = 2, color = "darkred").encode( x=ra_Xvar, y=ra_Yvar)
                                
                                line_plot = alt.Chart(line_plot_data,height=400).mark_circle(size=1).mark_point(opacity=0, color='darkred').encode(
                                    x = alt.X(ra_Xvar, scale = alt.Scale(domain = [min(reg_plot_data[ra_Xvar]), max(reg_plot_data[ra_Xvar])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y(ra_Yvar, scale = alt.Scale(domain = [min(min(reg_plot_data[ra_Yvar]),min(Y_pred)), max(max(reg_plot_data[ra_Yvar]),max(Y_pred))]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = [alt.Tooltip(shorthand=ra_Xvar,title=x_label), alt.Tooltip(shorthand=ra_Yvar,title='Fit_'+y_label),"Index"]
                                )


                                # Q-Q plot residuals
                                qq_plot_data = pd.DataFrame()
                                residuals=Y_data-Y_pred
                                qq_plot_data["Theoretical quantiles"] = stats.probplot(residuals, dist="norm")[0][0]
                                qq_plot_data["StandResiduals"] = sorted((residuals - residuals.mean())/residuals.std())
                                qq_plot_data["Index"] = qq_plot_data["Theoretical quantiles"].index
                                qq_plot = alt.Chart(qq_plot_data,height=400).mark_circle(size=20).encode(
                                    x = alt.X("Theoretical quantiles", title = "theoretical quantiles", scale = alt.Scale(domain = [min(qq_plot_data["Theoretical quantiles"]), max(qq_plot_data["Theoretical quantiles"])]), axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    y = alt.Y("StandResiduals", title = "stand. residuals", axis = alt.Axis(titleFontSize = 12, labelFontSize = 11)),
                                    tooltip = ["StandResiduals", "Theoretical quantiles","Index"]
                                )
                                line = alt.Chart(
                                    pd.DataFrame({"Theoretical quantiles": [min(qq_plot_data["Theoretical quantiles"]), max(qq_plot_data["Theoretical quantiles"])], "StandResiduals": [min(qq_plot_data["Theoretical quantiles"]), max(qq_plot_data["Theoretical quantiles"])]})).mark_line(size = 2, color = "darkred").encode(
                                            alt.X("Theoretical quantiles"),
                                            alt.Y("StandResiduals"),
                                )
                                    
                                st.subheader(reg_technique)

                                if ra_detailed_output:
                                    #st.table(mlr_reg_stats)
                                    st.table(mlr_reg_anova.style.format(precision=user_precision))
                                    st.table(mlr_reg_coef.style.format(precision=user_precision))
 
                                    #write output to xls file
                                    mlr_reg_anova.to_excel(excel_file, sheet_name="Anova_" + reg_technique)
                                    mlr_reg_coef.to_excel(excel_file, sheet_name="Coef_" + reg_technique)                         
                                    
                                    a4,a5=st.columns(2)
                                    with a4:
                                        st.subheader("Regression plot")
                                        st.altair_chart(reg_plot +line_plot_0+line_plot, use_container_width=True) 
                                    with a5:
                                        st.subheader("Q-Q plot")
                                        st.altair_chart(qq_plot + line, use_container_width=True)
                                else: 
                                    a4,a5=st.columns(2)
                                    with a4:
                                        st.subheader("Regression plot")
                                        st.altair_chart(reg_plot +line_plot_0+line_plot, use_container_width=True) 
                                    with a5:
                                        st.subheader("Q-Q plot")
                                        st.altair_chart(qq_plot + line, use_container_width=True)   

                                progress += 1
                                ra_bar.progress(progress/n_techniques)
                            
                            st.subheader('Regression techniques summary') 
                            model_output=(model_comparison[ra_tech]).transpose()   
                            st.write(model_output) 
                            st.write("") 
                            st.write("")  

                            
                            model_output.to_excel(excel_file, sheet_name="Model comparison")
                            excel_file.close()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "Regression_" + df_name +"_" + ra_Xvar + ra_Yvar + ".xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download regression results</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("")    
                            st.write("") 
                            st.write("")  
  

        #------------------------------------------------------
        #-------------------------------------------------------


        #-------------------------------------
        # Contingency analysis
        #--------------------------------------
        
        dev_expander_conting = st.expander("Contingency tables and association measures", expanded = False)
        cont_check=False
        cont_unique_limit=30
        data_reclas=0
        with dev_expander_conting:
            if len(df.columns)>2:
                dfnew=df[df.columns[0:2]] 
                listOfColumns = dfnew.columns.to_list()
                listOfAllColumns = df.columns.to_list()
                st.info("Your dataset has more than 2 variables! In case you want to analyse multivariate data, please select 'Multivariate data' on the sidebar. In case you want to analyse uni- and bivariate data please select up to two variables to analyse right below...")
                
                ub_sel_var=st.multiselect("Please select up to 2 variables to analyse",listOfAllColumns, listOfColumns)
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
                cont_group=st.selectbox("You can try some of these options:", ['-','Reclassify my data','Use my data anyway' ]) 
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
                        a4,a5=st.columns(2) 
                        with a4:
                            st.subheader(str(cont_numerical[0]))
                            st.table(pd.DataFrame(data={'min': [min(df[cont_numerical[0]])], 'max': [max(df[cont_numerical[0]])]},index=['range']))
                            low0=st.number_input(str(cont_numerical[0]) + ': 1st class should start at?')
                            up0=st.number_input(str(cont_numerical[0]) + ': Max limit for your classes?')
                            noclass0= st.number_input(str(cont_numerical[0]) + ': ax number of classes? (Fully empty classes will be ignored!)',step=1)
                        
                        with a5:
                            st.subheader(str(cont_numerical[1]))
                            st.table(pd.DataFrame(data={'min': [min(df[cont_numerical[1]])], 'max': [max(df[cont_numerical[1]])]},index=['range']))
                            low1=st.number_input(str(cont_numerical[1]) + ': 1st class should start at?')
                            up1=st.number_input(str(cont_numerical[1]) + ': Max limit for your classes?')
                            noclass1= st.number_input(str(cont_numerical[1]) + ': Max number of classes? (Fully empty classes will be ignored!)',step=1)
                        
                    elif len(cont_numerical)==1 and len(cont_categs)==1:
                        data_reclas=1
                        cont_check=True 
                        a4,a5=st.columns(2) 
                        with a4:
                            st.subheader(str(cont_numerical[0]))
                            st.table( {'min': [min(df[cont_numerical[0]])], 'max': [max(df[cont_numerical[0]])]})
                            low0=st.number_input(str(cont_numerical[0]) + ': 1st class should start at?')
                            up0=st.number_input(str(cont_numerical[0]) + ': Max limit for your classes?')
                            noclass0= st.number_input(str(cont_numerical[0]) + ': Max number of classes? (Fully empty classes will be ignored!)')
                                        
                    else:           
                        st.info("Please try data reclassification outside of Staty as the sort of classification you might need is not yet implemented!")    
                
                
            else:
                cont_check=True

            # central part - contigency analysis
            if cont_check==True:

                #check NAs
                if np.where(df.isnull())[0].size > 0:
                    df = df.dropna()
                    st.warning("WARNING: Your data set includes NAs. Rows with NAs are automatically deleted!")
        
                # xls-File download link:
                output = BytesIO()
                excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                              

                cont_extra=st.checkbox("Show marginal frequencies", value = False)        
                
                if st.checkbox("Show data for contingency analysis", value = False):        
                    st.write(df)
                cont_data_output=st.checkbox("Include data for contingency analysis in the output file", value = False)
                
                #xls write
                if cont_data_output==True:
                    df.to_excel(excel_file, sheet_name="data")
                
                st.write("")            
                run_cont = st.button("Press to start data processing...")           
                st.write("")
                st.write("")    
             
                if run_cont:                    
                    cont_df=df    
                    if sett_hints:
                        st.info(str(fc.learning_hints("contingency_hints")))
                        st.write("")

                    if data_reclas==2:
                        step0=(up0-low0)/noclass0
                        lim_ser = pd.Series(np.arange(low0, up0, step0)) 
                        lim_ser=lim_ser.round(user_precision) 

                        for k in range(len(lim_ser)-1):
                            cont_df.loc[(cont_df[cont_numerical[0]]>lim_ser[k]) & (cont_df[cont_numerical[0]] <= lim_ser[k+1]), cont_numerical[0]] = lim_ser[k]
                        cont_df.loc[cont_df[cont_numerical[0]]>lim_ser[k+1], cont_numerical[0]] = '>'+ str(max(lim_ser))#+step0

                        step1=(up1-low1)/noclass1
                        lim_ser1 = pd.Series(np.arange(low1, up1, step1))
                        lim_ser1=lim_ser1.round(user_precision) 
                        for k in range(len(lim_ser1)-1):
                            cont_df.loc[(cont_df[cont_numerical[1]]>lim_ser1[k]) & (cont_df[cont_numerical[1]] <= lim_ser1[k+1]), cont_numerical[1]] = lim_ser1[k]
                        cont_df.loc[cont_df[cont_numerical[1]]>lim_ser1[k+1], cont_numerical[1]] = '>'+ str(max(lim_ser1))
                    elif data_reclas==1:
                        step0=(up0-low0)/noclass0
                        lim_ser = pd.Series(np.arange(low0, up0, step0))                         
                        lim_ser=lim_ser.round(user_precision)   

                       
                        for k in range(len(lim_ser)-1):                                                  
                            cont_df.loc[(cont_df[cont_numerical[0]]>lim_ser[k]) & (cont_df[cont_numerical[0]] <= lim_ser[k+1]), cont_numerical[0]] = lim_ser[k+1]
                        cont_df.loc[cont_df[cont_numerical[0]]>lim_ser[k+1], cont_numerical[0]] = '>'+ str(max(lim_ser))
                     
                    bivarite_table = pd.crosstab(index= cont_df.iloc[:,0], columns= cont_df.iloc[:,1] , margins=True, margins_name="Total")
                    stat, p, dof, expected = chi2_contingency(pd.crosstab(index= cont_df.iloc[:,0], columns= cont_df.iloc[:,1])) 
                        
                    st.subheader('Contingency table with absolute frequencies')    
                    st.table(bivarite_table)
                    
                    #xls output:
                    bivarite_table.to_excel(excel_file, sheet_name="absolute frequencies")

                    no_vals=bivarite_table.iloc[len(bivarite_table)-1,len(bivarite_table.columns)-1]
                    st.subheader('Contingency table with relative frequencies')
                    cont_rel=bivarite_table/no_vals
                    st.table(cont_rel.style.format(precision=user_precision))

                    #xls output:
                    cont_rel.to_excel(excel_file, sheet_name="relative frequencies")

                    if cont_extra:
                        st.subheader('Contingency table with marginal frequencies ('+ str(ub_sel_var[0])+')')
                        bivarite_table_marg0 =bivarite_table.iloc[:,:].div(bivarite_table.Total, axis=0)
                        st.table(bivarite_table_marg0.style.format(precision=user_precision))

                        #xls output:
                        bivarite_table_marg0.to_excel(excel_file, sheet_name="marginal"+ ub_sel_var[0][:min(len(ub_sel_var[0]),15)])

                        st.subheader('Contingency table with marginal frequencies ('+ str(ub_sel_var[1])+')')
                        bivarite_table_marg1 =bivarite_table.div(bivarite_table.iloc[-1])
                        st.table(bivarite_table_marg1.style.format(precision=user_precision))

                        #xls output:
                        bivarite_table_marg1.to_excel(excel_file, sheet_name="marginal"+ ub_sel_var[1][:min(len(ub_sel_var[1]),15)])

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
                        a4, a5= st.columns(2) 
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
                    st.table(dfexp.style.format(precision=user_precision))

                    #xls output:
                    dfexp.to_excel(excel_file, sheet_name="expected frequencies")
                            
                    st.subheader('Contingency stats') 
                    st.write('')
                    st.write('$\chi^2$ = ' + str(stat.round(user_precision)))
                    st.write('p-value = ' + str(p.round(user_precision)))

                    pearson_coef=(np.sqrt(stat/(stat+len(df)))).round(user_precision)  
                    st.write('Pearson contingency coefficient $K$ = ' + str(pearson_coef))

                    min_kl=min(len(bivarite_table)-1,len(bivarite_table.columns)-1)
                    K_max=(np.sqrt((min_kl-1)/min_kl)).round(user_precision)
                    st.write('$K_{max}$ = ' + str(K_max))
                        
                    pearson_cor=(pearson_coef/K_max).round(user_precision)
                    st.write('Corrected Pearson contingency coefficient $K_{cor}$ = ' + str(pearson_cor))
                    
                    cont_stats = pd.DataFrame(index = ["Chi2", "p-value", "K", "Kmax", "Kcor"], columns = ["stats"])
                    cont_stats.loc["Chi2"]=stat.round(user_precision)
                    cont_stats.loc["p-value"]=p.round(user_precision)
                    cont_stats.loc["K"]=pearson_coef.round(user_precision)
                    cont_stats.loc["Kmax"]=K_max.round(user_precision)
                    cont_stats.loc["Kcor"]=pearson_cor

                    st.write("")    
                    st.write("") 

                    
                    cont_stats.to_excel(excel_file, sheet_name="stats")
                    excel_file.close()
                    excel_file = output.getvalue()
                    b64 = base64.b64encode(excel_file)
                    dl_file_name = "Contingency_" + df_name +"_" + ub_sel_var[0] +ub_sel_var[1]+ ".xlsx"
                    st.markdown(
                        f"""
                    <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download contingency results</a>
                    """,
                    unsafe_allow_html=True)
                    st.write("")    
                    st.write("") 
                    st.write("") 
                    



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


