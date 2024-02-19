import streamlit as st
import pandas as pd
import numpy as np
import elements as el
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
from streamlit_js_eval import streamlit_js_eval


def app():

    # Clear cache
    st.runtime.legacy_caching.clear_cache()

    # Hide traceback in error messages (comment out for de-bugging)
    #sys.tracebacklimit = 0

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
                if st.checkbox("Show data description", value = False):          
                    st.markdown("**Data source:**")
                    st.markdown("The data come from Box & Jenkins (1970), but we use the version that is integrated in the R package ['astsa'](https://www.stat.pitt.edu/stoffer/tsa4/ ) which is a companion to the book ['Time Series Analysis and Its Applications'](https://www.springer.com/de/book/9783319524511) by Shumway & Stoffer's (2017)  .")
                                       
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
            if st.checkbox("Show raw time series data", value = False):      
                #st.dataframe(df.style.apply(lambda x: ["background-color: #ffe5e5" if (not pd.isna(df_summary_mq_full.loc["1%-Q"][i]) and df_summary_vt_cat[i] == "numeric" and (v <= df_summary_mq_full.loc["1%-Q"][i] or v >= df_summary_mq_full.loc["99%-Q"][i]) or pd.isna(v)) else "" for i, v in enumerate(x)], axis = 1))
                st.write(df)
                st.write("Data shape: ", n_rows,  " rows and ", n_cols, " columns")
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
                a7, a8 = st.columns(2)
                with a7:
                    st.table(df_summary["Variable types"])
            # Show summary statistics (raw data)
            if st.checkbox('Show summary statistics (raw data)', value = False ): 
                #st.write(df_summary["ALL"])
                df_datasumstat=df_summary["ALL"]

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

                #dfStyler = df_datasumstat.style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector = 'th', props=[('text-align', 'left')])]) 
                a7, a8 = st.columns(2)
                with a7:
                    st.table(df_datasumstat)
                    if fc.get_mode(df).loc["n_unique"].any():
                        st.caption("** Mode is not unique.")

        #---------------------------------
        # DATA PROCESSING       
        #---------------------------------
        (df,sb_DM_dTrans_ohe, sb_DM_dTrans_log, sb_DM_dTrans_sqrt, sb_DM_dTrans_square, sb_DM_dTrans_cent, sb_DM_dTrans_stand, sb_DM_dTrans_norm, sb_DM_dTrans_numCat, sb_DM_dTrans_mult, sb_DM_dTrans_div )=el.data_processing(df_name,df, n_rows,n_cols,sett_hints, user_precision,in_wid_change)
        
                 
    
    #--------------------------------------------------
    #--------------------------------------------------
    # Time-series data
    #---------------------------------------------------
    # initialisation
    st_dif_order=1
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
                ts_var=st.selectbox('Select the variable for time-series analysis and modelling', list(num_cols))
                #ts_exo=st.selectbox('Select exogenous variables for your model', list(num_cols))
    
            with a5:
                ts_time=st.selectbox('Select the time info for your data',list(date_cols)+list(num_cols))
            
            #time series:
            ts=df[[ts_var,ts_time]]
            ts_show_ts=st.checkbox('Show time series data',value=False)

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
                

                st.write("")
                ts_expander_datavis = st.expander("Diagnosis plots and tests")
                with ts_expander_datavis:
                
                    st.write('**Time series pattern**')
                    
                    ts_pattern_sel=st.selectbox('Select the analysis type',['Fixed window statistics check','Simple moving window', 'Zoom in data'])
                
                    if ts_pattern_sel=='Fixed window statistics check':
                        
                        a4,a5=st.columns(2)  
                        time_list=list(ts.index) 
                        with a4: 
                            start_time=st.selectbox('Specify the window start',list(ts.index),index=0)
                        with a5:
                            end_time=st.selectbox('Specify the window end',list(ts.index),index=len(list(ts.index))-1)
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
            ts_expander_decomp = st.expander("Differencing, detrending and seasonal adjustment", expanded=True)
            with ts_expander_decomp:
                ts_decomp = st.selectbox("Specify your time series differencing and decomposition preferences:", 
                    ["n-order differences", "detrending", "seasonal adjustment", "detrending & seasonal adjustment"])

                #----------------------------------------------------------
                # n-order differences
                #----------------------------------------------------------
                if ts_decomp=="n-order differences":   
                                    
                    st_dif_order=st.number_input('Specify the highest differencing order',min_value=1)
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
                    st_order_selection=st.selectbox('Select data for further modelling',adf_list)
                    if st_order_selection=='raw series':
                        ts_sel_data=ts[ts_var] 
                    else:       
                        ts_sel_data=ts[st_order_selection]

                    ts_show_ndifData=st.checkbox('Show selected data?', value=False)    
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
                    
                    ts_show_ndifData=st.checkbox('Show selected data?', value=False)   
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
 
                    ts_show_ndifData=st.checkbox('Show selected data', value=False) 
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
                    
                    ts_show_ndifData=st.checkbox('Show selected data',  value=False) 
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
                ts_alg = st.selectbox("Select modelling technique", ts_alg_list)
                
                st.write("")
                if sett_hints:
                    st.info(str(fc.learning_hints("ts_models_hints")))
                    st.write("")
                
                # Validation Settings                
                ts_modval= st.checkbox("Use model validation?", value=False)
                if ts_modval:
                    a4,a5=st.columns(2)
                    with a4:
                        # Select training/ test ratio 
                        ts_train = st.slider("Select training data size", 0.5, 0.95, 0.8)
                        
                ts_forecast= st.checkbox("Use model for forecast?", value=False)
                if ts_forecast:
                    a4,a5=st.columns(2)
                    with a4:
                        ts_forecast_no=st.number_input('Specify the number of forecast steps',min_value=1,value=2)
                
                
                ts_parametrisation= st.checkbox('Automatic parameterization of models?',value=True)
                
                st.write("")
                if ts_parametrisation==False:  
                    #initialisation:              
                    p,q,d,pp,dd,qq,s=0,0,0,0,0,0,0
                    ts_trend_spec='constant term (intercept)' 

                    a4,a5=st.columns(2)
                    if ts_alg=='AR':
                        with a4:
                            p = st.slider("Select order of the AR model (p)", 1, 30, 2)
                    elif ts_alg=='MA':
                        with a4:
                            q = st.slider("Select the MA 'window' size over your data (q)", 1, 15, 2)
                    elif ts_alg=='ARMA':
                        with a4:
                            p = st.slider("Select order of the AR model (p)", 0, 15, 2)
                            q = st.slider("Select the MA 'window' size over your data (q)", 0, 15, 2)   
                    elif ts_alg =='non-seasonal ARIMA':
                        with a4:
                            p = st.slider("Select order of the AR model (p)", 0, 15, 2)
                            d= st.slider("Select the degree of differencing (d)", 0, 15, 2)
                            q = st.slider("Select the MA 'window' size over your data (q)", 0, 15, 2)   
                    elif ts_alg=='seasonal ARIMA':
                        with a4:
                            p = st.slider("Select order of the AR model (p)", 0, 15, 0)
                            d= st.slider("Select the degree of differencing (d)", 0, 15, 2)
                            q = st.slider("Select the MA 'window' size over your data (q)", 0, 15, 0)   

                        with a5:
                            pp = st.slider("Select the AR order of the seasonal component (P)", 0, 15, 1)
                            dd= st.slider("Select the integration order (D)", 0, 30, 0)
                            qq = st.slider("Select the MA order of the seasonal component (Q)", 0, 15, 1) 
                            s = st.slider("Specify the periodicity (number of periods in season)", 0, 52, 2) 
                        
                    #additional settings for the model calibartion:
                    ts_man_para_add=st.checkbox('Show additional settings for manual model calibration?', value=False) 
                    if ts_man_para_add:                    
                        # trend specification                           
                        a4,a5=st.columns(2)
                        with a4:
                            ts_trend_spec=st.selectbox('Include a trend component in the model specification', ['No', 'constant term (intercept)', 'linear trend', 'second order polinomial'])
                            
                    
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
                    st_para_spec=st.checkbox('Show additional settings for automatic model parametrisation?', value=False)              
                    if st_para_spec: 
                        #Information criterion used to select the model
                        a4,a5=st.columns(2)
                        with a4:
                            ts_ic=st.selectbox('Select the information crtiteria to be used for the model selection', ['AIC', 'BIC', 'HQIC', 'OOB'])  
                        
                        #specification of the maximum valus for the model paramaters
                        a4,a5=st.columns(2)               
                        if ts_alg=='AR':
                            with a4:                                
                                maxp = st.slider("Maximum order of the AR model (max p)?", 1, 30, 5)
                        elif ts_alg=='MA':
                            with a4:                                
                                maxq = st.slider("Maximum 'window' size over your data (max q)?", 1, 15, 5)
                        elif ts_alg=='ARMA':
                            with a4:                                
                                maxp = st.slider("Maximum order of the AR model (max p)?", 0, 15, 2)
                                maxq = st.slider("Maximum MA 'window' size over your data (max q)?", 0, 15, 2)   
                        elif ts_alg =='non-seasonal ARIMA':
                            with a4:                               
                                maxp = st.slider("Maximum order of the AR model (max p)?", 0, 15, 5)
                                maxd= st.slider("Maximum degree of differencing (max d)?", 0, 15, 2)
                                maxq = st.slider("Maximum MA 'window' size over your data (max q)?", 0, 15, 5)   
                        elif ts_alg=='seasonal ARIMA':
                            with a4:                               
                                maxp = st.slider("Maximum order of the AR model (max p)?", 0, 15, 5)
                                maxd= st.slider("Maximum degree of differencing (max d)?", 0, 15, 2)
                                maxq = st.slider("Maximum MA 'window' size over your data (max q)?", 0, 15, 5)   

                            with a5:
                                maxpp = st.slider("Maximum AR order of the seasonal component (max P)", 0, 15, 2)
                                maxdd= st.slider("Maximum integration order (max D)", 0, 30, 1)
                                maxqq = st.slider("Maximum MA order of the seasonal component (max Q)", 0, 15, 2) 
                                s = st.slider("Specify the periodicity (number of periods in season)", 0, 52, 2) 
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
                    excel_file.close()
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
                           
                        
                   


                                    
