
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
import modelling as ml
import os
import altair as alt
import bs4
import requests
from collections import Counter
import streamlit.components.v1 as components
import yfinance as yf
import datetime
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from streamlit import caching
import SessionState
import sys
import platform

from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.web_scraping import WebScraping
from pysummarization.abstractabledoc.std_abstractor import StdAbstractor
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
from sklearn.feature_extraction.text import CountVectorizer
#----------------------------------------------------------------------------------------------

def app():

    # Clear cache
    caching.clear_cache()

    # Hide traceback in error messages (comment out for de-bugging)
    sys.tracebacklimit = 0

   
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

    #st.sidebar.subheader("Reset")
    reset_clicked = st.sidebar.button("Reset all your input")
    session_state = SessionState.get(id = 0)
    if reset_clicked:
        session_state.id = session_state.id + 1
    st.sidebar.markdown("")

    #++++++++++++++++++++++++++++++++++++++++++++
    # Text Mining
    #++++++++++++++++++++++++++++++++++++++++++++
    
    basic_text="Let STATY do text/web processing for you and start exploring your data stories right below... "
    
    st.header('**Web scraping and text data**')
    tw_meth = ['Text analysis','Web-Page summary','Stock data analysis']
    tw_classifier = st.selectbox('What analysis would you like to perform?', list('-')+tw_meth, key = session_state.id)
    
    if tw_classifier in tw_meth:
        st.write("")
        st.write("")
        st.header('**'+tw_classifier+'**')
        st.markdown(basic_text)
        
    if tw_classifier=='Web-Page summary': 

          
        user_path = st.text_input("What what web page should I summarize in five sentences for you?","https://en.wikipedia.org/wiki/Data_mining")
        
        run_models = st.button("Press to start the data processing...")
        if run_models:    
            # Pysummarization of a web page:
            def pysumMain(url):
                web_scrape = WebScraping()
                # Web-scraping:
                document = web_scrape.scrape(url)
                auto_abstractor = AutoAbstractor()
                auto_abstractor.tokenizable_doc = SimpleTokenizer()
                # Set delimiter for a sentence:
                auto_abstractor.delimiter_list = [".", "\n"]
            
                abstractable_doc = TopNRankAbstractor()
                # Summarize a document:
                result_dict = auto_abstractor.summarize(document, abstractable_doc)
            
                # Set the limit for the number of output sentences:
                limit = 5
                i = 1

                for sentence in result_dict["summarize_result"]:
                    st.write(sentence)
                    if i >= limit:
                        break
                    i += 1

            #user_path = st.text_input("What what web page should I summarize in five sentences for you?","https://en.wikipedia.org/wiki/Data_mining")
            
            if user_path !='':
                a1, a2 = st.beta_columns(2) 
                with a1:
                    st.subheader('Web page preview:')
                    st.text("")
                    components.iframe(user_path,width=None,height=500,scrolling=True)
                with a2:
                    st.subheader('Web page summary:')
                    st.text("")
                    pysumMain(user_path)

    if tw_classifier =='Stock data analysis':   
        # dwonload first the list of comanies in the S&P500 and DAX indices
        payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        first_table = payload[0]
        df = first_table
        symbols = df['Symbol'].values.tolist()
        company = df['Security'].values.tolist()
        sector = df['GICS Sector'].values.tolist()
        #sectors = set(sectors)

        payload1=pd.read_html('https://en.wikipedia.org/wiki/DAX')
        DAXtable = payload1[3]
        df=DAXtable
        DAXsymbols = df['Ticker symbol'].values.tolist()
        DAXSector = df['Prime Standard Sector'].values.tolist()
        DAXcompany= df['Company'].values.tolist()

        #Merge indices data
        symbols_all=symbols+DAXsymbols
        sector_all=sector+DAXSector 
        company_all=company+DAXcompany

        #ticker specification
        st.subheader('Stock data analysis')
        a3, a4 = st.beta_columns(2) 
        with a3:
            selected_stock = st.text_input("Enter a stock ticker symbol", "TSLA")
            symbols_all=list('-')+symbols_all
            selected_symbol = st.selectbox('You can add an additional stock for comparision...',symbols_all)
        with a4:
            today = datetime.date.today()
            last_year = today - datetime.timedelta(days=365)
            start_date = st.date_input('Select start date', last_year)
            end_date = st.date_input('Select end date', today)
            if start_date > end_date:
                st.error('ERROR: End date must fall after start date.')     
        
        dev_expander_perf = st.beta_expander("Check the stock performance")
        with dev_expander_perf:
            #get data for a selected ticker symbol:
            stock_data = yf.Ticker(selected_stock)
            stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
            
            add_stock_data = yf.Ticker(selected_symbol)
            add_stock_df = add_stock_data.history(period='1d', start=start_date, end=end_date)
            
            #print stock values
            if st.checkbox("Show stock data for " + selected_stock, value = True): 
                st.write(stock_df)
            
            if selected_symbol !="-":    
                if st.checkbox("Show stock data for " + selected_symbol, value = False): 
                    st.write(add_stock_df)
                comparision_check=st.checkbox('Compare '+ selected_stock + " & " + selected_symbol, value = True)
            
            #draw line chart with stock prices
            a5, a6 = st.beta_columns(2) 
            with a5:
                stock_para= st.selectbox('Select ' + selected_stock + " info to draw", stock_df.columns)
                if selected_symbol !="-":   
                    if comparision_check: 
                        st.subheader('Daily data comparision '+ selected_stock + " & " + selected_symbol)
                        
                        c1=selected_stock + " " + stock_para
                        c2=selected_symbol + " " + stock_para
                        c1_data=stock_df[[stock_para]]
                        c1_data.rename(columns={c1_data.columns[0]: c1 }, inplace = True)
                        c2_data=add_stock_df[[stock_para]]
                        c2_data.rename(columns={c2_data.columns[0]: c2 }, inplace = True)
                        stock_dataToplot=pd.concat([c1_data, c2_data], axis=1)
                        
                        #st.write(stock_dataToplot)
                        st.line_chart(stock_dataToplot)
                    else:
                        st.subheader(stock_para + " price for " + selected_stock + " (daily)")
                        stock_dataToplot=stock_df[stock_para]
                        st.line_chart(stock_dataToplot)   
                else:    
                    st.subheader(stock_para + " price for " + selected_stock + " (daily)")
                    stock_dataToplot=stock_df[stock_para]
                    st.line_chart(stock_dataToplot)

            with a6:
                stock_para2= st.selectbox('Select ' + selected_stock + " info to draw", stock_df.columns, index=3)
                if selected_symbol !="-":   
                    if comparision_check: 
                        st.subheader('Daily data comparision '+ selected_stock + " & " + selected_symbol)
                        
                        c3=selected_stock + " " + stock_para2
                        c4=selected_symbol + " " + stock_para2
                        c3_data=stock_df[[stock_para2]]
                        c3_data.rename(columns={c3_data.columns[0]: c3 }, inplace = True)
                        c4_data=add_stock_df[[stock_para2]]
                        c4_data.rename(columns={c4_data.columns[0]: c4 }, inplace = True)
                        stock_dataToplot2=pd.concat([c3_data, c4_data], axis=1)
                        
                        #st.write(stock_dataToplot)
                        st.line_chart(stock_dataToplot2)
                    else:
                        st.subheader(stock_para2 + " price for " + selected_stock + " (daily)")
                        stock_dataToplot2=stock_df[stock_para2]
                        st.line_chart(stock_dataToplot2)   
                else:    
                    st.subheader(stock_para2 + " price for " + selected_stock + " (daily)")
                    stock_dataToplot2=stock_df[stock_para2]
                    st.line_chart(stock_dataToplot2)        

       

    if tw_classifier=='Text analysis':
        run_text_OK=False
        text_cv = CountVectorizer()
        


        user_color=21  
        def random_color_func(user_col,word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
            h = int(user_color)
            s = int(100.0 * 255.0 / 255.0)
            l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)
            return "hsl({}, {}%, {}%)".format(h, s, l)    
        
        #specify the data source
        word_sl=st.radio('Select the data source for text analysis',['text input','web page'])  

        if word_sl=='text input':
            user_text=st.text_area('Please enter or copy your text here', value='', height=600, )

            if len(user_text)>0:  
                text_cv_fit=text_cv.fit_transform([user_text])
                wordcount= pd.DataFrame(text_cv_fit.toarray().sum(axis=0), index=text_cv.get_feature_names(),columns=["WordCount"])
                word_sorted=wordcount.sort_values(by=["WordCount"], ascending=False)

                if st.checkbox('Show a word count for your text', value = False): 
                    st.write(word_sorted) 
                         
                word_stopwords=st.multiselect("Remove words from text", word_sorted.index.tolist(),word_sorted.index[0:min(10,len(word_sorted.index))].tolist())

                # specify color options for the WordCloud (user selection)
                color_options= pd.DataFrame(np.array([[21, 120, 12, 240, 30]]), 
                columns=['orange', 'green', 'red','blue','brown'])
                user_color_name=st.selectbox('Select the main color of your WordCloud',color_options.columns)
                user_color=color_options[user_color_name]

                run_text_OK = True
                
       
        
        elif word_sl=='web page':
            user_path_wp = st.text_input("What web page should I analyse?","https://en.wikipedia.org/wiki/Data_mining")
            
            if user_path_wp !='':

                web_scrape = WebScraping()
                user_text = web_scrape.scrape(user_path_wp)
                
                text_cv_fit=text_cv.fit_transform([user_text])
                wordcount= pd.DataFrame(text_cv_fit.toarray().sum(axis=0), index=text_cv.get_feature_names(),columns=["WordCount"])
                word_sorted=wordcount.sort_values(by=["WordCount"], ascending=False)


                if st.checkbox('Show a word count', value = False): 
                    st.write(word_sorted)                                    
           
                word_stopwords=st.multiselect("Remove words from text", word_sorted.index.tolist(),word_sorted.index[1:min(10,len(word_sorted.index))].tolist())
                # specify color options for the WordCloud (user selection)
                color_options= pd.DataFrame(np.array([[21, 120, 12, 240, 30]]), 
                columns=['orange', 'green', 'red','blue','brown'])
                user_color_name=st.selectbox('Select the main color of your WordCloud',color_options.columns)
                user_color=color_options[user_color_name]
                run_text_OK=True
        
        if run_text_OK==True:
            run_text = st.button("Press to start the data processing...")
                

            if run_text:
                st.write("")
                st.write("")

                # Word frequency
                st.subheader('Word frequency') 
                text_cv = CountVectorizer(stop_words=set(word_stopwords))
                text_cv_fit=text_cv.fit_transform([user_text])
                wordcount= pd.DataFrame(text_cv_fit.toarray().sum(axis=0), index=text_cv.get_feature_names(),columns=["WordCount"])
                word_sorted=wordcount.sort_values(by=["WordCount"], ascending=False)
                             
                st.write(word_sorted) 

                #Draw WordCloud
                wordcloud = WordCloud(background_color="white",
                    contour_color="white",max_words=100,stopwords=word_stopwords,
                    width=600,height=400,color_func=random_color_func).generate(user_text)  
                fig_text, ax = plt.subplots()
                ax=plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                
                st.subheader('WordCloud')
                st.pyplot(fig_text)            
                            
            
    
