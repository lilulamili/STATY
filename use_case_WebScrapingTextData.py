####################
# IMPORT LIBRARIES #
####################

from selectors import BaseSelector
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
import sys
import platform
import re
import base64
import time
from io import BytesIO

from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.web_scraping import WebScraping
from pysummarization.abstractabledoc.std_abstractor import StdAbstractor
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
from sklearn.feature_extraction.text import CountVectorizer
from difflib import SequenceMatcher

#----------------------------------------------------------------------------------------------

def app():

    # Clear cache
    #st.legacy_caching.clear_cache()

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
    #-------------------------------------------------------------------------
    # RESET INPUT
    
    #Session state
    if 'key' not in st.session_state:
        st.session_state['key'] = 0
    reset_clicked = st.sidebar.button("Reset all your input")
    if reset_clicked:
        st.session_state['key'] = st.session_state['key'] + 1
    st.sidebar.markdown("")

   
    #------------------------------------------------------------------------------------------
    # SETTINGS

    settings_expander=st.sidebar.expander('Settings')
    with settings_expander:
        st.caption("**Precision**")
        user_precision=int(st.number_input('Number of digits after the decimal point',min_value=0,max_value=10,step=1,value=4, key = st.session_state['key']))
        #st.caption("**Help**")
        #sett_hints = st.checkbox('Show learning hints', value=False)
        st.caption("**Appearance**")
        sett_wide_mode = st.checkbox('Wide mode', value=False, key = st.session_state['key'])
        sett_theme = st.selectbox('Theme', ["Light", "Dark"], key = st.session_state['key'])
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
    fc.theme_func_dl_button()

    def cc():
        st.legacy_caching.clear_cache()
        st.session_state['load_data_button'] = None

    #++++++++++++++++++++++++++++++++++++++++++++
    # Text Mining and web-scraping
    #++++++++++++++++++++++++++++++++++++++++++++
    
    basic_text="Let STATY do text/web processing for you and start exploring your data stories right below... "
    
    st.header('**Web scraping and text data**')
    tw_meth = ['Text analysis','Web-Page summary', 'Financial analysis']
    tw_classifier = st.selectbox('What analysis would you like to perform?', list('-')+tw_meth, key = st.session_state['key'], on_change=cc)
    

    if tw_classifier in tw_meth:
        st.markdown("")
        st.markdown("")
        st.header('**'+tw_classifier+'**')
        st.markdown(basic_text)
#------------------------------------------------------------
# Text summarization
# -----------------------------------------------------------        
    if tw_classifier=='Web-Page summary': 
        
        # Clear cache
        st.legacy_caching.clear_cache()
          
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
                a1, a2 = st.columns(2) 
                with a1:
                    st.subheader('Web page preview:')
                    st.text("")
                    components.iframe(user_path,width=None,height=500,scrolling=True)
                with a2:
                    st.subheader('Web page summary:')
                    st.text("")
                    pysumMain(user_path)

    if tw_classifier =='Stock data analysis':  

        # Clear cache
        st.legacy_caching.clear_cache()

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
        a3, a4 = st.columns(2) 
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

        st.markdown("")
        add_data_show=st.checkbox("Get additional data (cashflow, balance sheet etc.)", value = False)
        st.markdown("")
       
        dev_expander_perf = st.expander("Stock performance")
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
            a5, a6 = st.columns(2) 
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
        if add_data_show:
            dev_expander_cf = st.expander("Cashflow")
            with dev_expander_cf:
                st.subheader(selected_stock)
                stock_data_cf = yf.Ticker(selected_stock).cashflow
                st.write(stock_data_cf)
                if selected_symbol !='-':
                    st.subheader(selected_symbol)
                    st.write(yf.Ticker(selected_symbol).cashflow)

            dev_expander_bs = st.expander("Balance sheet")
            with dev_expander_bs:
                st.subheader(selected_stock)
                stock_data = yf.Ticker(selected_stock)
                stock_data_fi = stock_data.balance_sheet
                st.write(stock_data_fi)
                if selected_symbol !='-':
                    st.subheader(selected_symbol)
                    st.write(yf.Ticker(selected_symbol).balance_sheet)

            dev_expander_fi = st.expander("Other financials")
            with dev_expander_fi:
                st.subheader(selected_stock)
                stock_data = yf.Ticker(selected_stock)
                stock_data_fi = stock_data.financials
                st.write(stock_data_fi) 
                if selected_symbol !='-':
                    st.subheader(selected_symbol)
                    st.write(yf.Ticker(selected_symbol).financials)   
            
            dev_expander_info = st.expander("Stock basic info")
            with dev_expander_info:
                st.subheader(selected_stock)
                stock_data = yf.Ticker(selected_stock)
                st.write(stock_data.info ['longBusinessSummary'])
                if selected_symbol !='-':
                    st.subheader(selected_symbol)
                    st.write(yf.Ticker(selected_symbol).info ['longBusinessSummary'])
# ----------------------------------------------------------------
# Text Mining
#-----------------------------------------------------------------
    if tw_classifier=='Text analysis':

        # Clear cache
        st.legacy_caching.clear_cache()
        
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
            user_text=st.text_area('Please enter or copy your text here', value='STATY  \n\n STATY is growing out of the effort to bring more data insights to university education across all disciplines of the natural and social sciences. It is motivated by the belief that fostering data literacy, creativity and critical thinking are more effective towards innovation, than bringing endless units of introduction to programming to students who find learning programming an overwhelming task. By providing easy access to the methods of classical statistics and machine learning, STATY’s approach is to inspire students to explore issues they are studying in the curriculum directly on real data, practice interpreting the results and check the source code to see how it is done or to improve the code. STATY can be used in the process of teaching and learning data science, demonstrations of theoretical concepts across various disciplines, active learning, promotion of teamwork, research and beyond.', height=600, key = st.session_state['key'] )
            
            st.write("")
            if len(user_text)>0:  
                run_text_OK = True             
        
        elif word_sl=='web page':
            user_path_wp = st.text_input("What web page should I analyse?","https://en.wikipedia.org/wiki/Data_mining", key = st.session_state['key'])
            st.write("")

            if user_path_wp !='':
                web_scrape = WebScraping()
                user_text = web_scrape.scrape(user_path_wp)
                run_text_OK = True
        
        if run_text_OK == True:              

            # Basic text processing:    
            text_cv_fit=text_cv.fit_transform([user_text])
            wordcount= pd.DataFrame(text_cv_fit.toarray().sum(axis=0), index=text_cv.get_feature_names(),columns=["Word count"])
            word_sorted=wordcount.sort_values(by=["Word count"], ascending=False)
                                         

            #Stop words handling:
            stopword_selection=st.selectbox("Select stop word option",["No stop words (use all words)","Manually select stop words", "Use a built-in list of stop words in German", "Use a built-in list of stop words in English", "Specify stop words"], index=3, key=st.session_state['key'])
            if stopword_selection=="No stop words (use all words)":
                word_stopwords=[] 
            elif stopword_selection=="Manually select stop words":
                word_stopwords=st.multiselect("Select stop words (words to be removed from the text)", word_sorted.index.tolist(),word_sorted.index[1:min(10,len(word_sorted.index))].tolist(), key = st.session_state['key'])
            elif stopword_selection=="Use a built-in list of stop words in German":
                word_stopwords=["a","ab","aber","abermaliges","abermals","abgerufen","abgerufene","abgerufener","abgerufenes","abgesehen","ach","acht","achte","achten","achter","achtes","aehnlich","aehnliche","aehnlichem","aehnlichen","aehnlicher","aehnliches","aehnlichste","aehnlichstem","aehnlichsten","aehnlichster","aehnlichstes","aeusserst","aeusserste","aeusserstem","aeussersten","aeusserster","aeusserstes","ag","ähnlich","ähnliche","ähnlichem","ähnlichen","ähnlicher","ähnliches","ähnlichst","ähnlichste","ähnlichstem","ähnlichsten","ähnlichster","ähnlichstes","alle","allein","alleine","allem","allemal","allen","allenfalls","allenthalben","aller","allerdings","allerlei","alles","allesamt","allg","allg.","allgemein","allgemeine","allgemeinem","allgemeinen","allgemeiner","allgemeines","allgemeinste","allgemeinstem","allgemeinsten","allgemeinster","allgemeinstes","allmählich","allzeit","allzu","als","alsbald","also","am","an","and","andauernd","andauernde","andauerndem","andauernden","andauernder","andauerndes","ander","andere","anderem","anderen","anderenfalls","anderer","andererseits","anderes","anderm","andern","andernfalls","anderr","anders","anderst","anderweitig","anderweitige","anderweitigem","anderweitigen","anderweitiger","anderweitiges","anerkannt","anerkannte","anerkannter","anerkanntes","anfangen","anfing","angefangen","angesetze","angesetzt","angesetzten","angesetzter","ans","anscheinend","ansetzen","ansonst","ansonsten","anstatt","anstelle","arbeiten","au","auch","auf","aufgehört","aufgrund","aufhören","aufhörte","aufzusuchen","augenscheinlich","augenscheinliche","augenscheinlichem","augenscheinlichen","augenscheinlicher","augenscheinliches","augenscheinlichst","augenscheinlichste","augenscheinlichstem","augenscheinlichsten","augenscheinlichster","augenscheinlichstes","aus","ausdrücken","ausdrücklich","ausdrückliche","ausdrücklichem","ausdrücklichen","ausdrücklicher","ausdrückliches","ausdrückt","ausdrückte","ausgenommen","ausgenommene","ausgenommenem","ausgenommenen","ausgenommener","ausgenommenes","ausgerechnet","ausgerechnete","ausgerechnetem","ausgerechneten","ausgerechneter","ausgerechnetes","ausnahmslos","ausnahmslose","ausnahmslosem","ausnahmslosen","ausnahmsloser","ausnahmsloses","außen","außer","ausser","außerdem","ausserdem","außerhalb","äusserst","äusserste","äusserstem","äussersten","äusserster","äusserstes","author","autor","b","baelde","bald","bälde","bearbeite","bearbeiten","bearbeitete","bearbeiteten","bedarf","bedürfen","bedurfte","been","befahl","befiehlt","befiehlte","befohlene","befohlens","befragen","befragte","befragten","befragter","begann","beginnen","begonnen","behalten","behielt","bei","beide","beidem","beiden","beider","beiderlei","beides","beim","beinahe","beisammen","beispiel","beispielsweise","beitragen","beitrugen","bekannt","bekannte","bekannter","bekanntlich","bekanntliche","bekanntlichem","bekanntlichen","bekanntlicher","bekanntliches","bekennen","benutzt","bereits","berichten","berichtet","berichtete","berichteten","besonders","besser","bessere","besserem","besseren","besserer","besseres","bestehen","besteht","besten","bestenfalls","bestimmt","bestimmte","bestimmtem","bestimmten","bestimmter","bestimmtes","beträchtlich","beträchtliche","beträchtlichem","beträchtlichen","beträchtlicher","beträchtliches","betraechtlich","betraechtliche","betraechtlichem","betraechtlichen","betraechtlicher","betraechtliches","betreffend","betreffende","betreffendem","betreffenden","betreffender","betreffendes","bevor","bez","bez.","bezgl","bezgl.","bezueglich","bezüglich","bietet","bin","bis","bisher","bisherige","bisherigem","bisherigen","bisheriger","bisheriges","bislang","bisschen","bist","bitte","bleiben","bleibt","blieb","bloss","böden","boeden","brachte","brachten","brauchen","braucht","bräuchte","bringen","bsp","bsp.","bspw","bspw.","bzw","bzw.","c","ca","ca.","circa","d","d.h","da","dabei","dadurch","dafuer","dafür","dagegen","daher","dahin","dahingehend","dahingehende","dahingehendem","dahingehenden","dahingehender","dahingehendes","dahinter","damalige","damaligem","damaligen","damaliger","damaliges","damals","damit","danach","daneben","dank","danke","danken","dann","dannen","daran","darauf","daraus","darf","darfst","darin","darüber","darüberhinaus","darueber","darueberhinaus","darum","darunter","das","dasein","daselbst","daß","dass","dasselbe","Dat","davon","davor","dazu","dazwischen","dein","deine","deinem","deinen","deiner","deines","dem","dementsprechend","demgegenüber","demgegenueber","demgemaess","demgemäß","demgemäss","demnach","demselben","demzufolge","den","denen","denkbar","denkbare","denkbarem","denkbaren","denkbarer","denkbares","denn","dennoch","denselben","der","derart","derartig","derartige","derartigem","derartigen","derartiger","derem","deren","derer","derjenige","derjenigen","dermaßen","dermassen","derselbe","derselben","derzeit","derzeitig","derzeitige","derzeitigem","derzeitigen","derzeitiges","des","deshalb","desselben","dessen","dessenungeachtet","desto","desungeachtet","deswegen","dich","die","diejenige","diejenigen","dies","diese","dieselbe","dieselben","diesem","diesen","dieser","dieses","diesseitig","diesseitige","diesseitigem","diesseitigen","diesseitiger","diesseitiges","diesseits","dinge","dir","direkt","direkte","direkten","direkter","doch","doppelt","dort","dorther","dorthin","dran","drauf","drei","dreißig","drin","dritte","dritten","dritter","drittes","drüber","drueber","drum","drunter","du","duerfte","duerften","duerftest","duerftet","dunklen","durch","durchaus","durchweg","durchwegs","dürfen","dürft","durfte","dürfte","durften","dürften","durftest","dürftest","durftet","dürftet","e","eben","ebenfalls","ebenso","ect","ect.","ehe","eher","eheste","ehestem","ehesten","ehester","ehestes","ehrlich","ei","ei,","eigen","eigene","eigenem","eigenen","eigener","eigenes","eigenst","eigentlich","eigentliche","eigentlichem","eigentlichen","eigentlicher","eigentliches","ein","einander","einbaün","eine","einem","einen","einer","einerlei","einerseits","eines","einfach","einführen","einführte","einführten","eingesetzt","einig","einige","einigem","einigen","einiger","einigermaßen","einiges","einmal","einmalig","einmalige","einmaligem","einmaligen","einmaliger","einmaliges","eins","einseitig","einseitige","einseitigen","einseitiger","einst","einstmals","einzig","elf","empfunden","en","ende","endlich","entgegen","entlang","entsprechend","entsprechende","entsprechendem","entsprechenden","entsprechender","entsprechendes","entweder","er","ergänze","ergänzen","ergänzte","ergänzten","ergo","erhält","erhalten","erhielt","erhielten","erneut","ernst","eröffne","eröffnen","eröffnet","eröffnete","eröffnetes","erscheinen","erst","erste","erstem","ersten","erster","erstere","ersterem","ersteren","ersterer","ersteres","erstes","es","etc","etc.","etliche","etlichem","etlichen","etlicher","etliches","etwa","etwaige","etwas","euch","euer","eure","eurem","euren","eurer","eures","euretwegen","f","fall","falls","fand","fast","ferner","fertig","finde","finden","findest","findet","folgend","folgende","folgendem","folgenden","folgender","folgendermassen","folgendes","folglich","for","fordern","fordert","forderte","forderten","fort","fortsetzen","fortsetzt","fortsetzte","fortsetzten","fragte","frau","frei","freie","freier","freies","früher","fuer","fuers","fünf","fünfte","fünften","fünfter","fünftes","für","fürs","g","gab","gaenzlich","gaenzliche","gaenzlichem","gaenzlichen","gaenzlicher","gaenzliches","gängig","gängige","gängigen","gängiger","gängiges","ganz","ganze","ganzem","ganzen","ganzer","ganzes","gänzlich","gänzliche","gänzlichem","gänzlichen","gänzlicher","gänzliches","gar","gbr","geb","geben","geblieben","gebracht","gedurft","geehrt","geehrte","geehrten","geehrter","gefallen","gefälligst","gefällt","gefiel","gegeben","gegen","gegenüber","gegenueber","gehabt","gehalten","gehen","geht","gekannt","gekommen","gekonnt","gemacht","gemaess","gemäss","gemeinhin","gemocht","gemusst","genau","genommen","genug","gepriesener","gepriesenes","gerade","gern","gesagt","geschweige","gesehen","gestern","gestrige","getan","geteilt","geteilte","getragen","getrennt","gewesen","gewiss","gewisse","gewissem","gewissen","gewisser","gewissermaßen","gewisses","gewollt","geworden","ggf","ggf.","gib","gibt","gilt","ging","gleich","gleiche","gleichem","gleichen","gleicher","gleiches","gleichsam","gleichste","gleichstem","gleichsten","gleichster","gleichstes","gleichwohl","gleichzeitig","gleichzeitige","gleichzeitigem","gleichzeitigen","gleichzeitiger","gleichzeitiges","glücklicherweise","gluecklicherweise","gmbh","gott","gottseidank","gratulieren","gratuliert","gratulierte","groesstenteils","groß","gross","große","grosse","großen","grossen","großer","grosser","großes","grosses","grösstenteils","gruendlich","gründlich","gut","gute","guten","guter","gutes","h","hab","habe","haben","habt","haette","haeufig","haeufige","haeufigem","haeufigen","haeufiger","haeufigere","haeufigeren","haeufigerer","haeufigeres","halb","hallo","halten","hast","hat","hätt","hatte","hätte","hatten","hätten","hattest","hattet","häufig","häufige","häufigem","häufigen","häufiger","häufigere","häufigeren","häufigerer","häufigeres","heisst","hen","her","heraus","herein","herum","heute","heutige","heutigem","heutigen","heutiger","heutiges","hier","hierbei","hiermit","hiesige","hiesigem","hiesigen","hiesiger","hiesiges","hin","hindurch","hinein","hingegen","hinlanglich","hinlänglich","hinten","hintendran","hinter","hinterher","hinterm","hintern","hinunter","hoch","höchst","höchstens","http","hundert","i","ich","igitt","ihm","ihn","ihnen","ihr","ihre","ihrem","ihren","ihrer","ihres","ihretwegen","ihrige","ihrigen",
                "ihriges","im","immer","immerhin","immerwaehrend","immerwaehrende","immerwaehrendem","immerwaehrenden","immerwaehrender","immerwaehrendes","immerwährend","immerwährende","immerwährendem","immerwährenden","immerwährender","immerwährendes","immerzu","important","in","indem","indessen","Inf.","info","infolge","infolgedessen","information","innen","innerhalb","innerlich","ins","insbesondere","insgeheim","insgeheime","insgeheimer","insgesamt","insgesamte","insgesamter","insofern","inzwischen","irgend","irgendein","irgendeine","irgendeinem","irgendeiner","irgendeines","irgendetwas","irgendjemand","irgendjemandem","irgendwann","irgendwas","irgendwelche","irgendwen","irgendwenn","irgendwer","irgendwie","irgendwo","irgendwohin","ist","j","ja","jaehrig","jaehrige","jaehrigem","jaehrigen","jaehriger","jaehriges","jahr","jahre","jahren","jährig","jährige","jährigem","jährigen","jähriges","je","jede","jedem","jeden","jedenfalls","jeder","jederlei","jedermann","jedermanns","jedes","jedesmal","jedoch","jeglichem","jeglichen","jeglicher","jegliches","jemals","jemand","jemandem","jemanden","jemandes","jene","jenem","jenen","jener","jenes","jenseitig","jenseitigem","jenseitiger","jenseits","jetzt","jung","junge","jungem","jungen","junger","junges","k","kaeumlich","kam","kann","kannst","kaum","käumlich","kein","keine","keinem","keinen","keiner","keinerlei","keines","keineswegs","klar","klare","klaren","klares","klein","kleine","kleinen","kleiner","kleines","koennen","koennt","koennte","koennten","koenntest","koenntet","komme","kommen","kommt","konkret","konkrete","konkreten","konkreter","konkretes","könn","können","könnt","konnte","könnte","konnten","könnten","konntest","könntest","konntet","könntet","kuenftig","kuerzlich","kuerzlichst","künftig","kurz","kürzlich","kürzlichst","l","laengst","lag","lagen","lang","lange","langsam","längst","längstens","lassen","laut","lediglich","leer","legen","legte","legten","leicht","leide","leider","lesen","letze","letzte","letzten","letztendlich","letztens","letztere","letzterem","letzterer","letzteres","letztes","letztlich","lichten","lieber","liegt","liest","links","los","m","mache","machen","machst","macht","machte","machten","mag","magst","mahn","mal","man","manch","manche","manchem","manchen","mancher","mancherlei","mancherorts","manches","manchmal","mann","margin","massgebend","massgebende","massgebendem","massgebenden","massgebender","massgebendes","massgeblich","massgebliche","massgeblichem","massgeblichen","massgeblicher","mehr","mehrere","mehrerer","mehrfach","mehrmalig","mehrmaligem","mehrmaliger","mehrmaliges","mein","meine","meinem","meinen","meiner","meines","meinetwegen","meins","meist","meiste","meisten","meistens","meistenteils","mensch","menschen","meta","mich","mindestens","mir","mit","miteinander","mitgleich","mithin","mitnichten","mittel","mittels","mittelst","mitten","mittig","mitunter","mitwohl","mochte","möchte","mochten","möchten","möchtest","moechte","moeglich","moeglichst","moeglichste","moeglichstem","moeglichsten","moeglichster","mögen","möglich","mögliche","möglichen","möglicher","möglicherweise","möglichst","möglichste","möglichstem","möglichsten","möglichster","mögt","morgen","morgige","muessen","muesst","muesste","muß","muss","müssen","mußt","musst","müßt","müsst","musste","müsste","mussten","müssten","n","na","nach","nachdem","nacher","nachher","nachhinein","nächste","nacht","naechste","naemlich","nahm","nämlich","naturgemaess","naturgemäss","natürlich","ncht","neben","nebenan","nehmen","nein","neu","neue","neuem","neuen","neuer","neuerdings","neuerlich","neuerliche","neuerlichem","neuerlicher","neuerliches","neues","neulich","neun","neunte","neunten","neunter","neuntes","nicht","nichts","nichtsdestotrotz","nichtsdestoweniger","nie","niemals","niemand","niemandem","niemanden","niemandes","nimm","nimmer","nimmt","nirgends","nirgendwo","noch","noetigenfalls","nötigenfalls","nun","nur","nutzen","nutzt","nützt","nutzung","o","ob","oben","ober","oberen","oberer","oberhalb","oberste","obersten","oberster","obgleich","obs","obschon","obwohl","oder","oefter","oefters","off","offen","offenkundig","offenkundige","offenkundigem","offenkundigen","offenkundiger","offenkundiges","offensichtlich","offensichtliche","offensichtlichem","offensichtlichen","offensichtlicher","offensichtliches","oft","öfter","öfters","oftmals","ohne","ohnedies","online","ordnung","p","paar","partout","per","persoenlich","persoenliche","persoenlichem","persoenlicher","persoenliches","persönlich","persönliche","persönlicher","persönliches","pfui","ploetzlich","ploetzliche","ploetzlichem","ploetzlicher","ploetzliches","plötzlich","plötzliche","plötzlichem","plötzlicher","plötzliches","pro","q","quasi","r","reagiere","reagieren","reagiert","reagierte","recht","rechte","rechten","rechter","rechtes","rechts","regelmäßig","reichlich","reichliche","reichlichem","reichlichen","reichlicher","restlos","restlose","restlosem","restlosen","restloser","restloses","richtig","richtiggehend","richtiggehende","richtiggehendem","richtiggehenden","richtiggehender","richtiggehendes","rief","rund","rundheraus","rundum","runter","s","sa","sache","sage","sagen","sagt","sagte","sagten","sagtest","sagtet","sah","samt","sämtliche","sang","sangen","satt","sattsam","schätzen","schätzt","schätzte","schätzten","scheinbar","scheinen","schlecht","schlechter","schlicht","schlichtweg","schließlich","schluss","schlussendlich","schnell","schon","schreibe","schreiben","schreibens","schreiber","schwerlich","schwerliche","schwerlichem","schwerlichen","schwerlicher","schwerliches","schwierig","sechs","sechste","sechsten","sechster","sechstes","sect","sehe","sehen","sehr","sehrwohl","seht","sei","seid","seien","seiest","seiet","sein","seine","seinem","seinen","seiner","seines","seit","seitdem","seite","seiten","seither","selbe","selben","selber","selbst","selbstredend","selbstredende","selbstredendem","selbstredenden","selbstredender","selbstredendes","seltsamerweise","senke","senken","senkt","senkte","senkten","setzen","setzt","setzte","setzten","sich","sicher","sicherlich","sie","sieben","siebente","siebenten","siebenter","siebentes","siebte","siehe","sieht","sind","singen","singt","so","sobald","sodaß","soeben","sofern","sofort","sog","sogar","sogleich","solang","solange","solc","solchen","solch","solche","solchem","solchen","solcher","solches","soll","sollen","sollst","sollt","sollte","sollten","solltest","solltet","somit","sondern","sonst","sonstig","sonstige","sonstigem","sonstiger","sonstwo","sooft","soviel","soweit","sowie","sowieso","sowohl","später","spielen","startet","startete","starteten","startseite","statt","stattdessen","steht","steige","steigen","steigt","stellenweise","stellenweisem","stellenweisen","stets","stieg","stiegen","such","suche","suchen","t","tag","tage","tagen","tages","tat","tät","tatsächlich","tatsächlichen","tatsächlicher","tatsächliches","tatsaechlich","tatsaechlichen","tatsaechlicher","tatsaechliches","tausend","teil","teile","teilen","teilte","teilten","tel","tief","titel","toll","total","trage","tragen","trägt","tritt","trotzdem","trug","tun","tust","tut","txt","u","übel","über","überall","überallhin","überaus","überdies","überhaupt","überll","übermorgen","üblicherweise","übrig","übrigens","ueber","ueberall","ueberallhin","ueberaus","ueberdies","ueberhaupt","uebermorgen","ueblicherweise","uebrig","uebrigens","uhr","um","ums","umso","umstaendehalber","umständehalber","unbedingt","unbedingte","unbedingter","unbedingtes","und","unerhoert","unerhoerte","unerhoertem","unerhoerten","unerhoerter","unerhoertes","unerhört","unerhörte","unerhörtem","unerhörten","unerhörter","unerhörtes","ungefähr","ungemein","ungewoehnlich","ungewoehnliche","ungewoehnlichem","ungewoehnlichen","ungewoehnlicher","ungewoehnliches","ungewöhnlich","ungewöhnliche","ungewöhnlichem","ungewöhnlichen","ungewöhnlicher","ungewöhnliches","ungleich","ungleiche","ungleichem","ungleichen","ungleicher","ungleiches","unmassgeblich","unmassgebliche","unmassgeblichem","unmassgeblichen","unmassgeblicher","unmassgebliches","unmoeglich","unmoegliche","unmoeglichem","unmoeglichen","unmoeglicher","unmoegliches","unmöglich","unmögliche","unmöglichen","unmöglicher","unnötig","uns","unsaeglich","unsaegliche","unsaeglichem","unsaeglichen","unsaeglicher","unsaegliches","unsagbar","unsagbare","unsagbarem","unsagbaren","unsagbarer","unsagbares","unsäglich","unsägliche","unsäglichem","unsäglichen","unsäglicher","unsägliches","unse","unsem","unsen","unser","unsere","unserem","unseren","unserer","unseres","unserm","unses","unsre","unsrem","unsren","unsrer","unsres","unstreitig","unstreitige","unstreitigem","unstreitigen","unstreitiger","unstreitiges","unten","unter","unterbrach","unterbrechen","untere","unterem","unteres","unterhalb","unterste","unterster","unterstes","unwichtig","unzweifelhaft","unzweifelhafte","unzweifelhaftem","unzweifelhaften","unzweifelhafter","unzweifelhaftes","usw","usw.","v","vergangen","vergangene","vergangenen","vergangener","vergangenes","vermag","vermögen","vermutlich","vermutliche","vermutlichem","vermutlichen","vermutlicher","vermutliches","veröffentlichen","veröffentlicher","veröffentlicht","veröffentlichte","veröffentlichten","veröffentlichtes","verrate","verraten","verriet","verrieten","version","versorge","versorgen","versorgt","versorgte","versorgten","versorgtes","viel","viele","vielem","vielen","vieler","vielerlei","vieles","vielleicht","vielmalig","vielmals","vier","vierte","vierten","vierter","viertes","voellig","voellige","voelligem","voelligen","voelliger","voelliges","voelligst","vollends","völlig","völlige","völligem","völligen","völliger","völliges","völligst","vollstaendig","vollstaendige","vollstaendigem","vollstaendigen","vollstaendiger","vollstaendiges","vollständig","vollständige","vollständigem","vollständigen","vollständiger","vollständiges","vom","von","vor","voran","vorbei","vorgestern","vorher","vorherig","vorherige","vorherigem","vorheriger",
                "vorne","vorüber","vorueber","w","wachen","waehrend","waehrenddessen","waere","während","währenddem","währenddessen","wann","war","wär","wäre","waren","wären","warst","wart","warum","was","weder","weg","wegen","weil","weiß","weit","weiter","weitere","weiterem","weiteren","weiterer","weiteres","weiterhin","weitestgehend","weitestgehende","weitestgehendem","weitestgehenden","weitestgehender","weitestgehendes","weitgehend","weitgehende","weitgehendem","weitgehenden","weitgehender","weitgehendes","welche","welchem","welchen","welcher","welches","wem","wen","wenig","wenige","weniger","weniges","wenigstens","wenn","wenngleich","wer","werde","werden","werdet","weshalb","wessen","weswegen","wichtig","wie","wieder","wiederum","wieso","wieviel","wieviele","wievieler","wiewohl","will","willst","wir","wird","wirklich","wirklichem","wirklicher","wirkliches","wirst","wissen","wo","wobei","wodurch","wofuer","wofür","wogegen","woher","wohin","wohingegen","wohl","wohlgemerkt","wohlweislich","wolle","wollen","wollt","wollte","wollten","wolltest","wolltet","womit","womoeglich","womoegliche","womoeglichem","womoeglichen","womoeglicher","womoegliches","womöglich","womögliche","womöglichem","womöglichen","womöglicher","womögliches","woran","woraufhin","woraus","worden","worin","wuerde","wuerden","wuerdest","wuerdet","wurde","würde","wurden","würden","wurdest","würdest","wurdet","würdet","www","x","y","z","z.b","z.B.","zahlreich","zahlreichem","zahlreicher","zB","zb.","zehn","zehnte","zehnten","zehnter","zehntes","zeit","zeitweise","zeitweisem","zeitweisen","zeitweiser","ziehen","zieht","ziemlich","ziemliche","ziemlichem","ziemlichen","ziemlicher","ziemliches","zirka","zog","zogen","zu","zudem","zuerst","zufolge","zugleich","zuletzt","zum","zumal","zumeist","zumindest","zunächst","zunaechst","zur","zurück","zurueck","zusammen","zusehends","zuviel","zuviele","zuvieler","zuweilen","zwanzig","zwar","zwei","zweifelsfrei","zweifelsfreie","zweifelsfreiem","zweifelsfreien","zweifelsfreier","zweifelsfreies","zweite","zweiten","zweiter","zweites","zwischen","zwölf"]
            elif stopword_selection=="Use a built-in list of stop words in English":
                word_stopwords=['a','about','above','after','again','against','ain','all','am','an','and','any','are','aren',"aren't",'as','at','be','because','been','before','being','below','between','both','but','by','can','couldn',"couldn't",'d','did','didn',"didn't",'do','does','doesn',"doesn't",'doing','don',"don't",'down','during','each','few','for','from','further','had','hadn',"hadn't",'has','hasn',"hasn't",'have','haven',"haven't",'having','he','her','here','hers','herself','him','himself','his','how','i','if','in','into','is','isn',"isn't",'it',"it's",'its','itself','just','ll','m','ma','me','mightn',"mightn't",'more','most','mustn',"mustn't",'my','myself','needn',"needn't",'no','nor','not','now','o','of','off','on','once','only','or','other','our','ours','ourselves','out','over','own','re','s','same','shan',"shan't",'she',"she's",'should',"should've",'shouldn',"shouldn't",'so','some','such','t','than','that',"that'll",'the','their','theirs','them','themselves','then','there','these','they','this','those','through','to','too','under','until','up','ve','very','was','wasn',"wasn't",'we','were','weren',"weren't",'what','when','where','which','while','who','whom','why','will','with','won',"won't",'wouldn',"wouldn't",'y','you',"you'd","you'll","you're","you've",'your','yours','yourself','yourselves']
            elif stopword_selection=="Specify stop words":
                word_stopwords=[]
                user_stopwords=st.text_area('Please enter or copy stop words here', value='', height=200, key = st.session_state['key'] )
                if len(user_stopwords)>0:
                    stopwords_cv = CountVectorizer()
                    stopwords_cv_fit=stopwords_cv.fit_transform([user_stopwords])                    
                    word_stopwords=stopwords_cv.get_feature_names()
                                        
                st.write("")

            a4,a5=st.columns(2)
            with a4:
                # user specification of words to search
                word_list=pd.DataFrame(columns=word_sorted.index)                
                #words_cleaned=word_list.drop(word_stopwords,axis=1)
                words_cleaned=sorted(list(set(word_list)-set(word_stopwords)))       
                
                find_words=st.multiselect("Search sentences with following words", 
                    words_cleaned, key = st.session_state['key'])
            with a5:
                #user-specification of n-grams
                user_ngram=st.number_input("Specify the number of words to be extracted (n-grams)", min_value=1, value=2, key = st.session_state['key'])
            
            if st.checkbox('Show a word count', value = False, key = st.session_state['key']): 
                st.write(word_sorted)  
            
            st.write("")
            number_remove=st.checkbox("Remove numbers from text", value=True, key = st.session_state['key'])  
                       
            a4,a5=st.columns(2)
            with a4:
                #WordCloud color specification               
                st.write("")
                draw_WordCloud=st.checkbox("Create a Word Cloud", value=True, key = st.session_state['key'])  
            
            with a5:    
                if draw_WordCloud==True:              
                    #color options for the WordCloud (user selection)
                    color_options= pd.DataFrame(np.array([[21, 120, 12, 240, 30]]), 
                    columns=['orange', 'green', 'red','blue','brown'])
                    
                    user_color_name=st.selectbox('Select the main color of your WordCloud',color_options.columns, key = st.session_state['key'])
                    user_color=color_options[user_color_name]
            
            
            st.write("")
            st.write("")
            run_text = st.button("Press to start text processing...")
                

            if run_text:
                st.write("")
                st.write("")

                st.info("Text processing progress")
                text_bar = st.progress(0.0)
                progress = 0
                
                #---------------------------------------------------------------------------------
                # Basic NLP metrics and visualisations
                #---------------------------------------------------------------------------------
                wfreq_output = st.expander("Basic NLP metrics and visualisations ", expanded = False)
                with wfreq_output:
                    # Word frequency
                    st.subheader('Word count') 
                    
                    #calculate word frequency - stop words exluded:
                    word_sorted=fc.cv_text(user_text, word_stopwords, 1,user_precision,number_remove)
                                                         
                    st.write("")
                    st.write("Number of words: ", word_sorted["Word count"].sum())
                    st.write("Number of sentences", len(re.findall(r"([^.]*\.)" ,user_text)))
                    
                    if len(word_stopwords)>0:
                        st.warning("All analyses are based on text with stop words removed!") 
                    else:
                        st.warning("No stop words are removed from the text!") 

                    
                    st.write(word_sorted.style.format({"Rel. freq.": "{:.2f}"}))
                    
                    a4,a5=st.columns(2)
                    with a4:
                        # relative frequency for the top 10 words
                        txt_bar=word_sorted.head(min(len(word_sorted),10))
                        
                        fig = go.Figure()                
                        fig.add_trace(go.Bar(x=txt_bar["Rel. freq."], y=txt_bar.index, name='',marker_color = 'indianred',opacity=0.5,orientation='h'))
                        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',})  
                        fig.update_layout(xaxis=dict(title='relative fraction %', titlefont_size=14, tickfont_size=14,),)
                        fig.update_layout(hoverlabel=dict(bgcolor="white",align="left"))
                        fig.update_layout(height=400,width=400)
                        st.plotly_chart(fig, use_container_width=True) 
                        st.info("Top " + str(min(len(word_sorted),10)) + " words relative frequency")  

                    with a5:
                        fig = go.Figure(data=[go.Histogram(x=word_sorted["Word length"], histnorm='probability',marker_color ='steelblue',opacity=0.5)])
                        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',})  
                        fig.update_layout(xaxis=dict(title='word length', titlefont_size=14, tickfont_size=14,),)
                        fig.update_layout(hoverlabel=dict(bgcolor="white",align="left"))
                        fig.update_layout(height=400,width=400)
                        st.plotly_chart(fig, use_container_width=True)  
                        st.info("Word length distribution")  

                    
                    #word similarity vs. word length & word frequency
                    word_similarity=[]    
                    for word in txt_bar.index:  
                        d=0                         
                        for sword in txt_bar.index:
                            seq = SequenceMatcher(None,word,sword)
                            d = d+(seq.ratio()*100)                                                       
                        word_similarity.append([d])
                    
                    txt_bar["Similarity"]=(np.float_(word_similarity)/len(txt_bar.index)).round(user_precision)

                    a4,a5=st.columns(2)
                    with a4:
                        # bubble chart                    
                        fig = go.Figure(data=[go.Scatter(
                        y=txt_bar.index, x=txt_bar["Rel. freq."], mode='markers',text=txt_bar["Similarity"],
                            marker_size=txt_bar["Similarity"],marker_color='indianred') ])
                        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',})  
                        fig.update_layout(xaxis=dict(title='relative frequency', titlefont_size=14, tickfont_size=14,),)
                        fig.update_layout(hoverlabel=dict(bgcolor="white",align="left"))
                        fig.update_layout(height=400,width=400)
                        st.plotly_chart(fig, use_container_width=True) 
                        st.info("Bubble size eq. average word similarity across the top " + str(min(len(word_sorted),10)) +" words") 
                    with a5:
                        
                        df_to_plot=word_sorted
                        df_to_plot['word']=word_sorted.index
                        fig = px.scatter(data_frame=df_to_plot, x='Word length', y='Rel. freq.',hover_data=['word','Word length', 'Rel. freq.'], color_discrete_sequence=['steelblue'])
                        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
                        fig.update_layout(xaxis=dict(title="word length", titlefont_size=14, tickfont_size=14,),)
                        fig.update_layout(yaxis=dict(title="word frequency", titlefont_size=14, tickfont_size=14,),)
                        fig.update_layout(hoverlabel=dict(bgcolor="white", ))
                        fig.update_layout(height=400,width=400)
                        st.plotly_chart(fig) 
                        st.info("A comparision of frequencies of short and long words")

                    # bigram distribution
                    cv2_output=fc.cv_text(user_text, word_stopwords, 2,user_precision,number_remove)
                                       

                    # trigram distribution
                    cv3_output=fc.cv_text(user_text, word_stopwords, 3,user_precision,number_remove)
                                

                    a4,a5=st.columns(2)
                    with a4:                        
                        txt_bar=cv2_output.head(min(len(cv2_output),10))                        
                        fig = go.Figure()                
                        fig.add_trace(go.Bar(x=txt_bar["Rel. freq."], y=txt_bar.index, name='',marker_color = 'indianred',opacity=0.5,orientation='h'))
                        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',})  
                        fig.update_layout(xaxis=dict(title='relative fraction %', titlefont_size=14, tickfont_size=14,),)
                        fig.update_layout(hoverlabel=dict(bgcolor="white",align="left"))
                        fig.update_layout(height=400,width=400)
                        st.plotly_chart(fig, use_container_width=True) 
                        st.info("Top " + str(min(len(cv2_output),10)) + " bigrams relative frequency")  
                    
                    with a5:
                        txt_bar=cv3_output.head(10)                        
                        fig = go.Figure()                
                        fig.add_trace(go.Bar(x=txt_bar["Rel. freq."], y=txt_bar.index, name='',marker_color = 'indianred',opacity=0.5,orientation='h'))
                        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',})  
                        fig.update_layout(xaxis=dict(title='relative fraction %', titlefont_size=14, tickfont_size=14,),)
                        fig.update_layout(hoverlabel=dict(bgcolor="white",align="left"))
                        fig.update_layout(height=400,width=400)
                        st.plotly_chart(fig, use_container_width=True) 
                        st.info("Top " + str(min(len(cv2_output),10)) + " trigrams relative frequency")  


                    if draw_WordCloud==True:                    
                        #Draw WordCloud
                        wordcloud = WordCloud(background_color="white",
                            contour_color="white",max_words=100,stopwords=word_stopwords,
                            width=600,height=400,color_func=random_color_func).generate(user_text)  
                        fig_text, ax = plt.subplots()
                        ax=plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis("off")
                        
                        st.subheader('WordCloud')
                        st.pyplot(fig_text)   
                    
                    progress += 1
                    text_bar.progress(progress/3)

                                        
                     # Download link
                    st.write("")  
                    output = BytesIO()
                    excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                    word_sorted.to_excel(excel_file, sheet_name="words",index=False) 
                    if len(word_stopwords)>0:
                        pd.DataFrame(word_stopwords,columns=['stop words']).to_excel(excel_file, sheet_name="stop words",index=False)    
                    if len(cv2_output)>0:
                        cv2_output.to_excel(excel_file, sheet_name="bigrams",index=True) 
                    if len(cv3_output)>0:
                        cv3_output.to_excel(excel_file, sheet_name="trigrams",index=True) 
                    excel_file.save()
                    excel_file = output.getvalue()
                    b64 = base64.b64encode(excel_file)
                    dl_file_name = "BasicTextAnalysis.xlsx"
                    st.markdown(
                        f"""
                    <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download basic NLP metrics</a>
                    """,
                    unsafe_allow_html=True)
                    st.write("") 
                
                #---------------------------------------------------------------------------------
                # Sentences with specific words
                #---------------------------------------------------------------------------------
                if len(find_words)>0:                   
                    # extract all sentences with specific words:                 
                    sentences_list=[]                  
                    sentences = re.findall(r"([^.]*\.)" ,user_text) 
                    
                    for sentence in sentences:
                        if all(word in sentence for word in find_words):                                
                            if len(sentence)<1000: # threshold for to long sentences is 1000 characters
                                sentences_list.append(sentence)
                               
                    if len(sentences_list)>0: 
                        sentences_output = st.expander("Sentences with specific words", expanded = False)
                        with sentences_output:
                            for sentence in sentences_list:
                                st.write(sentence)
                                #st.table(pd.DataFrame({'Sentences':sentences_list}))
                        
                            # Download link
                            st.write("")  
                            output = BytesIO()
                            excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                            pd.DataFrame({'Sentences':sentences_list}).to_excel(excel_file, sheet_name="Sentences",index=False) 
                            excel_file.save()
                            excel_file = output.getvalue()
                            b64 = base64.b64encode(excel_file)
                            dl_file_name = "Sentences with specific words.xlsx"
                            st.markdown(
                                f"""
                            <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download sentences</a>
                            """,
                            unsafe_allow_html=True)
                            st.write("") 
                                
                progress += 1
                text_bar.progress(progress/3)
                
                
                #---------------------------------------------------------------------------------
                # User specific n-grams
                #---------------------------------------------------------------------------------
                #extract n-grams:
                ngram_list=[]              
                
                text_cv = fc.cv_text(user_text, word_stopwords,user_ngram,user_precision,number_remove)
                
                #CountVectorizer(analyzer='word', stop_words=set(word_stopwords), ngram_range=(user_ngram, user_ngram))
                #text_cv_fit=text_cv.fit_transform([user_text])
                #listToString='. '.join(text_cv.get_feature_names())
                listToString='. '.join(text_cv.index)
                sentences = re.findall(r"([^.]*\.)" ,listToString)  
                            
                for sentence in sentences:
                    if all(word in sentence for word in find_words):  
                        sentence=re.sub('[.]', '', sentence)                                             
                        ngram_list.append(sentence)
                
                if len(ngram_list)>0:
                    ngram_output = st.expander("n-grams", expanded = False)
                    with ngram_output: 
                        st.write("")
                        st.subheader("n-grams")
                        st.write("")
                        
                        for sentence in ngram_list:
                            st.write(sentence)

                        # Download link
                        st.write("")  
                        output = BytesIO()
                        excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                        pd.DataFrame({'n-gram':ngram_list}).to_excel(excel_file, sheet_name=str(user_ngram) +"-gram",index=False) 
                        excel_file.save()
                        excel_file = output.getvalue()
                        b64 = base64.b64encode(excel_file)
                        dl_file_name = str(user_ngram)+"gram.xlsx"
                        st.markdown(
                            f"""
                        <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download n-grams</a>
                        """,
                        unsafe_allow_html=True)                    
                
                        st.write("") 
                
                progress += 1
                text_bar.progress(progress/3)
                # Success message
                #st.success('Text processing completed')
 
 #----------------------------------------------------
 #  Stock data analysis - Yahoo Finance
 # ---------------------------------------------------               
    if tw_classifier=='Financial analysis': 
       
        st.write('Check stock prices and key performance indicators for companies included in [S&P 500] (https://en.wikipedia.org/wiki/List_of_S%26P_500_companies), [DAX] (https://en.wikipedia.org/wiki/DAX), [FTSE 100] (https://en.wikipedia.org/wiki/FTSE_100_Index), [CSI 300](https://en.wikipedia.org/wiki/CSI_300_Index), [Nikkei 225] (https://de.wikipedia.org/wiki/Nikkei_225), [CAC 40] (https://en.wikipedia.org/wiki/CAC_40), [BSE SENSEX] (https://en.wikipedia.org/wiki/BSE_SENSEX) and [KOSPI](https://en.wikipedia.org/wiki/KOSPI) indexes, or for any company available via [Yahoo Finance](https://finance.yahoo.com/). Note, the bulk download and data processing may take some time!')
        
        a1,a2,a3=st.columns([1,3,1])
        with a1:
            stock_search_option=st.radio('',['Indexes', 'Symbol'])
        with a2:
            st.markdown("")
            st.markdown("")
            st.write("Start your analysis by either selecting companies from specific key indexes, or by entering a ticker symbol")
       
                   
        # delete session state if input widget change
        def in_wid_change():
            st.session_state['load_data_button'] = None

        if stock_search_option !='Symbol':
            # select ticker for KPI-Dashboard
            co1 = st.container()
        
            st.write('Consider stocks from the following indices:')
            c1, c2, c3,c4 = st.columns(4)
            ticker_options = []
             
            SP500 = c1.checkbox('S&P 500', True, on_change=in_wid_change)
            CSI300 = c1.checkbox('CSI 300', False, on_change=in_wid_change)

            DAX = c2.checkbox('DAX', True, on_change=in_wid_change)
            NIKKEI225 = c2.checkbox('NIKKEI 225', False, on_change=in_wid_change)
            
            CAC40=c3.checkbox('CAC40', False, on_change=in_wid_change)
            BSE_SENSEX=c3.checkbox('BSE SENSEX', False, on_change=in_wid_change)

            FTSE = c4.checkbox('FTSE', False, on_change=in_wid_change)
            KOSPI = c4.checkbox('KOSPI', False, on_change=in_wid_change)

        else: #selected option is 'Symbol'
            selected_stock = st.text_input("Enter at least one stock ticker symbol! Please use space for ticker symbol separation!", "TSLA")
            selected_stock=(list(selected_stock.split(" ")))
            selected_stock = list(filter(None, selected_stock))
            list_companies=[]
            list_symbols = []           
            list_sectors = []
            list_stockindex = []
           

            for i in range(len(selected_stock)):
                sel_stock_string=selected_stock[i]
                
                try:
                    yf.Ticker(sel_stock_string).info['longName']
                    
                except:    
                    st.error('Cannot get any data on ' + sel_stock_string + ', so the ticker probably does not exist!  \n Please check spelling, or try another ticker symbol!')
                    return
                  
                list_companies.append(yf.Ticker(selected_stock[i]).info['longName'])                    
                list_sectors.append('XX')
                list_stockindex.append('XX')
            
            list_symbols = selected_stock            

            if len(selected_stock)==0:            
                selected_stock=[]            
                
        
        #--------------------------------
        # Get lists of company's/tickers 
        # FUNCTIONS
        
        # download list of companies
        @st.cache()
        def load_ticker():           
                                     
            #---------------------------------------------------------------------------------------------
            symbols_SP, symbols_DAX, symbols_FTSE, symbols_BSE, symbols_CAC, symbols_CSI, symbols_KO, symbols_NIK=[],[],[],[],[],[],[],[]
            company_SP , company_DAX , company_FTSE, company_BSE,company_CAC,company_CSI,company_KO,company_NIK=[],[],[],[],[],[],[],[]
            sector_SP , sector_DAX , sector_FTSE, sector_BSE,sector_CAC,sector_CSI,sector_KO,sector_NIK=[],[],[],[],[],[],[],[]
            index_SP , index_DAX , index_FTSE, index_BSE,index_CAC,index_CSI,index_KO,index_NIK=[],[],[],[],[],[],[],[]

            #S&P500:
            if SP500:
                (symbols_SP,company_SP,sector_SP,index_SP)= fc.get_stock_list('S&P500','https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',0, 0, 1, 3)
            #DAX
            if DAX:
                (symbols_DAX,company_DAX,sector_DAX,index_DAX)=fc.get_stock_list('DAX','https://en.wikipedia.org/wiki/DAX',3, 3, 1, 2)
            #FTSE
            if FTSE:
                (symbols_FTSE,company_FTSE,sector_FTSE,index_FTSE)=fc.get_stock_list('FTSE100','https://en.wikipedia.org/wiki/FTSE_100_Index',3, 1, 0, 2)    
            #CSI300
            if CSI300:
                (symbols_CSI,company_CSI,sector_CSI,index_CSI)=fc.get_stock_list('CSI300','https://en.wikipedia.org/wiki/CSI_300_Index',3, 0, 1, 4)    
            #NIKKEI225
            if NIKKEI225:
                (symbols_NIK,company_NIK,sector_NIK,index_NIK)=fc.get_stock_list('NIKKEI225','https://topforeignstocks.com/indices/the-components-of-the-nikkei-225-index/',0, 2, 1, 3)
            #CAC40 
            if CAC40:       
                (symbols_CAC,company_CAC,sector_CAC,index_CAC)=fc.get_stock_list('CAC40','https://en.wikipedia.org/wiki/CAC_40',3, 3, 0, 1)    
            #BSE_SENSEX
            if BSE_SENSEX:
                (symbols_BSE,company_BSE,sector_BSE,index_BSE)=fc.get_stock_list('BSE_SENSEX','https://en.wikipedia.org/wiki/BSE_SENSEX',1, 2, 3, 4)
            #S&P_TSX60 
            #(symbols_TS,company_TS,sector_TS,index_TS)=fc.get_stock_list('S&P_TSX60','https://topforeignstocks.com/indices/the-components-of-the-sptsx-composite-index/')        
            #KOSPI
            if KOSPI:
                (symbols_KO,company_KO,sector_KO,index_KO)=fc.get_stock_list('KOSPI','https://topforeignstocks.com/indices/the-components-of-the-korea-stock-exchange-kospi-index/',0, 2, 1, 3)        

            #---------------------------------------------------------------------------------------------
            # merge into one dataframe
            list_symbols = symbols_SP + symbols_DAX + symbols_FTSE+ symbols_BSE+symbols_CAC+symbols_CSI+symbols_KO+symbols_NIK
            list_companies = company_SP + company_DAX + company_FTSE+ company_BSE+company_CAC+company_CSI+company_KO+company_NIK
            list_sectors = sector_SP + sector_DAX + sector_FTSE+ sector_BSE+sector_CAC+sector_CSI+sector_KO+sector_NIK
            list_stockindex = index_SP + index_DAX + index_FTSE+ index_BSE+index_CAC+index_CSI+index_KO+index_NIK

            df_indicesdata = pd.DataFrame({'Ticker': list_symbols, 'Company': list_companies, 'Sector': list_sectors, 'Stock index': list_stockindex})
            
            return df_indicesdata, symbols_SP, symbols_DAX, symbols_FTSE, symbols_BSE, symbols_CAC, symbols_CSI, symbols_KO, symbols_NIK
            

        

        # define ticker object
        @st.cache(allow_output_mutation=True)
        def function_ticker():
            ticker = []
            for i in range(len(list_symbols)):
                ticker.append(Class_ticker())
            return ticker

               
        # function for multiselect ticker vs. company
        def ticker_dict_func(option):
            return ticker_dict[option]
        
        # function for multiselect index vs. company
        def index_dict_func(option):
            return ticker[option].company

        # create df_selected
        @st.cache(allow_output_mutation=True)
        def function_df_selected():
            df_selected = pd.DataFrame(columns=['Ticker', 'Company', 'Sector'])
            index_list_rename = []

            for i in range(len(selected_stock)):
                df_selected = df_selected.append(df_indicesdata[df_indicesdata['Ticker']==selected_stock[i]])
                index_list_rename += [i+1]

            index_list = df_selected.index.tolist()
            df_selected.index = index_list_rename
            return df_selected, index_list

        def data_available(label, source, selected_year):
            if label in source.index and source.at[label, selected_year] is not None and source.at[label, selected_year] != 0 :
                return True
            else:
                return False

        # calculate average
        def average_func(index_list, df_selected, df):
            if len(index_list) > 1:
                if len(df_selected['Stock index'])*2 == (len(df_selected[df_selected['Stock index']==df_selected.at[1, 'Stock index']])+len(df_selected[df_selected['Sector']==df_selected.at[1, 'Sector']])):
                    means = df.mean(axis=1)
                    df.insert(0, 'average' , means)
            return df

        # create df
        def fill_df_func(df, ticker, i, label_index, label_column):
            if df.empty:
                df = pd.DataFrame(ticker[i].kpis, index=label_index, columns=[label_column])
            else:
                df.insert(len(df.columns) , label_column, ticker[i].kpis)
            return df

        #----------------------------------------------------------------------------------------------
        # define class of tickers
        class Class_ticker:
            # master data
            def __init__(self):
                self.symbol = None
                self.company = None
                self.sector = None
                self.dict = None

            def master_data(self, i):
                self.symbol = list_symbols[i]
                self.company = list_companys[i]
                self.sector = list_sectors[i]
                self.stockindex = list_stockindex[i]
                
            # data loading
            def load_data(self):
                self.data = yf.Ticker(self.symbol)
            
            #----------------------------------------------------------------------------------------------
            # calculate kpis

            #profitability
            def kpi_profitability(self, selected_year):
                if data_available('Ebit', self.fi, selected_year) and data_available('Total Assets', self.bs, selected_year):
                    self.roi = self.fi.at['Ebit', selected_year] / self.bs.at['Total Assets', selected_year] * 100
                else:
                    self.roi = np.nan
                if data_available('Ebit', self.fi, selected_year) and data_available('Total Stockholder Equity', self.bs, selected_year):
                    self.roe = self.fi.at['Ebit', selected_year] / self.bs.at['Total Stockholder Equity', selected_year] * 100
                else:
                    self.roe = np.nan
                if data_available('Total Revenue', self.fi, selected_year):
                    self.revenues = self.fi.at['Total Revenue', selected_year] / 1000000000
                else:
                    self.revenues = np.nan
                if data_available('Ebit', self.fi, selected_year) and data_available('Depreciation', self.cf, selected_year) and data_available('Total Revenue', self.fi, selected_year):
                    self.ebitda_margin = (self.fi.at['Ebit', selected_year] + self.cf.at['Depreciation', selected_year]) / self.fi.at['Total Revenue', selected_year] *100
                else:
                    self.ebitda_margin = np.nan
                if data_available('Ebit', self.fi, selected_year) and data_available('Total Revenue', self.fi, selected_year):
                    self.ebit_margin = self.fi.at['Ebit', selected_year] / self.fi.at['Total Revenue', selected_year] * 100
                else:
                    self.ebit_margin = np.nan
                self.kpis = [self.roi, self.roe, self.revenues, self.ebitda_margin, self.ebit_margin]

            #debt capital
            def kpi_debt_capital(self, selected_year):
                if data_available('Total Liab', self.bs, selected_year) and data_available('Cash', self.bs, selected_year) and data_available('Ebit', self.fi, selected_year) and data_available('Depreciation', self.cf, selected_year):
                    self.netdebt_ebitda = (self.bs.at['Total Liab', selected_year] - self.bs.at['Cash', selected_year]) / (self.fi.at['Ebit', selected_year] + self.cf.at['Depreciation', selected_year])
                else:
                    self.netdebt_ebitda = np.nan
                if data_available('Ebit', self.fi, selected_year) and data_available('Depreciation', self.cf, selected_year) and data_available('Interest Expense', self.fi, selected_year):
                    self.ebita_interest = (self.fi.at['Ebit', selected_year] + self.cf.at['Depreciation', selected_year]) / self.fi.at['Interest Expense', selected_year] * -1 
                else:
                    self.ebita_interest = np.nan
                if data_available('Total Current Liabilities', self.bs, selected_year) and data_available('Total Current Assets', self.bs, selected_year):
                    self.current_ratio = self.bs.at['Total Current Liabilities', selected_year] / self.bs.at['Total Current Assets', selected_year]
                else:
                    self.current_ratio = np.nan
                if data_available('Accounts Payable', self.bs, selected_year) and data_available('Cost Of Revenue', self.fi, selected_year) :
                    self.dpo = self.bs.at['Accounts Payable', selected_year] * 365 / self.fi.at['Cost Of Revenue', selected_year]
                else:
                    self.dpo = np.nan
                self.kpis = [self.netdebt_ebitda, self.ebita_interest, self.current_ratio, self.dpo]

            #equity capital
            def kpi_equity_capital(self):
                if 'revenuePerShare' in self.info:
                    self.revenuepershare = self.info['revenuePerShare']
                else:
                    self.revenuepershare = np.nan    
                if 'forwardEps' in self.info:
                    self.eps = self.info['forwardEps']
                else:
                    self.eps = np.nan    
                if 'dividendRate' in self.info:
                    self.dividendrate = self.info['dividendRate']
                else:
                    self.dividendrate = np.nan    
                self.kpis = [self.revenuepershare, self.eps, self.dividendrate]

            #valuation
            def kpi_valuation(self):
                if 'forwardPE' in self.info:
                    self.forwardPE = self.info['forwardPE']
                else:
                    self.forwardPE = np.nan
                if 'pegRatio' in self.info:
                    self.pegratio = self.info['pegRatio']
                else:
                    self.pegratio = np.nan
                if 'priceToBook' in self.info:
                    self.pricetobook = self.info['priceToBook']
                else:
                    self.pricetobook = np.nan
                if 'enterpriseValue' in self.info:
                    self.ev = self.info['enterpriseValue'] / 1000000000
                else:
                    self.ev = np.nan
                if 'enterpriseToRevenue' in self.info:
                    self.evtorevenue = self.info['enterpriseToRevenue']
                else:
                    self.evtorevenue = np.nan
                if 'enterpriseToEbitda' in self.info:
                    self.evtoebitda = self.info['enterpriseToEbitda']
                else:
                    self.evtoebitda = np.nan
                self.kpis = [self.forwardPE, self.pegratio, self.pricetobook, self.ev, self.evtorevenue, self.evtoebitda]
                
            def stock_history(self):
                self.history = self.data.history(period=str(stock_period))

            #capital procurement
            def kpi_capital_procurement(self, selected_year):
                if data_available('Total Cash From Operating Activities', self.cf, selected_year) and data_available('Total Cashflows From Investing Activities', self.cf, selected_year):
                    self.selffinancingratio = self.cf.at['Total Cash From Operating Activities', selected_year]/ self.cf.at['Total Cashflows From Investing Activities', selected_year] * -1
                else:
                    self.selffinancingratio = np.nan
                if data_available('Total Stockholder Equity', self.bs, selected_year) and data_available('Total Assets', self.bs, selected_year):
                    self.equityratio = self.bs.at['Total Stockholder Equity', selected_year] / self.bs.at['Total Assets', selected_year] * 100
                else:
                    self.equityratio = np.nan            
                self.kpis = [self.selffinancingratio, self.equityratio]
            
            #capital allocation
            def kpi_capital_allocation(self, selected_year):
                if data_available('Capital Expenditures', self.cf, selected_year):
                    self.capexrevenueratio = self.cf.at['Capital Expenditures', selected_year] * -1 / self.fi.at['Total Revenue', selected_year] 
                else:
                    self.capexrevenueratio = np.nan
                if data_available('Research Development', self.fi, selected_year):
                    self.RDrevenueratio = self.fi.at['Research Development', selected_year] / self.fi.at['Total Revenue', selected_year]
                else:
                    self.RDrevenueratio = np.nan
                if data_available('Inventory', self.bs, selected_year) and data_available('Net Receivables', self.bs, selected_year):
                    self.ccc = (self.bs.at['Inventory', selected_year] * 365 / self.fi.at['Total Revenue', selected_year]) + (self.bs.at['Net Receivables', selected_year] * 365 / self.fi.at['Total Revenue', selected_year]) - self.dpo   
                else:
                    self.ccc = np.nan
                self.kpis = [self.capexrevenueratio, self.RDrevenueratio, self.ccc]

            #procurement market
            def kpi_procurement_market(self, selected_year):
                if selected_year == self.fi.columns[0] and 'fullTimeEmployees' in self.info and data_available('Total Revenue', self.fi, selected_year):
                    self.labour_productivity = self.fi.at['Total Revenue', self.fi.columns[0]] / 1000 / self.info['fullTimeEmployees'] 
                else:
                    self.labour_productivity = np.nan
                if data_available('Total Revenue', self.fi, selected_year) and data_available('Net Receivables', self.bs, selected_year):
                    self.asset_turnover = self.fi.at['Total Revenue', selected_year] / self.bs.at['Net Receivables', selected_year] * 100
                else:
                    self.asset_turnover = np.nan
                self.kpis = [self.labour_productivity, self.asset_turnover]
        
        #----------------------------------------------------------------------------------------------
        # load ticker
        if stock_search_option=='Symbol':        
            
            df_indicesdata = pd.DataFrame({'Ticker': list_symbols, 'Company': list_companies, 'Sector': list_sectors, 'Stock index': list_stockindex})
            symbols_SP, symbols_DAX, symbols_FTSE, symbols_BSE, symbols_CAC, symbols_CSI, symbols_KO, symbols_NIK=[],[],[],[],[],[] ,[] ,[]  
        else:
            df_indicesdata, symbols_SP, symbols_DAX, symbols_FTSE, symbols_BSE, symbols_CAC, symbols_CSI, symbols_KO, symbols_NIK = load_ticker()
        
        list_symbols = df_indicesdata['Ticker'].values.tolist()
        list_companys = df_indicesdata['Company'].values.tolist()
        list_sectors = df_indicesdata['Sector'].values.tolist()
        list_stockindex = df_indicesdata['Stock index'].values.tolist()

       # ini_msg=st.info("Initializing stock data loading...")
        #my_bar = st.progress(0.0)  
       # progress_sum=len(list_symbols)         
        #progress = 0

        # define ticker object and load data
        ticker = function_ticker()

        # create dictonary, master data, load data
        ticker_dict = {}
        
        for i in range(len(list_symbols)):
            ticker[i].master_data(i)
            ticker_dict[ticker[i].symbol] = ticker[i].company
            ticker[i].load_data()
            #progress += 1
            #my_bar.progress(progress/progress_sum)
        #suc_msg=st.success('Stock data loading... done!')    
       # time.sleep(1)
       # my_bar.empty()
       # suc_msg.empty()
        #ini_msg.empty()

        # check session state
        if 'selected_stock' not in st.session_state:
            st.session_state['selected_stock'] = None
        if 'load_data_button' not in st.session_state:
            st.session_state['load_data_button'] = None
        if 'list_years' not in st.session_state:
            st.session_state['list_years'] = None
        #----------------------------------------------------------------------------------------------
        if stock_search_option !='Symbol':
           
            co1 = st.container()  
           
            
            if SP500:
                ticker_options += symbols_SP
            if DAX:
                ticker_options +=symbols_DAX
            if FTSE:
                ticker_options += symbols_FTSE
            if CSI300:
                ticker_options += symbols_CSI
            if NIKKEI225:
                ticker_options += symbols_NIK
            if BSE_SENSEX:
                ticker_options += symbols_BSE
            if CAC40:
                ticker_options += symbols_CAC                
            if KOSPI:
                ticker_options += symbols_KO
        


            if SP500:
                default = 'MSFT'
            elif DAX:
                default = 'VOW3.DE'
            elif FTSE:
                default = 'VOD.L'
            elif CSI300:
                default = '601318.SS'
            elif NIKKEI225:
                default = '9983.T'
            elif BSE_SENSEX:
                default = 'HDFCBANK.BO'
            elif CAC40:
                default = 'BNP.PA'                
            elif KOSPI:
                default = '005930.KS'  
            else:
                st.error('Please select at least one stock index.')
                st.stop()
            
             
            st.markdown("")
            selected_stock = st.multiselect('Select at least one company', ticker_options, default , format_func=ticker_dict_func, on_change=in_wid_change)
        else:
            list_symbols = selected_stock            
            #list_companies = ticker[0].data.info['longName']
            #list_sectors = 'XX'
            #list_stockindex = 'XX'            

            df_indicesdata = pd.DataFrame({'Ticker': list_symbols, 'Company': list_companies, 'Sector': list_sectors, 'Stock index': list_stockindex})
            #st.write(df_indicesdata)   

        # output specification
        tb_options=['Basic info','Institutional Holders','Stock Price', 'Balance Sheet', 'Cashflow','Other Financials']
        tkpi_options= ['Profitability', 'Debt Capital', 'Equity Capital','Valuation', 'Capital Procurement','Capital Allocation','Procurement Market']
        tb_output=st.multiselect('Select ticker basic information',tb_options,['Stock Price', 'Balance Sheet','Cashflow','Other Financials'])
        tkpi_output=st.multiselect('Select performance indicators',tkpi_options,['Profitability', 'Debt Capital', 'Valuation'])
         
       
        if len(selected_stock) == 0:
            st.error('Please select at least one ticker.')
            st.session_state['selected_stock'] = None
            st.stop()

        # ticker from the same sector for 1 selection only
        if len(selected_stock) == 1:
            sector_list = []
            sector_ticker = []
            sector_list = df_indicesdata[df_indicesdata['Ticker']==selected_stock[0]].index.tolist()
            for i in range(len(ticker)):
                if ticker[i].sector == ticker[sector_list[0]].sector and ticker[i].stockindex == ticker[sector_list[0]].stockindex and ticker[i].company != ticker[sector_list[0]].company:
                    sector_list += [i]
                    sector_ticker += [ticker[i].symbol]

         #   c1 = st.container()
         #   c2 = st.container()
         #   select_all = c2.checkbox('Select all companies of '+ ticker[sector_list[0]].sector + ' in ' + ticker[sector_list[0]].stockindex + ' (can take some time)', on_change=in_wid_change)
         #   if select_all:
         #       sector_comparison = c1.multiselect('Select a Ticker from the same sector and stock index for comparison', sector_ticker, sector_ticker, format_func=ticker_dict_func, on_change=in_wid_change)
         #   else:
         #       sector_comparison = c1.multiselect('Select a Ticker from the same sector and stock index for comparison', sector_ticker, format_func=ticker_dict_func, on_change=in_wid_change)
         #   selected_stock += sector_comparison
        #else: select_all = None

        # create df_selected
        df_selected, index_list = function_df_selected()
        
        #st.write('**Your selection:**')
        #st.write(df_selected)

        # company information
       # if len(index_list) < 2:
       #     for i in index_list:
       #         if st.checkbox('Show company information for ' + ticker[i].company):
       #             st.write(ticker[i].data.info['longBusinessSummary'])
       # else: 
       #     company_info = st.multiselect('Show company information for ', index_list, format_func=index_dict_func)
       #     for i in company_info:
       #         st.markdown('Company information for ' + ticker[i].company)
       #         st.write(ticker[i].data.info['longBusinessSummary'])

        #----------------------------------------------------------------------------------------------
        ######## DATA LOADING #########
        st.markdown("")
        b1, b5 = st.columns([5,1])
        load_data_button = b1.button('Get Financial Figures')
        if b5.button('Clear Cache', on_click=in_wid_change):
            st.legacy_caching.clear_cache()
        if load_data_button:
            st.session_state['load_data_button'] = load_data_button

        if st.session_state['load_data_button']:          

            c3 = st.container()
            my_bar = st.progress(0.0)
            progress_sum = len(index_list) + len(tb_options)+len(tkpi_options)
           
            progress = 0

            if selected_stock != st.session_state['selected_stock'] or load_data_button:
                st.session_state['selected_stock'] = selected_stock
                list_years = []
                for i in index_list:
                    ticker[i].bs = ticker[i].data.balance_sheet
                    if ticker[i].bs.empty:
                        index_list.remove(i)
                        st.warning('No data found for ' + ticker[i].company)
                    else:
                        ticker[i].bs.columns = pd.DatetimeIndex(ticker[i].bs.columns).year
                        for element in ticker[i].bs.columns:
                            if element not in list_years:
                                list_years.append(element)
                        ticker[i].cf = ticker[i].data.cashflow
                        ticker[i].cf.columns = pd.DatetimeIndex(ticker[i].cf.columns).year
                        ticker[i].fi = ticker[i].data.financials
                        ticker[i].fi.columns = pd.DatetimeIndex(ticker[i].fi.columns).year
                        ticker[i].info = ticker[i].data.info
                        # check and adjust duplicate years
                        list_columns = ticker[i].bs.columns.values.tolist()
                        for n in range(len(ticker[i].bs.columns)):
                            if any(ticker[i].bs.columns.duplicated()):
                                list_duplicate_bool = ticker[i].bs.columns.duplicated()
                                index_duplicate = [i for i, x in enumerate(list_duplicate_bool) if x]
                                list_columns[index_duplicate[0]] = list_columns[index_duplicate[0]]-1
                                ticker[i].bs.columns = list_columns
                                ticker[i].cf.columns = list_columns
                                ticker[i].fi.columns = list_columns

                    progress += 1
                    my_bar.progress(progress/progress_sum)

                # sort list of years
                list_years.sort(reverse=True)
                st.session_state['list_years'] = list_years
            else: 
                list_years = st.session_state['list_years']
                progress += len(index_list)
                my_bar.progress(progress/progress_sum)

            #----------------------------------------------------------------------------------------------
                      
            st.subheader('Ticker basics')

            #----------------------------------------------------------------------------------------------
            # basic info
            if 'Basic info' in tb_output:
                with st.expander('Company Info'):                                  

                    if len(index_list) == 1:  
                        st.markdown('Company information for ' + ticker[0].company)
                        st.write(ticker[0].data.info['longBusinessSummary'])                 
                                              
                    else:                        
                        for i in index_list:
                            st.markdown("")
                            st.markdown('Company information for ' + ticker[i].company)
                            st.write(ticker[0].data.info['longBusinessSummary'])   
            #progress
            progress += 1
            my_bar.progress(progress/progress_sum)                
                    

            # basic info
            if 'Institutional Holders' in tb_output:
                with st.expander('Institutional Holders'):
                    if len(index_list) == 1:                       
                        st.markdown('Institutional holders for ' + ticker[0].company +":")
                        st.write(ticker[0].data.institutional_holders)                                                
                    else:                        
                        for i in index_list:
                            st.markdown("")                               
                            st.markdown('Institutional holders for ' + ticker[i].company +":")
                            st.write(ticker[i].data.institutional_holders)  
            #progress
            progress += 1
            my_bar.progress(progress/progress_sum)

            #Stock price
            if 'Stock Price' in tb_output:
                with st.expander('Stock Price'):
                    st.write('**Stock Price Development**')
                    # Selection of parameter
                    #if len(index_list) == 1:
                    #   stock_para = st.multiselect('Select parameter for stock price visualization', ['Open', 'High', 'Low', 'Close', 'Volume'], 'Open')
                    #elif len(index_list) > 1:
                    stock_para = st.selectbox('Select stock price info', ['Open', 'High', 'Low', 'Close', 'Volume'])
                    
                    stock_period = st.selectbox('Select time period', ['start/end day','max','10y','5y','2y','1y','ytd','6mo','3mo','1mo','5d', '1d'], 2)
                    if stock_period=='start/end day':
                        today = datetime.date.today()
                        last_year = today - datetime.timedelta(days=365)
                        a1,a2=st.columns(2)
                        with a1:
                            start_date = st.date_input('Select start date', last_year, key=2)
                        with a2:
                            end_date = st.date_input('Select end date', today,key=3)
                        if start_date > end_date:
                            st.error('ERROR: End date must fall after start date.')   

                    
                    if len(index_list) < 6:
                        df_history = pd.DataFrame()
                        selected_company = []
                        for i in index_list:
                            if stock_period=='start/end day':
                                ticker[i].history = ticker[i].data.history(period='1d', start=start_date, end=end_date)
                            else:
                                ticker[i].history = ticker[i].data.history(period=str(stock_period))
                            if df_history.empty:
                                df_history = pd.DataFrame(ticker[i].history[stock_para])
                                selected_company += [ticker[i].company]
                            else:
                                df_history.insert(len(df_history.columns) , ticker[i].company, ticker[i].history[stock_para])
                                selected_company += [ticker[i].company]
                        df_history.columns = selected_company
                    else:
                        df_history = pd.DataFrame()
                        selected_company = []

                        for i in st.multiselect('Select company for visualization', index_list, index_list[0], format_func=index_dict_func):
                            if stock_period=='start & end day':
                                ticker[i].history = ticker[i].data.history(period='1d',start=start_date,end=end_date)
                            else:
                                ticker[i].history = ticker[i].data.history(period=str(stock_period))
                            if df_history.empty:
                                df_history = pd.DataFrame(ticker[i].history[stock_para])
                                selected_company += [ticker[i].company]
                            else:
                                df_history.insert(len(df_history.columns) , ticker[i].company, ticker[i].history[stock_para])
                                selected_company += [ticker[i].company]
                        df_history.columns = selected_company

                    st.line_chart(df_history)
            #progress
            progress += 1
            my_bar.progress(progress/progress_sum)        
            
            # Balance Sheet
            if 'Balance Sheet' in tb_output:
                with st.expander('Balance Sheet'):

                    df_bs = pd.DataFrame()
                    no_rows=len(ticker[index_list[0]].data.balance_sheet.index)
                    label_index=ticker[index_list[0]].data.balance_sheet.index

                    if len(index_list) == 1:        
                        for y in ticker[index_list[0]].data.balance_sheet.columns:
                            df_bs=ticker[index_list[0]].data.balance_sheet
                            
                    if len(index_list) > 1:
                        selected_year =  st.selectbox('select year', list_years, key=4)
                        year_id=max(list_years)-selected_year 
                     
                        for i in index_list:
                            if selected_year in pd.DatetimeIndex(ticker[i].data.balance_sheet.columns).year:   
                                year_id=max(list_years)-selected_year                
                                df_col=ticker[i].data.balance_sheet.iloc[:,[year_id]]
                                df_col.columns=[ticker[i].company]
                            else:
                                df_col =np.empty(no_rows)
                                df_col[:] = np.NaN
                                df_col=pd.DataFrame(df_col,index=label_index,columns=[ticker[i].company])
                                                                               
                            if df_bs.empty:                                
                                df_bs = df_col 
                            else:
                                df_bs[ticker[i].company]=df_col
                    
                    st.dataframe(df_bs.style.format("{:.2f}"))
            #progress
            progress += 1
            my_bar.progress(progress/progress_sum)    

            # Cashflow
            if 'Cashflow' in tb_output:
                with st.expander('Cashflow'):                   
                    
                    df_cf = pd.DataFrame()
                    no_rows=len(ticker[index_list[0]].data.cashflow.index)
                    label_index=ticker[index_list[0]].data.cashflow.index

                    if len(index_list) == 1:        
                        for y in ticker[index_list[0]].cf.columns:
                            df_cf=ticker[index_list[0]].data.cashflow
                            
                    if len(index_list) > 1:
                        selected_year =  st.selectbox('select year', list_years, key=5)
                        year_id=max(list_years)-selected_year 

                        for i in index_list:
                            if selected_year in ticker[i].cf.columns:   
                                year_id=max(list_years)-selected_year                
                                df_col=ticker[i].data.cashflow.iloc[:,[year_id]]
                                df_col.columns=[ticker[i].company]
                            else:
                                df_col =np.empty(no_rows)
                                df_col[:] = np.NaN
                                df_col=pd.DataFrame(df_col,index=label_index,columns=[ticker[i].company])
                                                                               
                            if df_cf.empty:                                
                                df_cf = df_col 
                            else:
                                df_cf[ticker[i].company]=df_col
                    
                    st.dataframe(df_cf.style.format("{:.2f}"))
            #progress
            progress += 1
            my_bar.progress(progress/progress_sum)

            # Other financials
            if 'Other Financials' in tb_output:
                with st.expander('Other Financials'):
                    df_of = pd.DataFrame()
                    no_rows=len(ticker[index_list[0]].data.financials.index)
                    label_index=ticker[index_list[0]].data.financials.index
                                        
                    if len(index_list) == 1:        
                        for y in pd.DatetimeIndex(ticker[0].data.financials.columns).year:                            
                            df_of=ticker[index_list[0]].data.financials
                            
                    if len(index_list) > 1:
                        selected_year =  st.selectbox('select year', list_years, key=6)
                        year_id=max(list_years)-selected_year 

                        for i in index_list:
                            if selected_year in pd.DatetimeIndex(ticker[i].data.financials.columns).year:   
                                year_id=max(list_years)-selected_year                
                                df_col=ticker[i].data.financials.iloc[:,[year_id]]
                                df_col.columns=[ticker[i].company]
                            else:
                                df_col =np.empty(no_rows)
                                df_col[:] = np.NaN
                                df_col=pd.DataFrame(df_col,index=label_index,columns=[ticker[i].company])
                                                                               
                            if df_of.empty:                                
                                df_of = df_col 
                            else:
                                df_of[ticker[i].company]=df_col
                    
                    st.dataframe(df_of)
       
            #progress
            progress += 1
            my_bar.progress(progress/progress_sum)

            if len(tb_output)>0:
                #download excel file
                st.markdown("")
                output = BytesIO()
                excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                
                if 'Balance Sheet' in tb_output:
                    df_bs.to_excel(excel_file, sheet_name='Balance sheet')
                if 'Stock Price' in tb_output:
                    df_history.to_excel(excel_file, sheet_name='Stock Price')
                if 'Cashflow' in tb_output:
                    df_cf.to_excel(excel_file, sheet_name='Cashflow')               
                if 'Other Financials' in tb_output:
                    df_of.to_excel(excel_file, sheet_name='Other Financials')
                    

                excel_file.save()
                excel_file = output.getvalue()
                b64 = base64.b64encode(excel_file)
                dl_file_name = "Stock Basic Info.xlsx"
                st.markdown(
                    f"""
                <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download Stock Basic Info</a>
                """,
                unsafe_allow_html=True)    





            #------------------------------------------------------------------------
            st.subheader('KPI-Dashboard')          
            
            # profitability
            if 'Profitability' in tkpi_output:
                with st.expander('Profitability'):
                    
                    label_index = 'Return on Investment (ROI) [in %]', 'Return on Equity (ROE) [in %]', 'Total Revenue [in billion]', 'EBITDA-Margin [in %]', 'EBIT-Margin [in %]'
                    df_profitability = pd.DataFrame()

                    if len(index_list) == 1:        
                        for y in ticker[index_list[0]].bs.columns:
                            ticker[index_list[0]].kpi_profitability(y)
                            df_profitability = fill_df_func(df_profitability, ticker, index_list[0], label_index, y)
                    if len(index_list) > 1:
                        selected_year =  st.selectbox('select year', list_years, key=7)
                        for i in index_list:
                            if selected_year in ticker[i].bs.columns:
                                ticker[i].kpi_profitability(selected_year)
                            else:
                                ticker[i].kpis = np.empty(len(label_index))
                                ticker[i].kpis.fill(np.nan)

                            df_profitability = fill_df_func(df_profitability, ticker, i, label_index, ticker[i].company)
                        
                        df_profitability = average_func(index_list, df_selected, df_profitability)

                    st.dataframe(df_profitability.style.format("{:.2f}"))
            #progress
            progress += 1
            my_bar.progress(progress/progress_sum)
            
            #----------------------------------------------------------------------------------------------
            # debt capital
            if 'Debt Capital' in tkpi_output:
                with st.expander('Debt capital'):
                    
                    label_index = 'Net Debt/EBITDA', 'EBITDA/Interest', 'Current Ratio', 'Days Payable Outstanding [in days]'
                    df_debt_capital = pd.DataFrame()

                    if len(index_list) == 1:        
                        for y in ticker[index_list[0]].bs.columns:
                            ticker[index_list[0]].kpi_debt_capital(y)
                            df_debt_capital = fill_df_func(df_debt_capital, ticker, index_list[0], label_index, y)
                    if len(index_list) > 1:
                        selected_year =  st.selectbox('select year', list_years, key=8)
                        for i in index_list:
                            if selected_year in ticker[i].bs.columns:
                                ticker[i].kpi_debt_capital(selected_year)
                            else:
                                ticker[i].kpis = np.empty(len(label_index))
                                ticker[i].kpis.fill(np.nan)

                            df_debt_capital = fill_df_func(df_debt_capital, ticker, i, label_index, ticker[i].company)
                        
                        df_debt_capital = average_func(index_list, df_selected, df_debt_capital)

                    st.dataframe(df_debt_capital.style.format("{:.2f}"))
            #progress
            progress += 1
            my_bar.progress(progress/progress_sum)

            #----------------------------------------------------------------------------------------------
            # equity capital
            if 'Equity Capital' in tkpi_output:
                with st.expander('Equity capital'):

                    label_index = 'Revenues per Share', ' Forward EPS', 'Forward Annual Dividend Rate' 
                    df_equity_capital = pd.DataFrame()
                    
                    for i in index_list:
                        ticker[i].kpi_equity_capital()
                        df_equity_capital = fill_df_func(df_equity_capital, ticker, i, label_index, ticker[i].company)

                    df_equity_capital = average_func(index_list, df_selected, df_equity_capital)                

                    st.dataframe(df_equity_capital.style.format("{:.2f}"))
            #progress
            progress += 1
            my_bar.progress(progress/progress_sum)

            #----------------------------------------------------------------------------------------------
            # valuation
            if 'Valuation' in tkpi_output:
                with st.expander('Valuation'):

                    label_index = 'Forward P/E', 'PEG Ratio (5yr expected)', 'P/B Ratio', 'Enterprise Value (EV) [in billion]', 'EV/Revenue', 'EV/EBITDA' 
                    df_valuation = pd.DataFrame()

                    for i in index_list:
                        ticker[i].kpi_valuation()
                        df_valuation = fill_df_func(df_valuation, ticker, i, label_index, ticker[i].company)

                    df_valuation = average_func(index_list, df_selected, df_valuation)                
                    st.dataframe(df_valuation.style.format(formatter="{:.2f}"))
                    st.write('')


            #progress
            progress += 1
            my_bar.progress(progress/progress_sum)

            #----------------------------------------------------------------------------------------------
            # capital procurement
            if 'Capital Procurement' in tkpi_output:
                with st.expander('Capital Procurement'):
                    
                    label_index = 'Self-Financing Ratio', 'Equity Ratio [in %]'
                    df_capital_procurement = pd.DataFrame()

                    if len(index_list) == 1:        
                        for y in ticker[index_list[0]].bs.columns:
                            ticker[index_list[0]].kpi_capital_procurement(y)
                            df_capital_procurement = fill_df_func(df_capital_procurement, ticker, index_list[0], label_index, y)
                    if len(index_list) > 1:
                        selected_year =  st.selectbox('select year', list_years, key=9)
                        for i in index_list:
                            if selected_year in ticker[i].bs.columns:
                                ticker[i].kpi_capital_procurement(selected_year)
                            else:
                                ticker[i].kpis = np.empty(len(label_index))
                                ticker[i].kpis.fill(np.nan)

                            df_capital_procurement = fill_df_func(df_capital_procurement, ticker, i, label_index, ticker[i].company)
                        
                        df_capital_procurement = average_func(index_list, df_selected, df_capital_procurement)

                    st.dataframe(df_capital_procurement.style.format("{:.2f}"))
            #progress
            progress += 1
            my_bar.progress(progress/progress_sum)

            #----------------------------------------------------------------------------------------------
            # capital allocation
            if 'Capital Allocation' in tkpi_output:
                with st.expander('Capital Allocation'):
                    
                    label_index = 'CapEx/Revenue', 'Research & Development/Revenue', 'Cash Conversion Cycle [in days]'
                    df_capital_allocation = pd.DataFrame()

                    if len(index_list) == 1:        
                        for y in ticker[index_list[0]].bs.columns:
                            ticker[index_list[0]].kpi_capital_allocation(y)
                            df_capital_allocation = fill_df_func(df_capital_allocation, ticker, index_list[0], label_index, y)
                    if len(index_list) > 1:
                        selected_year =  st.selectbox('select year', list_years, key=10)
                        for i in index_list:
                            if selected_year in ticker[i].bs.columns:
                                ticker[i].kpi_capital_allocation(selected_year)
                            else:
                                ticker[i].kpis = np.empty(len(label_index))
                                ticker[i].kpis.fill(np.nan)

                            df_capital_allocation = fill_df_func(df_capital_allocation, ticker, i, label_index, ticker[i].company)
                        
                        df_capital_allocation = average_func(index_list, df_selected, df_capital_allocation)

                    st.dataframe(df_capital_allocation.style.format("{:.2f}"))
            #progress
            progress += 1
            my_bar.progress(progress/progress_sum)

            #----------------------------------------------------------------------------------------------
            # procurement market
            if 'Procurement Market' in tkpi_output:
                with st.expander('Procurement Market'):
                    
                    label_index = 'Labour Productivity [in T per employee]', 'Asset turnover [in %]'
                    df_procurement_market = pd.DataFrame()

                    if len(index_list) == 1:        
                        for y in ticker[index_list[0]].bs.columns:
                            ticker[index_list[0]].kpi_procurement_market(y)
                            df_procurement_market = fill_df_func(df_procurement_market, ticker, index_list[0], label_index, y)
                    if len(index_list) > 1:
                        selected_year =  st.selectbox('select year', list_years, key=11)
                        for i in index_list:
                            if selected_year in ticker[i].bs.columns:
                                ticker[i].kpi_procurement_market(selected_year)
                            else:
                                ticker[i].kpis = np.empty(len(label_index))
                                ticker[i].kpis.fill(np.nan)

                            df_procurement_market = fill_df_func(df_procurement_market, ticker, i, label_index, ticker[i].company)
                        
                        df_procurement_market = average_func(index_list, df_selected, df_procurement_market)

                    st.dataframe(df_procurement_market.style.format("{:.2f}"))
            #progress
            progress += 1
            my_bar.progress(progress/progress_sum)
            if progress == progress_sum:
                c3suc_msg=c3.success('Data loading is completed!')
                time.sleep(2)
                my_bar.empty()
                c3suc_msg.empty()

            if len(tkpi_output)>0:
                #download excel file
                st.markdown("")
                output = BytesIO()
                excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                
                if 'Profitability' in tkpi_output:
                    df_profitability.to_excel(excel_file, sheet_name='KPI-Dashboard', startrow=3)
                    excel_file.sheets['KPI-Dashboard'].write(1, 0, 'Profitability')
                if 'Debt Capital' in tkpi_output:
                    df_debt_capital.to_excel(excel_file, sheet_name='KPI-Dashboard', startrow=12)
                    excel_file.sheets['KPI-Dashboard'].write(10, 0, 'Dept Capital')
                if 'Equity Capital' in tkpi_output:
                    df_equity_capital.to_excel(excel_file, sheet_name='KPI-Dashboard', startrow=20)
                    excel_file.sheets['KPI-Dashboard'].write(18, 0, 'Equity Capital')
                if 'Valuation' in tkpi_output:
                    df_valuation.to_excel(excel_file, sheet_name='KPI-Dashboard', startrow=27)
                    excel_file.sheets['KPI-Dashboard'].write(25, 0, 'Valuation')
                if 'Capital Procurement' in tkpi_output:
                    df_capital_procurement.to_excel(excel_file, sheet_name='KPI-Dashboard', startrow=37)
                    excel_file.sheets['KPI-Dashboard'].write(35, 0, 'Capital Procurement')
                if 'Capital Allocation' in tkpi_output:
                    df_capital_allocation.to_excel(excel_file, sheet_name='KPI-Dashboard', startrow=43)
                    excel_file.sheets['KPI-Dashboard'].write(41, 0, 'Capital Allocation')
                if 'Procurement Market' in tkpi_output:
                    df_procurement_market.to_excel(excel_file, sheet_name='KPI-Dashboard', startrow=50)
                    excel_file.sheets['KPI-Dashboard'].write(48, 0, 'Procurement Market')

                excel_file.save()
                excel_file = output.getvalue()
                b64 = base64.b64encode(excel_file)
                dl_file_name = "Financial Analysis.xlsx"
                st.markdown(
                    f"""
                <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download KPI-Dashboard</a>
                """,
                unsafe_allow_html=True)                            
                
        
