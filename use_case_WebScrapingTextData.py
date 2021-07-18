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
import re
import base64
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
    caching.clear_cache()

    # Hide traceback in error messages (comment out for de-bugging)
    sys.tracebacklimit = 0

   
    #------------------------------------------------------------------------------------------
    # SETTINGS

    settings_expander=st.sidebar.beta_expander('Settings')
    with settings_expander:
        st.caption("**Precision**")
        user_precision=st.number_input('Number of digits after the decimal point',min_value=0,max_value=10,step=1,value=4)
        #st.caption("**Help**")
        #sett_hints = st.checkbox('Show learning hints', value=False)
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
    fc.theme_func_dl_button()

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
            user_text=st.text_area('Please enter or copy your text here', value='STATY  \n\n STATY is growing out of the effort to bring more data insights to university education across all disciplines of the natural and social sciences. It is motivated by the belief that fostering data literacy, creativity and critical thinking are more effective towards innovation, than bringing endless units of introduction to programming to students who find learning programming an overwhelming task. By providing easy access to the methods of classical statistics and machine learning, STATY’s approach is to inspire students to explore issues they are studying in the curriculum directly on real data, practice interpreting the results and check the source code to see how it is done or to improve the code. STATY can be used in the process of teaching and learning data science, demonstrations of theoretical concepts across various disciplines, active learning, promotion of teamwork, research and beyond.', height=600, key = session_state.id )
            
            st.write("")
            if len(user_text)>0:  
                run_text_OK = True             
        
        elif word_sl=='web page':
            user_path_wp = st.text_input("What web page should I analyse?","https://en.wikipedia.org/wiki/Data_mining", key = session_state.id)
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
            stopword_selection=st.selectbox("Select stop word option",["No stop words (use all words)","Manually select stop words", "Use a built-in list of stop words in German", "Use a built-in list of stop words in English", "Specify stop words"], index=3, key=session_state.id)
            if stopword_selection=="No stop words (use all words)":
                word_stopwords=[] 
            elif stopword_selection=="Manually select stop words":
                word_stopwords=st.multiselect("Select stop words (words to be removed from the text)", word_sorted.index.tolist(),word_sorted.index[1:min(10,len(word_sorted.index))].tolist(), key = session_state.id)
            elif stopword_selection=="Use a built-in list of stop words in German":
                word_stopwords=["a","ab","aber","abermaliges","abermals","abgerufen","abgerufene","abgerufener","abgerufenes","abgesehen","ach","acht","achte","achten","achter","achtes","aehnlich","aehnliche","aehnlichem","aehnlichen","aehnlicher","aehnliches","aehnlichste","aehnlichstem","aehnlichsten","aehnlichster","aehnlichstes","aeusserst","aeusserste","aeusserstem","aeussersten","aeusserster","aeusserstes","ag","ähnlich","ähnliche","ähnlichem","ähnlichen","ähnlicher","ähnliches","ähnlichst","ähnlichste","ähnlichstem","ähnlichsten","ähnlichster","ähnlichstes","alle","allein","alleine","allem","allemal","allen","allenfalls","allenthalben","aller","allerdings","allerlei","alles","allesamt","allg","allg.","allgemein","allgemeine","allgemeinem","allgemeinen","allgemeiner","allgemeines","allgemeinste","allgemeinstem","allgemeinsten","allgemeinster","allgemeinstes","allmählich","allzeit","allzu","als","alsbald","also","am","an","and","andauernd","andauernde","andauerndem","andauernden","andauernder","andauerndes","ander","andere","anderem","anderen","anderenfalls","anderer","andererseits","anderes","anderm","andern","andernfalls","anderr","anders","anderst","anderweitig","anderweitige","anderweitigem","anderweitigen","anderweitiger","anderweitiges","anerkannt","anerkannte","anerkannter","anerkanntes","anfangen","anfing","angefangen","angesetze","angesetzt","angesetzten","angesetzter","ans","anscheinend","ansetzen","ansonst","ansonsten","anstatt","anstelle","arbeiten","au","auch","auf","aufgehört","aufgrund","aufhören","aufhörte","aufzusuchen","augenscheinlich","augenscheinliche","augenscheinlichem","augenscheinlichen","augenscheinlicher","augenscheinliches","augenscheinlichst","augenscheinlichste","augenscheinlichstem","augenscheinlichsten","augenscheinlichster","augenscheinlichstes","aus","ausdrücken","ausdrücklich","ausdrückliche","ausdrücklichem","ausdrücklichen","ausdrücklicher","ausdrückliches","ausdrückt","ausdrückte","ausgenommen","ausgenommene","ausgenommenem","ausgenommenen","ausgenommener","ausgenommenes","ausgerechnet","ausgerechnete","ausgerechnetem","ausgerechneten","ausgerechneter","ausgerechnetes","ausnahmslos","ausnahmslose","ausnahmslosem","ausnahmslosen","ausnahmsloser","ausnahmsloses","außen","außer","ausser","außerdem","ausserdem","außerhalb","äusserst","äusserste","äusserstem","äussersten","äusserster","äusserstes","author","autor","b","baelde","bald","bälde","bearbeite","bearbeiten","bearbeitete","bearbeiteten","bedarf","bedürfen","bedurfte","been","befahl","befiehlt","befiehlte","befohlene","befohlens","befragen","befragte","befragten","befragter","begann","beginnen","begonnen","behalten","behielt","bei","beide","beidem","beiden","beider","beiderlei","beides","beim","beinahe","beisammen","beispiel","beispielsweise","beitragen","beitrugen","bekannt","bekannte","bekannter","bekanntlich","bekanntliche","bekanntlichem","bekanntlichen","bekanntlicher","bekanntliches","bekennen","benutzt","bereits","berichten","berichtet","berichtete","berichteten","besonders","besser","bessere","besserem","besseren","besserer","besseres","bestehen","besteht","besten","bestenfalls","bestimmt","bestimmte","bestimmtem","bestimmten","bestimmter","bestimmtes","beträchtlich","beträchtliche","beträchtlichem","beträchtlichen","beträchtlicher","beträchtliches","betraechtlich","betraechtliche","betraechtlichem","betraechtlichen","betraechtlicher","betraechtliches","betreffend","betreffende","betreffendem","betreffenden","betreffender","betreffendes","bevor","bez","bez.","bezgl","bezgl.","bezueglich","bezüglich","bietet","bin","bis","bisher","bisherige","bisherigem","bisherigen","bisheriger","bisheriges","bislang","bisschen","bist","bitte","bleiben","bleibt","blieb","bloss","böden","boeden","brachte","brachten","brauchen","braucht","bräuchte","bringen","bsp","bsp.","bspw","bspw.","bzw","bzw.","c","ca","ca.","circa","d","d.h","da","dabei","dadurch","dafuer","dafür","dagegen","daher","dahin","dahingehend","dahingehende","dahingehendem","dahingehenden","dahingehender","dahingehendes","dahinter","damalige","damaligem","damaligen","damaliger","damaliges","damals","damit","danach","daneben","dank","danke","danken","dann","dannen","daran","darauf","daraus","darf","darfst","darin","darüber","darüberhinaus","darueber","darueberhinaus","darum","darunter","das","dasein","daselbst","daß","dass","dasselbe","Dat","davon","davor","dazu","dazwischen","dein","deine","deinem","deinen","deiner","deines","dem","dementsprechend","demgegenüber","demgegenueber","demgemaess","demgemäß","demgemäss","demnach","demselben","demzufolge","den","denen","denkbar","denkbare","denkbarem","denkbaren","denkbarer","denkbares","denn","dennoch","denselben","der","derart","derartig","derartige","derartigem","derartigen","derartiger","derem","deren","derer","derjenige","derjenigen","dermaßen","dermassen","derselbe","derselben","derzeit","derzeitig","derzeitige","derzeitigem","derzeitigen","derzeitiges","des","deshalb","desselben","dessen","dessenungeachtet","desto","desungeachtet","deswegen","dich","die","diejenige","diejenigen","dies","diese","dieselbe","dieselben","diesem","diesen","dieser","dieses","diesseitig","diesseitige","diesseitigem","diesseitigen","diesseitiger","diesseitiges","diesseits","dinge","dir","direkt","direkte","direkten","direkter","doch","doppelt","dort","dorther","dorthin","dran","drauf","drei","dreißig","drin","dritte","dritten","dritter","drittes","drüber","drueber","drum","drunter","du","duerfte","duerften","duerftest","duerftet","dunklen","durch","durchaus","durchweg","durchwegs","dürfen","dürft","durfte","dürfte","durften","dürften","durftest","dürftest","durftet","dürftet","e","eben","ebenfalls","ebenso","ect","ect.","ehe","eher","eheste","ehestem","ehesten","ehester","ehestes","ehrlich","ei","ei,","eigen","eigene","eigenem","eigenen","eigener","eigenes","eigenst","eigentlich","eigentliche","eigentlichem","eigentlichen","eigentlicher","eigentliches","ein","einander","einbaün","eine","einem","einen","einer","einerlei","einerseits","eines","einfach","einführen","einführte","einführten","eingesetzt","einig","einige","einigem","einigen","einiger","einigermaßen","einiges","einmal","einmalig","einmalige","einmaligem","einmaligen","einmaliger","einmaliges","eins","einseitig","einseitige","einseitigen","einseitiger","einst","einstmals","einzig","elf","empfunden","en","ende","endlich","entgegen","entlang","entsprechend","entsprechende","entsprechendem","entsprechenden","entsprechender","entsprechendes","entweder","er","ergänze","ergänzen","ergänzte","ergänzten","ergo","erhält","erhalten","erhielt","erhielten","erneut","ernst","eröffne","eröffnen","eröffnet","eröffnete","eröffnetes","erscheinen","erst","erste","erstem","ersten","erster","erstere","ersterem","ersteren","ersterer","ersteres","erstes","es","etc","etc.","etliche","etlichem","etlichen","etlicher","etliches","etwa","etwaige","etwas","euch","euer","eure","eurem","euren","eurer","eures","euretwegen","f","fall","falls","fand","fast","ferner","fertig","finde","finden","findest","findet","folgend","folgende","folgendem","folgenden","folgender","folgendermassen","folgendes","folglich","for","fordern","fordert","forderte","forderten","fort","fortsetzen","fortsetzt","fortsetzte","fortsetzten","fragte","frau","frei","freie","freier","freies","früher","fuer","fuers","fünf","fünfte","fünften","fünfter","fünftes","für","fürs","g","gab","gaenzlich","gaenzliche","gaenzlichem","gaenzlichen","gaenzlicher","gaenzliches","gängig","gängige","gängigen","gängiger","gängiges","ganz","ganze","ganzem","ganzen","ganzer","ganzes","gänzlich","gänzliche","gänzlichem","gänzlichen","gänzlicher","gänzliches","gar","gbr","geb","geben","geblieben","gebracht","gedurft","geehrt","geehrte","geehrten","geehrter","gefallen","gefälligst","gefällt","gefiel","gegeben","gegen","gegenüber","gegenueber","gehabt","gehalten","gehen","geht","gekannt","gekommen","gekonnt","gemacht","gemaess","gemäss","gemeinhin","gemocht","gemusst","genau","genommen","genug","gepriesener","gepriesenes","gerade","gern","gesagt","geschweige","gesehen","gestern","gestrige","getan","geteilt","geteilte","getragen","getrennt","gewesen","gewiss","gewisse","gewissem","gewissen","gewisser","gewissermaßen","gewisses","gewollt","geworden","ggf","ggf.","gib","gibt","gilt","ging","gleich","gleiche","gleichem","gleichen","gleicher","gleiches","gleichsam","gleichste","gleichstem","gleichsten","gleichster","gleichstes","gleichwohl","gleichzeitig","gleichzeitige","gleichzeitigem","gleichzeitigen","gleichzeitiger","gleichzeitiges","glücklicherweise","gluecklicherweise","gmbh","gott","gottseidank","gratulieren","gratuliert","gratulierte","groesstenteils","groß","gross","große","grosse","großen","grossen","großer","grosser","großes","grosses","grösstenteils","gruendlich","gründlich","gut","gute","guten","guter","gutes","h","hab","habe","haben","habt","haette","haeufig","haeufige","haeufigem","haeufigen","haeufiger","haeufigere","haeufigeren","haeufigerer","haeufigeres","halb","hallo","halten","hast","hat","hätt","hatte","hätte","hatten","hätten","hattest","hattet","häufig","häufige","häufigem","häufigen","häufiger","häufigere","häufigeren","häufigerer","häufigeres","heisst","hen","her","heraus","herein","herum","heute","heutige","heutigem","heutigen","heutiger","heutiges","hier","hierbei","hiermit","hiesige","hiesigem","hiesigen","hiesiger","hiesiges","hin","hindurch","hinein","hingegen","hinlanglich","hinlänglich","hinten","hintendran","hinter","hinterher","hinterm","hintern","hinunter","hoch","höchst","höchstens","http","hundert","i","ich","igitt","ihm","ihn","ihnen","ihr","ihre","ihrem","ihren","ihrer","ihres","ihretwegen","ihrige","ihrigen",
                "ihriges","im","immer","immerhin","immerwaehrend","immerwaehrende","immerwaehrendem","immerwaehrenden","immerwaehrender","immerwaehrendes","immerwährend","immerwährende","immerwährendem","immerwährenden","immerwährender","immerwährendes","immerzu","important","in","indem","indessen","Inf.","info","infolge","infolgedessen","information","innen","innerhalb","innerlich","ins","insbesondere","insgeheim","insgeheime","insgeheimer","insgesamt","insgesamte","insgesamter","insofern","inzwischen","irgend","irgendein","irgendeine","irgendeinem","irgendeiner","irgendeines","irgendetwas","irgendjemand","irgendjemandem","irgendwann","irgendwas","irgendwelche","irgendwen","irgendwenn","irgendwer","irgendwie","irgendwo","irgendwohin","ist","j","ja","jaehrig","jaehrige","jaehrigem","jaehrigen","jaehriger","jaehriges","jahr","jahre","jahren","jährig","jährige","jährigem","jährigen","jähriges","je","jede","jedem","jeden","jedenfalls","jeder","jederlei","jedermann","jedermanns","jedes","jedesmal","jedoch","jeglichem","jeglichen","jeglicher","jegliches","jemals","jemand","jemandem","jemanden","jemandes","jene","jenem","jenen","jener","jenes","jenseitig","jenseitigem","jenseitiger","jenseits","jetzt","jung","junge","jungem","jungen","junger","junges","k","kaeumlich","kam","kann","kannst","kaum","käumlich","kein","keine","keinem","keinen","keiner","keinerlei","keines","keineswegs","klar","klare","klaren","klares","klein","kleine","kleinen","kleiner","kleines","koennen","koennt","koennte","koennten","koenntest","koenntet","komme","kommen","kommt","konkret","konkrete","konkreten","konkreter","konkretes","könn","können","könnt","konnte","könnte","konnten","könnten","konntest","könntest","konntet","könntet","kuenftig","kuerzlich","kuerzlichst","künftig","kurz","kürzlich","kürzlichst","l","laengst","lag","lagen","lang","lange","langsam","längst","längstens","lassen","laut","lediglich","leer","legen","legte","legten","leicht","leide","leider","lesen","letze","letzte","letzten","letztendlich","letztens","letztere","letzterem","letzterer","letzteres","letztes","letztlich","lichten","lieber","liegt","liest","links","los","m","mache","machen","machst","macht","machte","machten","mag","magst","mahn","mal","man","manch","manche","manchem","manchen","mancher","mancherlei","mancherorts","manches","manchmal","mann","margin","massgebend","massgebende","massgebendem","massgebenden","massgebender","massgebendes","massgeblich","massgebliche","massgeblichem","massgeblichen","massgeblicher","mehr","mehrere","mehrerer","mehrfach","mehrmalig","mehrmaligem","mehrmaliger","mehrmaliges","mein","meine","meinem","meinen","meiner","meines","meinetwegen","meins","meist","meiste","meisten","meistens","meistenteils","mensch","menschen","meta","mich","mindestens","mir","mit","miteinander","mitgleich","mithin","mitnichten","mittel","mittels","mittelst","mitten","mittig","mitunter","mitwohl","mochte","möchte","mochten","möchten","möchtest","moechte","moeglich","moeglichst","moeglichste","moeglichstem","moeglichsten","moeglichster","mögen","möglich","mögliche","möglichen","möglicher","möglicherweise","möglichst","möglichste","möglichstem","möglichsten","möglichster","mögt","morgen","morgige","muessen","muesst","muesste","muß","muss","müssen","mußt","musst","müßt","müsst","musste","müsste","mussten","müssten","n","na","nach","nachdem","nacher","nachher","nachhinein","nächste","nacht","naechste","naemlich","nahm","nämlich","naturgemaess","naturgemäss","natürlich","ncht","neben","nebenan","nehmen","nein","neu","neue","neuem","neuen","neuer","neuerdings","neuerlich","neuerliche","neuerlichem","neuerlicher","neuerliches","neues","neulich","neun","neunte","neunten","neunter","neuntes","nicht","nichts","nichtsdestotrotz","nichtsdestoweniger","nie","niemals","niemand","niemandem","niemanden","niemandes","nimm","nimmer","nimmt","nirgends","nirgendwo","noch","noetigenfalls","nötigenfalls","nun","nur","nutzen","nutzt","nützt","nutzung","o","ob","oben","ober","oberen","oberer","oberhalb","oberste","obersten","oberster","obgleich","obs","obschon","obwohl","oder","oefter","oefters","off","offen","offenkundig","offenkundige","offenkundigem","offenkundigen","offenkundiger","offenkundiges","offensichtlich","offensichtliche","offensichtlichem","offensichtlichen","offensichtlicher","offensichtliches","oft","öfter","öfters","oftmals","ohne","ohnedies","online","ordnung","p","paar","partout","per","persoenlich","persoenliche","persoenlichem","persoenlicher","persoenliches","persönlich","persönliche","persönlicher","persönliches","pfui","ploetzlich","ploetzliche","ploetzlichem","ploetzlicher","ploetzliches","plötzlich","plötzliche","plötzlichem","plötzlicher","plötzliches","pro","q","quasi","r","reagiere","reagieren","reagiert","reagierte","recht","rechte","rechten","rechter","rechtes","rechts","regelmäßig","reichlich","reichliche","reichlichem","reichlichen","reichlicher","restlos","restlose","restlosem","restlosen","restloser","restloses","richtig","richtiggehend","richtiggehende","richtiggehendem","richtiggehenden","richtiggehender","richtiggehendes","rief","rund","rundheraus","rundum","runter","s","sa","sache","sage","sagen","sagt","sagte","sagten","sagtest","sagtet","sah","samt","sämtliche","sang","sangen","satt","sattsam","schätzen","schätzt","schätzte","schätzten","scheinbar","scheinen","schlecht","schlechter","schlicht","schlichtweg","schließlich","schluss","schlussendlich","schnell","schon","schreibe","schreiben","schreibens","schreiber","schwerlich","schwerliche","schwerlichem","schwerlichen","schwerlicher","schwerliches","schwierig","sechs","sechste","sechsten","sechster","sechstes","sect","sehe","sehen","sehr","sehrwohl","seht","sei","seid","seien","seiest","seiet","sein","seine","seinem","seinen","seiner","seines","seit","seitdem","seite","seiten","seither","selbe","selben","selber","selbst","selbstredend","selbstredende","selbstredendem","selbstredenden","selbstredender","selbstredendes","seltsamerweise","senke","senken","senkt","senkte","senkten","setzen","setzt","setzte","setzten","sich","sicher","sicherlich","sie","sieben","siebente","siebenten","siebenter","siebentes","siebte","siehe","sieht","sind","singen","singt","so","sobald","sodaß","soeben","sofern","sofort","sog","sogar","sogleich","solang","solange","solc","solchen","solch","solche","solchem","solchen","solcher","solches","soll","sollen","sollst","sollt","sollte","sollten","solltest","solltet","somit","sondern","sonst","sonstig","sonstige","sonstigem","sonstiger","sonstwo","sooft","soviel","soweit","sowie","sowieso","sowohl","später","spielen","startet","startete","starteten","startseite","statt","stattdessen","steht","steige","steigen","steigt","stellenweise","stellenweisem","stellenweisen","stets","stieg","stiegen","such","suche","suchen","t","tag","tage","tagen","tages","tat","tät","tatsächlich","tatsächlichen","tatsächlicher","tatsächliches","tatsaechlich","tatsaechlichen","tatsaechlicher","tatsaechliches","tausend","teil","teile","teilen","teilte","teilten","tel","tief","titel","toll","total","trage","tragen","trägt","tritt","trotzdem","trug","tun","tust","tut","txt","u","übel","über","überall","überallhin","überaus","überdies","überhaupt","überll","übermorgen","üblicherweise","übrig","übrigens","ueber","ueberall","ueberallhin","ueberaus","ueberdies","ueberhaupt","uebermorgen","ueblicherweise","uebrig","uebrigens","uhr","um","ums","umso","umstaendehalber","umständehalber","unbedingt","unbedingte","unbedingter","unbedingtes","und","unerhoert","unerhoerte","unerhoertem","unerhoerten","unerhoerter","unerhoertes","unerhört","unerhörte","unerhörtem","unerhörten","unerhörter","unerhörtes","ungefähr","ungemein","ungewoehnlich","ungewoehnliche","ungewoehnlichem","ungewoehnlichen","ungewoehnlicher","ungewoehnliches","ungewöhnlich","ungewöhnliche","ungewöhnlichem","ungewöhnlichen","ungewöhnlicher","ungewöhnliches","ungleich","ungleiche","ungleichem","ungleichen","ungleicher","ungleiches","unmassgeblich","unmassgebliche","unmassgeblichem","unmassgeblichen","unmassgeblicher","unmassgebliches","unmoeglich","unmoegliche","unmoeglichem","unmoeglichen","unmoeglicher","unmoegliches","unmöglich","unmögliche","unmöglichen","unmöglicher","unnötig","uns","unsaeglich","unsaegliche","unsaeglichem","unsaeglichen","unsaeglicher","unsaegliches","unsagbar","unsagbare","unsagbarem","unsagbaren","unsagbarer","unsagbares","unsäglich","unsägliche","unsäglichem","unsäglichen","unsäglicher","unsägliches","unse","unsem","unsen","unser","unsere","unserem","unseren","unserer","unseres","unserm","unses","unsre","unsrem","unsren","unsrer","unsres","unstreitig","unstreitige","unstreitigem","unstreitigen","unstreitiger","unstreitiges","unten","unter","unterbrach","unterbrechen","untere","unterem","unteres","unterhalb","unterste","unterster","unterstes","unwichtig","unzweifelhaft","unzweifelhafte","unzweifelhaftem","unzweifelhaften","unzweifelhafter","unzweifelhaftes","usw","usw.","v","vergangen","vergangene","vergangenen","vergangener","vergangenes","vermag","vermögen","vermutlich","vermutliche","vermutlichem","vermutlichen","vermutlicher","vermutliches","veröffentlichen","veröffentlicher","veröffentlicht","veröffentlichte","veröffentlichten","veröffentlichtes","verrate","verraten","verriet","verrieten","version","versorge","versorgen","versorgt","versorgte","versorgten","versorgtes","viel","viele","vielem","vielen","vieler","vielerlei","vieles","vielleicht","vielmalig","vielmals","vier","vierte","vierten","vierter","viertes","voellig","voellige","voelligem","voelligen","voelliger","voelliges","voelligst","vollends","völlig","völlige","völligem","völligen","völliger","völliges","völligst","vollstaendig","vollstaendige","vollstaendigem","vollstaendigen","vollstaendiger","vollstaendiges","vollständig","vollständige","vollständigem","vollständigen","vollständiger","vollständiges","vom","von","vor","voran","vorbei","vorgestern","vorher","vorherig","vorherige","vorherigem","vorheriger",
                "vorne","vorüber","vorueber","w","wachen","waehrend","waehrenddessen","waere","während","währenddem","währenddessen","wann","war","wär","wäre","waren","wären","warst","wart","warum","was","weder","weg","wegen","weil","weiß","weit","weiter","weitere","weiterem","weiteren","weiterer","weiteres","weiterhin","weitestgehend","weitestgehende","weitestgehendem","weitestgehenden","weitestgehender","weitestgehendes","weitgehend","weitgehende","weitgehendem","weitgehenden","weitgehender","weitgehendes","welche","welchem","welchen","welcher","welches","wem","wen","wenig","wenige","weniger","weniges","wenigstens","wenn","wenngleich","wer","werde","werden","werdet","weshalb","wessen","weswegen","wichtig","wie","wieder","wiederum","wieso","wieviel","wieviele","wievieler","wiewohl","will","willst","wir","wird","wirklich","wirklichem","wirklicher","wirkliches","wirst","wissen","wo","wobei","wodurch","wofuer","wofür","wogegen","woher","wohin","wohingegen","wohl","wohlgemerkt","wohlweislich","wolle","wollen","wollt","wollte","wollten","wolltest","wolltet","womit","womoeglich","womoegliche","womoeglichem","womoeglichen","womoeglicher","womoegliches","womöglich","womögliche","womöglichem","womöglichen","womöglicher","womögliches","woran","woraufhin","woraus","worden","worin","wuerde","wuerden","wuerdest","wuerdet","wurde","würde","wurden","würden","wurdest","würdest","wurdet","würdet","www","x","y","z","z.b","z.B.","zahlreich","zahlreichem","zahlreicher","zB","zb.","zehn","zehnte","zehnten","zehnter","zehntes","zeit","zeitweise","zeitweisem","zeitweisen","zeitweiser","ziehen","zieht","ziemlich","ziemliche","ziemlichem","ziemlichen","ziemlicher","ziemliches","zirka","zog","zogen","zu","zudem","zuerst","zufolge","zugleich","zuletzt","zum","zumal","zumeist","zumindest","zunächst","zunaechst","zur","zurück","zurueck","zusammen","zusehends","zuviel","zuviele","zuvieler","zuweilen","zwanzig","zwar","zwei","zweifelsfrei","zweifelsfreie","zweifelsfreiem","zweifelsfreien","zweifelsfreier","zweifelsfreies","zweite","zweiten","zweiter","zweites","zwischen","zwölf"]
            elif stopword_selection=="Use a built-in list of stop words in English":
                word_stopwords=['a','about','above','after','again','against','ain','all','am','an','and','any','are','aren',"aren't",'as','at','be','because','been','before','being','below','between','both','but','by','can','couldn',"couldn't",'d','did','didn',"didn't",'do','does','doesn',"doesn't",'doing','don',"don't",'down','during','each','few','for','from','further','had','hadn',"hadn't",'has','hasn',"hasn't",'have','haven',"haven't",'having','he','her','here','hers','herself','him','himself','his','how','i','if','in','into','is','isn',"isn't",'it',"it's",'its','itself','just','ll','m','ma','me','mightn',"mightn't",'more','most','mustn',"mustn't",'my','myself','needn',"needn't",'no','nor','not','now','o','of','off','on','once','only','or','other','our','ours','ourselves','out','over','own','re','s','same','shan',"shan't",'she',"she's",'should',"should've",'shouldn',"shouldn't",'so','some','such','t','than','that',"that'll",'the','their','theirs','them','themselves','then','there','these','they','this','those','through','to','too','under','until','up','ve','very','was','wasn',"wasn't",'we','were','weren',"weren't",'what','when','where','which','while','who','whom','why','will','with','won',"won't",'wouldn',"wouldn't",'y','you',"you'd","you'll","you're","you've",'your','yours','yourself','yourselves']
            elif stopword_selection=="Specify stop words":
                word_stopwords=[]
                user_stopwords=st.text_area('Please enter or copy stop words here', value='', height=200, key = session_state.id )
                if len(user_stopwords)>0:
                    stopwords_cv = CountVectorizer()
                    stopwords_cv_fit=stopwords_cv.fit_transform([user_stopwords])                    
                    word_stopwords=stopwords_cv.get_feature_names()
                                        
                st.write("")

            a4,a5=st.beta_columns(2)
            with a4:
                # user specification of words to search
                word_list=pd.DataFrame(columns=word_sorted.index)                
                #words_cleaned=word_list.drop(word_stopwords,axis=1)
                words_cleaned=sorted(list(set(word_list)-set(word_stopwords)))       
                
                find_words=st.multiselect("Search sentences with following words", 
                    words_cleaned, key = session_state.id)
            with a5:
                #user-specification of n-grams
                user_ngram=st.number_input("Specify the number of words to be extracted (n-grams)", min_value=1, value=2, key = session_state.id)
            
            if st.checkbox('Show a word count', value = False, key = session_state.id): 
                st.write(word_sorted)  
            
            st.write("")
            number_remove=st.checkbox("Remove numbers from text", value=True, key = session_state.id)  
                       
            a4,a5=st.beta_columns(2)
            with a4:
                #WordCloud color specification               
                st.write("")
                draw_WordCloud=st.checkbox("Create a Word Cloud", value=True, key = session_state.id)  
            
            with a5:    
                if draw_WordCloud==True:              
                    #color options for the WordCloud (user selection)
                    color_options= pd.DataFrame(np.array([[21, 120, 12, 240, 30]]), 
                    columns=['orange', 'green', 'red','blue','brown'])
                    
                    user_color_name=st.selectbox('Select the main color of your WordCloud',color_options.columns, key = session_state.id)
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
                wfreq_output = st.beta_expander("Basic NLP metrics and visualisations ", expanded = False)
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
                    
                    a4,a5=st.beta_columns(2)
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

                    a4,a5=st.beta_columns(2)
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
                                

                    a4,a5=st.beta_columns(2)
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
                        sentences_output = st.beta_expander("Sentences with specific words", expanded = False)
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
                    ngram_output = st.beta_expander("n-grams", expanded = False)
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
                

                             
                            
            
    
