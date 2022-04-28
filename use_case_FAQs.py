#----------------------------------------------------------------------------------------------

####################
# IMPORT LIBRARIES #
####################

import streamlit as st
import functions as fc

#----------------------------------------------------------------------------------------------

def app():

    #------------------------------------------------------------------------------------------
    # SETTINGS

    settings_expander=st.sidebar.expander('Settings')
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

    #------------------------------------------------------------------------------------------
    # FAQs

    st.header("**FAQs**")
    st.markdown("If your question is not answered, please [contact us](mailto:staty@quant-works.de?subject=Staty-App)!")
        
    general_questions_container = st.container()
    with general_questions_container:
        st.subheader("General questions")
        with st.expander("Which libraries were used for figures?"):
            st.write("A combination of [plotly] (https://plotly.com/python/) and [altair] (https://altair-viz.github.io/index.html) was used for creating figures.")
        with st.expander("Why is my data file not uploading?"):
            st.write("Data can only be uploaded as .csv or .txt file with a maximum size of 200MB.")
        with st.expander("Why are variables not identified correctly after uploading?"):
            st.write("Variables must be seperated either by a semicolon ( ; ), a comma ( , ) or a tabstop to be identified correctly.")
        with st.expander("Why does the row index not coincide with the index identified for rows with NA/duplicates?"):
            st.write("The counting of rows in STATY starts with 0 and not 1. If you have differing indices for rows, please delete the information about row indices prior to uploading your data.")
        with st.expander("Why is a variable not transformed?"):
            st.write("Only numeric variables can be transformed. Please check the corresponding variable type.")
        with st.expander("Why are there still NAs after choosing to replace NAs by grouping?"):
            st.write("NAs might still occur, because the variable selected for grouping might have no observations for a specific group.")
        with st.expander("Why is the model showing an error?"):
            st.write("This can have several reasons. Please make sure to read the error message carefully. In some cases this can be traced back to the model inputs. For example, highly correlated variables or the inclusion of variable x and a transformation of variable x might lead to difficulties in solving for the right model parameters.")
        with st.expander("What are robust covariance matrices and when are they needed?"):
            st.write("For more information about robust covariance matrices please check Chapter 5.1 from 'Introduction to Econometrics' by James H. Stock and Mark W. Watson, and the [publication] (https://link.springer.com/article/10.3758/BF03192961) 'Using heteroskedasticity-consistent standard error estimators in OLS regression: An introduction and software implementation' by Andrew F. Hayes and Li Cai.")
        with st.expander("What to do if heteroskedasticity is present?"):
            st.write("For guidelines on how to deal with heteroskedasticity please check Chapter 5.4 from 'Introduction to Econometrics' by James H. Stock and Mark W. Watson.")
        with st.expander("Why are the new data not loading correctly?"):
            st.write("For the upload of new data in 'Multivariate Data' or 'Panel data' the same upload settings are used as for the original data. Please make sure that the data characteristics match.")
    
    uniBivariate_questions_container = st.container()
    with uniBivariate_questions_container:
        st.subheader("Uni- and bivariate data")
        with st.expander("Which libraries are used within uni- and bivariate data analyses?"):
            st.write("ANOVA uses [statsmodels] (https://www.statsmodels.org/stable/index.html), distribution fitting is based on [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html), while the regression models use [sklearn] (https://scikit-learn.org/stable/).")
        with st.expander("How to change the order of x-labels within the univariate frequency analysis?"):
            st.write("Select “Show additional…settings” and select the x-labels in the order in which you want them to be displayed on the charts and in the tables.")
        with st.expander("What theoretical distributions can be fitted within the 'Distribution fitting' section?"):
            st.write("You can fit more than 90 continuous distributions from [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html).")
        with st.expander("What does it mean “non-numerical variables are factorized to enable regression analysis“?"):
            st.write("To enable correlation analysis, non-numerical variables are automatically transformed to numerical variables through factorisation – i.e., each unique sample simple gets an integer identifier as a new value.")
        with st.expander("Why my variables are not suitable for regression analysis?"):
            st.write("Regression analysis can be performed only with the numerical variables.")
        with st.expander("I want to perform a contingency analysis but I got an error that my dataset has too many unique values?"):
            st.write("Contingency analysis is not suitable for continuous random variables or discrete ones where the number of unique values is large. You can try reclassifying your data, that is, you can try grouping them into classes and use the class frequencies for the contingency analysis.")

    multivariate_questions_container = st.container()
    with multivariate_questions_container:
        st.subheader("Multivariate data")
        with st.expander("Which libraries are used for the multivariate data regression models?"):
            st.write("Models are based on [sklearn] (https://scikit-learn.org/stable/), [statsmodels] (https://www.statsmodels.org/stable/index.html) and [pygam] (https://pygam.readthedocs.io/en/latest/index.html). Please also check the documentation!")
        with st.expander("Which libraries are used for multivariate data decomposition?"):
            st.write("Principal Component Analysis is based on [sklearn] (https://scikit-learn.org/stable/), and Factor Analysis is based on [factor-analyzer] (https://factor-analyzer.readthedocs.io/en/latest/index.html#). Please also check the documentation!")
        with st.expander("Which models can be tuned?"):
            st.write("Hyperparameter ranges can only be specified for Random Forest, Boosted Regression Trees and Artificial Neural Networks.")
        with st.expander("How are models tuned?"):
            st.markdown("Based on the selected search method, several parameter combinations are tested to achieve the best possible performance of the model. To determine the best hyperparameter combination, the models are trained with 80% of the data using (stratified, if binary) cross-validation. After determining the best model either by using R² (continuous) or AUC (binary) from the cross-validation, the model is tested on unseen data, which consist of the remaining 20% of the data.")
        with st.expander("Are models trained with transformed data?"):
            st.write("No, except for Artificial Neural Networks. Artificial Neural Networks are sensitive to different scales of the data. Therefore, the data is re-scaled by standardization, such that the mean is 0 and the variance is 1. In case of testing during hyperparameter-tuning or model validation, the same scaling is applied to the test data set.")
        with st.expander("Why is the hyperparameter 'maximum number of features' not shown for Random Forest in hyperparameter-tuning?"):
            st.write("The hyperparameter is not shown when only one explanatory variable is included. If this is the case, the hyperparameter is automatically set to 1.")
        with st.expander("How many hidden layers can be included in Artificial Neural Networks?"):
            st.write("Currently, the number of hidden layers is restriced to three layers.")
        with st.expander("Why is there no partial dependence plot for some variables?"):
            st.write("If no curve for certain variables is shown, please check the names of the variables. Make sure that variable names do not contain any dots, e.g. my.variable, rather use my_variable.")

    panel_questions_container = st.container()
    with panel_questions_container:
        st.subheader("Panel data")
        with st.expander("Which python libraries are used for the panel data models?"):
            st.write("Models are based on [linearmodels] (https://bashtage.github.io/linearmodels/doc/).  Please also check the documentation for links!")
        with st.expander("How are effects determined for Fixed Effects models?"):
            st.write("The single effects of the entity, time or both are determined with the within estimation method. They can be additionally determined with the Least Squares Dummy Variable (LSDV) method, which leads to the same results.")
        with st.expander("How are effects determined for Random Effects models?"):
            st.write("The single effects of the entities are determined with the quasi-demeaned estimation method.")
    
    timeseries_questions_container = st.container()
    with timeseries_questions_container:
        st.subheader("Time series data")
        with st.expander("Which libraries are used for time series models?"):
            st.write("For the manual model calibration we used statsmodels [SARIMAX]( https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_sarimax_stata.html).")
            st.write("For automatic model calibration we used [pmdarima] (http://alkaline-ml.com/pmdarima/), which is equivalent of R's auto.arima functionality.")
        with st.expander("What is the suitable format for the time variable of my dataset?"):
            st.write("Pandas object with a datetime index - please check [pandas.to_datetime] (https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html) requirements.")
        with st.expander("What kind of detrending and seasonal adjustment is used in STATY?"):
            st.write("We used the function [seasonal_decompose] (https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html) from statsmodels which is a seasonal decomposition using moving averages. Please note that this is a naive additive decomposition.")
        with st.expander("Can I use n-order differences of my time-series as a starting points for my model?"):
            st.write("Sure- look for a drop-down menu called “Select data for further modelling” and select the data you want to use.")
        
    webAndText_questions_container = st.container()
    with webAndText_questions_container:
        st.subheader("Web scraping and text data")
        with st.expander("Which libraries are used for web scraping and text data analyses?"):
            st.write("Text processing was mainly done using CountVectorizer from [sklearn] (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html).")
            st.write("For the the web-page summary we used [pysummarization] (https://code.accel-brain.com/Automatic-Summarization/), which is using a kind of natural language processing and neural network language model.")
            st.write("WordCloud analysis is based on the library [wordcloud] (https://pypi.org/project/wordcloud/).")
            st.write("To connect to Yahoo Finance, we used the library [yfinance] (https://pypi.org/project/yfinance/).")
        with st.expander("Where the data about the stocks are coming from?"):
            st.write("We connect to [Yahoo Finance] (https://www.yahoo.com/ ) to get the data about the stock performance.")
        with st.expander("Can I use any script within the WordCloud analysis?"):
            st.write("Logographic scripts are not supported.")

    geospatial_questions_container = st.container()
    with geospatial_questions_container:
        st.subheader("Geospatial data")
        with st.expander("Which libraries are used for geospatial data processing?"):
            st.write("We used the library [json] (https://docs.python.org/3/library/json.html).")
        with st.expander("What data are used for country boundaries?"):
            st.write("We used 110m resolution [geojson] (http://geojson.xyz/) based on Natural Earth Data.")

