# FUNCTIONS FOR THE MACHINE LEARNING SECTIONS

#############
# FUNCTIONS #
#############

import pandas as pd
import numpy as np
import streamlit as st
import itertools 
import time
import statsmodels.api as sm

# Modelling specifications
import mlp_wrapper as mlp
import sklearn
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import ensemble
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, roc_auc_score, max_error, log_loss, average_precision_score, precision_recall_curve, auc, roc_curve, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, balanced_accuracy_score, cohen_kappa_score, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
#from sklearn.inspection import plot_partial_dependence, partial_dependence, permutation_importance
from sklearn.inspection import PartialDependenceDisplay, partial_dependence, permutation_importance
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer 
from skopt.utils import use_named_args
from skopt import gp_minimize, forest_minimize
from scipy import stats
from scipy.linalg import toeplitz 
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pygam import LinearGAM, LogisticGAM, s
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier


#----------------------------------------------------------------------------------------------
#FUNCTION FOR CREATING SEQUENCEs FOR PARAMETER RANGES

def create_seq(min_val, max_val, step, dec_places):
    if min_val == max_val:
        x = [round(min_val, dec_places)]
    elif min_val < max_val:
        x = [round(min_val, dec_places)]
        while min_val + step < max_val:
            min_val +=step
            x.append(round(min_val, dec_places))
        if min_val + step >= max_val:
            x.append(round(max_val, dec_places))
    return x
  
#----------------------------------------------------------------------------------------------
#FUNCTION FOR HYPERPARAMETER TUNING
#@st.cache(suppress_st_warning = True)
def model_tuning(data, algorithms, hypTune_method, hypTune_iter, hypTune_nCV, hyPara_values, response_var_type, response_var, expl_var):
    
    # Progress bar
    st.info("Tuning progress")
    my_bar = st.progress(0.0)
    progress = 0
    algs_with_tuning = list(["Random Forest", "Boosted Regression Trees", "Artificial Neural Networks"])
    algs_no = sum(al in algorithms for al in algs_with_tuning)
    
    # Save results
    tuning_results = {}

    #-----------------------------------------------------------------------------------------
    # Objective function for continuous response variables

    if response_var_type == "continuous":
        scoring_function = make_scorer(r2_score, greater_is_better = True)

    #-----------------------------------------------------------------------------------------
    # Objective function for binary response variables

    if response_var_type == "binary":
        scoring_function = make_scorer(roc_auc_score, greater_is_better = True, needs_proba = True)

    #-----------------------------------------------------------------------------------------
    # Objective function for multi-class response variables

    if response_var_type == "multi-class":
        scoring_function = make_scorer(accuracy_score, greater_is_better = True)

    #-----------------------------------------------------------------------------------------
    # Split data into training and test data (always 80%/20%)

    Y_data = data[response_var]
    X_data = data[expl_var]
    if response_var_type == "continuous":
        X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, train_size = 0.8, random_state = 1)
    if response_var_type == "binary":
        X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, train_size = 0.8, random_state = 1, stratify = Y_data)
    if response_var_type == "multi-class":
        X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, train_size = 0.8, random_state = 1, stratify = Y_data)

    #---------------------------------------------------------------------------------
    # Random Forest
    if any(a for a in algorithms if a == "Random Forest"):

        # Results table
        rf_tuning_results = pd.DataFrame(index = ["value"], columns = ["scoring", "number of models", "number of trees", "maximum tree depth", "maximum number of features", "sample rate", "mean score", "std score", "test score"])
        
        # Different scoring metrics for response variable type
        if response_var_type == "continuous":
            rf_tuning_results["scoring"] = "% VE"
        if response_var_type == "binary":
            rf_tuning_results["scoring"] = "AUC ROC"
        if response_var_type == "multi-class":
            rf_tuning_results["scoring"] = "accuracy"

        # Extract hyperparameter range settings
        min_not = hyPara_values["rf"]["number of trees"]["min"]
        max_not = hyPara_values["rf"]["number of trees"]["max"]
        min_mtd = hyPara_values["rf"]["maximum tree depth"]["min"]
        max_mtd = hyPara_values["rf"]["maximum tree depth"]["max"]
        min_mnof = hyPara_values["rf"]["maximum number of features"]["min"]
        max_mnof = hyPara_values["rf"]["maximum number of features"]["max"]
        min_sr = hyPara_values["rf"]["sample rate"]["min"]
        max_sr = hyPara_values["rf"]["sample rate"]["max"]
        if hypTune_method == "grid-search" or hypTune_method == "random grid-search":
            not_seq = create_seq(min_not, max_not, step = 5, dec_places = 0)
            if min_mtd is not None:
                mtd_seq = create_seq(min_mtd, max_mtd, step = 1, dec_places = 0)
            mnof_seq = create_seq(min_mnof, max_mnof, step = 1, dec_places = 0)
            sr_seq = create_seq(min_sr, max_sr, step = 0.05, dec_places = 2)
        if hypTune_method == "Bayes optimization":
            not_seq = Integer(min_not,max_not)
            if min_mtd is not None:
                mtd_seq = Integer(min_mtd,max_mtd)
            mnof_seq = Integer(min_mnof,max_mnof)
            sr_seq = Real(min_sr, max_sr)
        
        # Set parameter space
        if hypTune_method == "grid-search" or hypTune_method == "random grid-search" or hypTune_method == "Bayes optimization":
            if min_mtd is not None:
                param_space = dict(n_estimators = not_seq, max_depth = mtd_seq, max_features = mnof_seq, max_samples = sr_seq,)
            else: 
                param_space = dict(n_estimators = not_seq, max_features = mnof_seq, max_samples = sr_seq,)
        if hypTune_method == "sequential model-based optimization":
            if min_mtd is not None:
                param_space =  [
                Integer(min_not, max_not, name = "n_estimators"),
                Integer(min_mtd, max_mtd, name = "max_depth"),
                Integer(min_mnof, max_mnof, name = "max_features"),
                Real(min_sr, max_sr, name = "max_samples")
                ]  
            else:        
                param_space =  [
                Integer(min_not, max_not, name = "n_estimators"),
                Integer(min_mnof, max_mnof, name = "max_features"),
                Real(min_sr, max_sr, name = "max_samples")
                ]  

        # Model specification for continuous variables
        if response_var_type == "continuous":
            # Model
            rf = ensemble.RandomForestRegressor()
            # Cross-validation technique
            cv = RepeatedKFold(n_splits = hypTune_nCV, n_repeats = 1, random_state = 0)
        # Model specification for binary variables
        if response_var_type == "binary":
            # Model
            rf = ensemble.RandomForestClassifier()
            # Cross-validation technique
            cv = RepeatedStratifiedKFold(n_splits = hypTune_nCV, n_repeats = 1, random_state = 0)
        # Model specification for multi-class variables
        if response_var_type == "multi-class":
            # Model
            rf = ensemble.RandomForestClassifier()
            # Cross-validation technique
            cv = RepeatedStratifiedKFold(n_splits = hypTune_nCV, n_repeats = 1, random_state = 0)

        # Search method
        # Grid-search
        if hypTune_method == "grid-search":
            rf_grid = GridSearchCV(estimator = rf, param_grid = param_space, scoring = scoring_function, cv = cv, verbose = 1)#, n_jobs = -1)
        # Random grid-search
        if hypTune_method == "random grid-search":
            rf_grid = RandomizedSearchCV(estimator = rf, param_distributions = param_space, n_iter = hypTune_iter, scoring = scoring_function, cv = cv, verbose = 1)#, n_jobs = -1)
        # Bayes search
        if hypTune_method == "Bayes optimization":
            rf_grid = BayesSearchCV(estimator = rf, search_spaces = param_space, n_iter = hypTune_iter, scoring = scoring_function, cv = cv, verbose = 1)#, n_jobs = -1)
        # Sequential model-based optimization 
        if hypTune_method == "sequential model-based optimization":
            # Define new objective function
            @use_named_args(param_space)
            def objective(**params):
                rf.set_params(**params)
                return -np.mean(cross_val_score(rf, X_train, Y_train, cv = cv, scoring = scoring_function))#, n_jobs = -1))

        # Grid-search, random grid-search and Bayes search results
        if hypTune_method == "grid-search" or hypTune_method == "random grid-search" or hypTune_method == "Bayes optimization":
            # Search results
            rf_grid_test_fit = rf_grid.fit(X_train, Y_train)

            # Extract number of models
            if hypTune_method == "grid-search" or hypTune_method == "random grid-search":
                rf_tuning_results["number of models"] = rf_grid_test_fit.cv_results_["mean_test_score"].size
            if hypTune_method == "Bayes optimization":
                rf_tuning_results["number of models"] = len(rf_grid_test_fit.cv_results_["mean_test_score"])
            
            # Extract best model parameters (parameter setting that gave the best results on the hold out data)
            rf_final_para = rf_grid_test_fit.best_params_
            rf_tuning_results["number of trees"] = rf_final_para.get("n_estimators")
            if min_mtd is not None:
                rf_tuning_results["maximum tree depth"] = rf_final_para.get("max_depth")
            else:
                rf_tuning_results["maximum tree depth"] = str(None)
            rf_tuning_results["maximum number of features"] = rf_final_para.get("max_features")
            rf_tuning_results["sample rate"] = rf_final_para.get("max_samples")

            # Extract best score (mean cross-validated score of the best_estimator) and standard deviation
            rf_tuning_results["mean score"] = rf_grid_test_fit.best_score_
            rf_tuning_results["std score"] = rf_grid_test_fit.cv_results_["std_test_score"][rf_grid_test_fit.best_index_]

            # Extract parameters for testing on test data set
            if min_mtd is not None:
                params = {'n_estimators': rf_final_para.get("n_estimators"),
                    'max_depth': rf_final_para.get("max_depth"),
                    'max_features': rf_final_para.get("max_features"),
                    'max_samples': rf_final_para.get("max_samples")
                }
            else:
                params = {'n_estimators': rf_final_para.get("n_estimators"),
                    'max_features': rf_final_para.get("max_features"),
                    'max_samples': rf_final_para.get("max_samples")
                }

        # Sequential model-based optimization results
        if hypTune_method == "sequential model-based optimization":
            rf_grid_test_fit = forest_minimize(objective, param_space, n_calls = hypTune_iter, verbose = 1)#, n_jobs = -1)

            # Extract number of models
            rf_tuning_results["number of models"] = rf_grid_test_fit.func_vals.size

            # Extract best model parameters (order important!)
            rf_tuning_results["number of trees"] = rf_grid_test_fit.x[0]
            if min_mtd is not None:
                rf_tuning_results["maximum tree depth"] = rf_grid_test_fit.x[1]
                rf_tuning_results["maximum number of features"] = rf_grid_test_fit.x[2]
                rf_tuning_results["sample rate"] = rf_grid_test_fit.x[3]
            else:
                rf_tuning_results["maximum tree depth"] = str(None)
                rf_tuning_results["maximum number of features"] = rf_grid_test_fit.x[1]
                rf_tuning_results["sample rate"] = rf_grid_test_fit.x[2]

            # Extract parameters for testing on test data set
            if min_mtd is not None:
                params = {'n_estimators': rf_tuning_results["number of trees"][0],
                    'max_depth': rf_tuning_results["maximum tree depth"][0],
                    'max_features': rf_tuning_results["maximum number of features"][0],
                    'max_samples': rf_tuning_results["sample rate"][0]
                }
            else:
                params = {'n_estimators': rf_tuning_results["number of trees"][0],
                    'max_features': rf_tuning_results["maximum number of features"][0],
                    'max_samples': rf_tuning_results["sample rate"][0]
                }

            # Re-run cv for best paras in SMBO
            if response_var_type == "continuous":
                cv_best_para = cross_val_score(ensemble.RandomForestRegressor(**params), X_train, Y_train, cv = cv, scoring = scoring_function)#, n_jobs = -1)
            if response_var_type == "binary":
                cv_best_para = cross_val_score(ensemble.RandomForestClassifier(**params), X_train, Y_train, cv = cv, scoring = scoring_function)#, n_jobs = -1)
            if response_var_type == "multi-class":
                cv_best_para = cross_val_score(ensemble.RandomForestClassifier(**params), X_train, Y_train, cv = cv, scoring = scoring_function)#, n_jobs = -1)

            # Extract best score (mean cross-validated score of the best_estimator) and standard deviation
            rf_tuning_results["mean score"] = np.mean(cv_best_para)
            rf_tuning_results["std score"] = np.std(cv_best_para)
        
        # Test final parameters on test dataset
        if response_var_type == "continuous":
            rf_final_model = ensemble.RandomForestRegressor(**params)
        if response_var_type == "binary":
            rf_final_model = ensemble.RandomForestClassifier(**params)
        if response_var_type == "multi-class":
            rf_final_model = ensemble.RandomForestClassifier(**params)
        rf_final_model.fit(X_train, Y_train)
    
        # Prediction for Y_test (continuous)
        if response_var_type == "continuous":
            Y_test_pred = rf_final_model.predict(X_test)
        # Prediction of probability for Y_test (binary)
        if response_var_type == "binary":
            Y_test_pred = rf_final_model.predict_proba(X_test)[:, 1]
        # Prediction for Y_test (multi-class)
        if response_var_type == "multi-class":
            Y_test_pred = rf_final_model.predict(X_test)

        # R² for test data (continuous)
        if response_var_type == "continuous":
            rf_tuning_results["test score"] = r2_score(Y_test, Y_test_pred)
        # AUC for test data (binary)
        if response_var_type == "binary":
            rf_tuning_results["test score"] = roc_auc_score(Y_test, Y_test_pred)
        # Accuracy for test data (multi-class)
        if response_var_type == "multi-class":
            rf_tuning_results["test score"] = accuracy_score(Y_test, Y_test_pred)

        # Save results
        tuning_results["rf tuning"] = rf_tuning_results

        progress += 1
        my_bar.progress(progress/algs_no)

    #---------------------------------------------------------------------------------
    # Boosted Regression Trees
    if any(a for a in algorithms if a == "Boosted Regression Trees"):

        # Results table
        brt_tuning_results = pd.DataFrame(index = ["value"], columns = ["scoring", "number of models", "number of trees", "learning rate", "maximum tree depth", "sample rate", "mean score", "std score", "test score"])
        
        # Different scoring metrics for response variable type
        if response_var_type == "continuous":
            brt_tuning_results["scoring"] = "% VE"
        if response_var_type == "binary":
            brt_tuning_results["scoring"] = "AUC ROC"

        # Extract hyperparameter range settings
        min_not = hyPara_values["brt"]["number of trees"]["min"]
        max_not = hyPara_values["brt"]["number of trees"]["max"]
        min_lr = hyPara_values["brt"]["learning rate"]["min"]
        max_lr = hyPara_values["brt"]["learning rate"]["max"]
        min_mtd = hyPara_values["brt"]["maximum tree depth"]["min"]
        max_mtd = hyPara_values["brt"]["maximum tree depth"]["max"]
        min_sr = hyPara_values["brt"]["sample rate"]["min"]
        max_sr = hyPara_values["brt"]["sample rate"]["max"]
        if hypTune_method == "grid-search" or hypTune_method == "random grid-search":
            not_seq = create_seq(min_not, max_not, step = 5, dec_places = 0)
            lr_seq = create_seq(min_lr, max_lr, step = 0.001, dec_places = 3)
            mtd_seq = create_seq(min_mtd, max_mtd, step = 1, dec_places = 0)
            sr_seq = create_seq(min_sr, max_sr, step = 0.05, dec_places = 2)
        if hypTune_method == "Bayes optimization":
            not_seq = Integer(min_not,max_not)
            lr_seq = Real(min_lr, max_lr)
            mtd_seq = Integer(min_mtd,max_mtd)
            sr_seq = Real(min_sr, max_sr)
        
        # Set parameter space
        if hypTune_method == "grid-search" or hypTune_method == "random grid-search" or hypTune_method == "Bayes optimization":
            param_space = dict(n_estimators = not_seq, learning_rate = lr_seq, max_depth = mtd_seq, subsample = sr_seq,)
        if hypTune_method == "sequential model-based optimization":
            param_space =  [
            Integer(min_not, max_not, name = "n_estimators"),
            Real(min_lr, max_lr, name = "learning_rate"),
            Integer(min_mtd, max_mtd, name = "max_depth"),
            Real(min_sr, max_sr, name = "subsample")
            ]           
        
        # Model specification for continuous variables
        if response_var_type == "continuous":
            # Model
            brt = ensemble.GradientBoostingRegressor()
            # Cross-validation technique
            cv = RepeatedKFold(n_splits = hypTune_nCV, n_repeats = 1, random_state = 0)
        # Model specification for binary variables
        if response_var_type == "binary":
            # Model
            brt = ensemble.GradientBoostingClassifier()
            # Cross-validation technique
            cv = RepeatedStratifiedKFold(n_splits = hypTune_nCV, n_repeats = 1, random_state = 0)

        # Search method
        # Grid-search
        if hypTune_method == "grid-search":
            brt_grid = GridSearchCV(estimator = brt, param_grid = param_space, scoring = scoring_function, cv = cv, verbose = 1)#, n_jobs = -1)
        # Random grid-search
        if hypTune_method == "random grid-search":
            brt_grid = RandomizedSearchCV(estimator = brt, param_distributions = param_space, n_iter = hypTune_iter, scoring = scoring_function, cv = cv, verbose = 1)#, n_jobs = -1)
        # Bayes search
        if hypTune_method == "Bayes optimization":
            brt_grid = BayesSearchCV(estimator = brt, search_spaces = param_space, n_iter = hypTune_iter, scoring = scoring_function, cv = cv, verbose = 1)#, n_jobs = -1)
        # Sequential model-based optimization
        if hypTune_method == "sequential model-based optimization":
            # Define new objective function
            @use_named_args(param_space)
            def objective(**params):
                brt.set_params(**params)
                return -np.mean(cross_val_score(brt, X_train, Y_train, cv = cv, scoring = scoring_function))#, n_jobs = -1))

        # Grid-search, random grid-search and Bayes search results
        if hypTune_method == "grid-search" or hypTune_method == "random grid-search" or hypTune_method == "Bayes optimization":
            # Search results
            brt_grid_test_fit = brt_grid.fit(X_train, Y_train.values.ravel())

            # Extract number of models
            if hypTune_method == "grid-search" or hypTune_method == "random grid-search":
                brt_tuning_results["number of models"] = brt_grid_test_fit.cv_results_["mean_test_score"].size
            if hypTune_method == "Bayes optimization":
                brt_tuning_results["number of models"] = len(brt_grid_test_fit.cv_results_["mean_test_score"])
            
            # Extract best model parameters (parameter setting that gave the best results on the hold out data)
            brt_final_para = brt_grid_test_fit.best_params_
            brt_tuning_results["number of trees"] = brt_final_para.get("n_estimators")
            brt_tuning_results["learning rate"] = brt_final_para.get("learning_rate")
            brt_tuning_results["maximum tree depth"] = brt_final_para.get("max_depth")
            brt_tuning_results["sample rate"] = brt_final_para.get("subsample")

            # Extract best score (mean cross-validated score of the best_estimator) and standard deviation
            brt_tuning_results["mean score"] = brt_grid_test_fit.best_score_
            brt_tuning_results["std score"] = brt_grid_test_fit.cv_results_["std_test_score"][brt_grid_test_fit.best_index_]

            # Extract parameters for testing on test data set
            params = {'n_estimators': brt_final_para.get("n_estimators"),
                'learning_rate': brt_final_para.get("learning_rate"),
                'max_depth': brt_final_para.get("max_depth"),
                'subsample': brt_final_para.get("subsample")
            }

        # Sequential model-based optimization results
        if hypTune_method == "sequential model-based optimization":
            brt_grid_test_fit = forest_minimize(objective, param_space, n_calls = hypTune_iter, verbose = 1)#, n_jobs = -1)

            # Extract number of models
            brt_tuning_results["number of models"] = brt_grid_test_fit.func_vals.size

            # Extract best model parameters (order important!)
            brt_tuning_results["number of trees"] = brt_grid_test_fit.x[0]
            brt_tuning_results["learning rate"] = brt_grid_test_fit.x[1]
            brt_tuning_results["maximum tree depth"] = brt_grid_test_fit.x[2]
            brt_tuning_results["sample rate"] = brt_grid_test_fit.x[3]

            # Extract parameters for testing on test data set
            params = {'n_estimators': brt_tuning_results["number of trees"][0],
                'learning_rate': brt_tuning_results["learning rate"][0],
                'max_depth': brt_tuning_results["maximum tree depth"][0],
                'subsample': brt_tuning_results["sample rate"][0]
            }

            # Re-run cv for best paras in SMBO
            if response_var_type == "continuous":
                cv_best_para = cross_val_score(ensemble.GradientBoostingRegressor(**params), X_train, Y_train, cv = cv, scoring = scoring_function)#, n_jobs = -1)
            if response_var_type == "binary":
                cv_best_para = cross_val_score(ensemble.GradientBoostingClassifier(**params), X_train, Y_train, cv = cv, scoring = scoring_function)#, n_jobs = -1)

            # Extract best score (mean cross-validated score of the best_estimator) and standard deviation
            brt_tuning_results["mean score"] = np.mean(cv_best_para)
            brt_tuning_results["std score"] = np.std(cv_best_para)
        
        # Test final parameters on test dataset
        if response_var_type == "continuous":
            brt_final_model = ensemble.GradientBoostingRegressor(**params)
        if response_var_type == "binary":
            brt_final_model = ensemble.GradientBoostingClassifier(**params)
        brt_final_model.fit(X_train, Y_train.values.ravel())
    
        # Prediction for Y_test (continuous)
        if response_var_type == "continuous":
            Y_test_pred = brt_final_model.predict(X_test)
        # Prediction of probability for Y_test (binary)
        if response_var_type == "binary":
            Y_test_pred = brt_final_model.predict_proba(X_test)[:, 1]

        # R² for test data (continuous)
        if response_var_type == "continuous":
            brt_tuning_results["test score"] = r2_score(Y_test, Y_test_pred)
        # AUC for test data (binary)
        if response_var_type == "binary":
            brt_tuning_results["test score"] = roc_auc_score(Y_test, Y_test_pred)

        # Save results
        tuning_results["brt tuning"] = brt_tuning_results

        progress += 1
        my_bar.progress(progress/algs_no)

    #---------------------------------------------------------------------------------
    # Artificial Neural Networks
    if any(a for a in algorithms if a == "Artificial Neural Networks"):
        
        # Standardize X_data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_ann = scaler.transform(X_train)
        X_test_ann = scaler.transform(X_test)     

        # Results table
        ann_tuning_results = pd.DataFrame(index = ["value"], columns = ["scoring", "number of models", "weight optimization solver", "maximum number of iterations", "activation function", "hidden layer sizes", "learning rate", "L² regularization", "mean score", "std score", "test score"])

        # Different scoring metrics for response variable type
        if response_var_type == "continuous":
            ann_tuning_results["scoring"] = "% VE"
        if response_var_type == "binary":
            ann_tuning_results["scoring"] = "AUC ROC"
        if response_var_type == "multi-class":
            ann_tuning_results["scoring"] = "accuracy"

        # Extract hyperparameter range settings
        woa_list = hyPara_values["ann"]["weight optimization solver"]["min"]
        min_mni =  hyPara_values["ann"]["maximum number of iterations"]["min"]
        max_mni =  hyPara_values["ann"]["maximum number of iterations"]["max"]
        af_list =  hyPara_values["ann"]["activation function"]["min"]
        nhl_value = hyPara_values["ann"]["number of hidden layers"]["min"]
        if nhl_value > 1:
            min_nnhl = hyPara_values["ann"]["nodes per hidden layer"]["min"]
            max_nnhl = hyPara_values["ann"]["nodes per hidden layer"]["max"]
        elif nhl_value == 1:
            min_nnhl = [hyPara_values["ann"]["nodes per hidden layer"]["min"]]
            max_nnhl = [hyPara_values["ann"]["nodes per hidden layer"]["max"]]
        # Restructure hidden layer info for tuning
        hl_nodes = []
        for hln in range(nhl_value):
            hl_nodes.append(list(create_seq(min_nnhl[hln], max_nnhl[hln], step = 2, dec_places = 0)))
        hl_nodes = list(itertools.product(*hl_nodes))
        min_lr = hyPara_values["ann"]["learning rate"]["min"]
        max_lr = hyPara_values["ann"]["learning rate"]["max"]
        #lrs_list = hyPara_values["ann"]["learning rate schedule"]["min"]
        #min_mom = hyPara_values["ann"]["momentum"]["min"]
        #max_mom = hyPara_values["ann"]["momentum"]["max"]
        min_l2reg = hyPara_values["ann"]["L² regularization"]["min"]
        max_l2reg = hyPara_values["ann"]["L² regularization"]["max"]
        #min_eps = hyPara_values["ann"]["epsilon"]["min"]
        #max_eps = hyPara_values["ann"]["epsilon"]["max"]

        if hypTune_method == "grid-search" or hypTune_method == "random grid-search":
            mni_seq = create_seq(min_mni, max_mni, step = 5, dec_places = 0)
            lr_seq = create_seq(min_lr, max_lr, step = 0.0005, dec_places = 4)
            #mom_seq = create_seq(min_mom, max_mom, step = 0.01, dec_places = 2)
            l2reg_seq = create_seq(min_l2reg, max_l2reg, step = 0.00005, dec_places = 5)
            #eps_seq = create_seq(min_eps, max_eps, step = 0.000000005, dec_places = 9)
        if hypTune_method == "Bayes optimization":
            if nhl_value == 1:
                layer1 = Integer(min_nnhl[0], max_nnhl[0])
            if nhl_value == 2:
                layer1 = Integer(min_nnhl[0], max_nnhl[0])
                layer2 = Integer(min_nnhl[1], max_nnhl[1])
            if nhl_value == 3:
                layer1 = Integer(min_nnhl[0], max_nnhl[0])
                layer2 = Integer(min_nnhl[1], max_nnhl[1])
                layer3 = Integer(min_nnhl[2], max_nnhl[2])
            af_list = Categorical(af_list)
            woa_list = Categorical(woa_list)
            #lrs_list = Categorical(lrs_list)
            mni_seq = Integer(min_mni, max_mni)
            lr_seq = Real(min_lr, max_lr)
            #mom_seq = Real(min_mom, max_mom)
            l2reg_seq = Real(min_l2reg, max_l2reg)
            #eps_seq = Real(min_eps, max_eps)
               
        # Set parameter space
        if hypTune_method == "grid-search" or hypTune_method == "random grid-search":
            param_space = dict(hidden_layer_sizes = hl_nodes, activation = af_list, solver = woa_list, alpha = l2reg_seq, learning_rate_init = lr_seq, max_iter = mni_seq,) #epsilon = eps_seq,)
        if hypTune_method == "Bayes optimization":
            if nhl_value == 1:
                param_space = dict(layer1 = layer1, activation = af_list, solver = woa_list, alpha = l2reg_seq, learning_rate_init = lr_seq, max_iter = mni_seq,) #epsilon = eps_seq,)
            if nhl_value == 2:
                param_space = dict(layer1 = layer1, layer2 = layer2, activation = af_list, solver = woa_list, alpha = l2reg_seq, learning_rate_init = lr_seq, max_iter = mni_seq,) #epsilon = eps_seq,)
            if nhl_value == 3:
                param_space = dict(layer1 = layer1, layer2 = layer2, layer3 = layer3, activation = af_list, solver = woa_list, alpha = l2reg_seq, learning_rate_init = lr_seq, max_iter = mni_seq,) #epsilon = eps_seq,)
        if hypTune_method == "sequential model-based optimization":
            if nhl_value == 1:
                param_space =  [
                Integer(min_nnhl[0], max_nnhl[0], name = "layer1"),
                Categorical(af_list, name = "activation"),
                Categorical(woa_list, name = "solver"),
                #Categorical(lrs_list, name = "learning_rate"),
                Integer(min_mni, max_mni, name = "max_iter"),
                Real(min_lr, max_lr, name = "learning_rate_init"),
                #Real(min_mom, max_mom, name = "momentum"),
                Real(min_l2reg, max_l2reg, name = "alpha")
                #Real(min_eps, max_eps, name = "epsilon")
                ]
            if nhl_value == 2:
                param_space =  [
                Integer(min_nnhl[0], max_nnhl[0], name = "layer1"),
                Integer(min_nnhl[1], max_nnhl[1], name = "layer2"),
                Categorical(af_list, name = "activation"),
                Categorical(woa_list, name = "solver"),
                #Categorical(lrs_list, name = "learning_rate"),
                Integer(min_mni, max_mni, name = "max_iter"),
                Real(min_lr, max_lr, name = "learning_rate_init"),
                #Real(min_mom, max_mom, name = "momentum"),
                Real(min_l2reg, max_l2reg, name = "alpha")
                #Real(min_eps, max_eps, name = "epsilon")
                ]
            if nhl_value == 3:
                param_space =  [
                Integer(min_nnhl[0], max_nnhl[0], name = "layer1"),
                Integer(min_nnhl[1], max_nnhl[1], name = "layer2"),
                Integer(min_nnhl[2], max_nnhl[2], name = "layer3"),
                Categorical(af_list, name = "activation"),
                Categorical(woa_list, name = "solver"),
                #Categorical(lrs_list, name = "learning_rate"),
                Integer(min_mni, max_mni, name = "max_iter"),
                Real(min_lr, max_lr, name = "learning_rate_init"),
                #Real(min_mom, max_mom, name = "momentum"),
                Real(min_l2reg, max_l2reg, name = "alpha")
                #Real(min_eps, max_eps, name = "epsilon")
                ]

        # Model specification for continuous variables
        if response_var_type == "continuous":
            # Wrapper for MLP due to hidden_layer_nodes tuple property in Bayesian search
            if hypTune_method == "Bayes optimization" or hypTune_method == "sequential model-based optimization":
                if nhl_value == 1:
                    ann = mlp.MLPWrapperCon_1Layer()
                elif nhl_value == 2:
                    ann = mlp.MLPWrapperCon_2Layer()
                elif nhl_value == 3:
                    ann = mlp.MLPWrapperCon_3Layer()
            elif hypTune_method == "grid-search" or hypTune_method == "random grid-search":
                ann = MLPRegressor()
            # Cross-validation technique
            cv = RepeatedKFold(n_splits = hypTune_nCV, n_repeats = 1, random_state = 0)
        # Model specification for binary variables
        if response_var_type == "binary":
            # Wrapper for MLP due to hidden_layer_nodes tuple property in Bayesian search
            if hypTune_method == "Bayes optimization" or hypTune_method == "sequential model-based optimization":
                if nhl_value == 1:
                    ann = mlp.MLPWrapperBin_1Layer()
                elif nhl_value == 2:
                    ann = mlp.MLPWrapperBin_2Layer()
                elif nhl_value == 3:
                    ann = mlp.MLPWrapperBin_3Layer()
            elif hypTune_method == "grid-search" or hypTune_method == "random grid-search":
                ann = MLPClassifier()
            # Cross-validation technique
            cv = RepeatedStratifiedKFold(n_splits = hypTune_nCV, n_repeats = 1, random_state = 0) 
        # Model specification for multi-class variables
        if response_var_type == "multi-class":
            # Wrapper for MLP due to hidden_layer_nodes tuple property in Bayesian search
            if hypTune_method == "Bayes optimization" or hypTune_method == "sequential model-based optimization":
                if nhl_value == 1:
                    ann = mlp.MLPWrapperBin_1Layer()
                elif nhl_value == 2:
                    ann = mlp.MLPWrapperBin_2Layer()
                elif nhl_value == 3:
                    ann = mlp.MLPWrapperBin_3Layer()
            elif hypTune_method == "grid-search" or hypTune_method == "random grid-search":
                ann = MLPClassifier()
            # Cross-validation technique
            cv = RepeatedStratifiedKFold(n_splits = hypTune_nCV, n_repeats = 1, random_state = 0)            

        # Search method
        # Grid-search
        if hypTune_method == "grid-search":
            ann_grid = GridSearchCV(estimator = ann, param_grid = param_space, scoring = scoring_function, cv = cv, verbose = 1)#, n_jobs = -1)
        # Random grid-search
        if hypTune_method == "random grid-search":
            ann_grid = RandomizedSearchCV(estimator = ann, param_distributions = param_space, n_iter = hypTune_iter, scoring = scoring_function, cv = cv, verbose = 1)#, n_jobs = -1)
        # Bayes search
        if hypTune_method == "Bayes optimization":
            ann_grid = BayesSearchCV(estimator = ann, search_spaces = param_space, n_iter = hypTune_iter, scoring = scoring_function, cv = cv, verbose = 1)#, n_jobs = -1)
        # Sequential model-based optimization
        if hypTune_method == "sequential model-based optimization":
            # Define new objective function
            @use_named_args(param_space)
            def objective1(**params):
                ann.set_params(**params)
                return -np.mean(cross_val_score(ann, X_train_ann, Y_train, cv = cv, scoring = scoring_function))#, n_jobs = -1))

        # Grid-search, random grid-search and Bayes search results
        if hypTune_method == "grid-search" or hypTune_method == "random grid-search" or hypTune_method == "Bayes optimization":
            # Search results
            ann_grid_test_fit = ann_grid.fit(X_train_ann, Y_train)

            # Extract number of models
            if hypTune_method == "grid-search" or hypTune_method == "random grid-search":
                ann_tuning_results["number of models"] = ann_grid_test_fit.cv_results_["mean_test_score"].size
            if hypTune_method == "Bayes optimization":
                ann_tuning_results["number of models"] = len(ann_grid_test_fit.cv_results_["mean_test_score"])
            
            # Extract best model parameters (parameter setting that gave the best results on the hold out data)
            ann_final_para = ann_grid_test_fit.best_params_
            ann_tuning_results["weight optimization solver"] = ann_final_para.get("solver")
            ann_tuning_results["maximum number of iterations"] = ann_final_para.get("max_iter")
            ann_tuning_results["activation function"] = ann_final_para.get("activation")
            if hypTune_method == "Bayes optimization" and nhl_value == 1:
                ann_tuning_results["hidden layer sizes"] = [(ann_final_para.get("layer1"),)]
            if hypTune_method == "Bayes optimization" and nhl_value == 2:
                ann_tuning_results["hidden layer sizes"] = list([(ann_final_para.get("layer1"), ann_final_para.get("layer2"))])
            if hypTune_method == "Bayes optimization" and nhl_value == 3:
                ann_tuning_results["hidden layer sizes"] = list([(ann_final_para.get("layer1"), ann_final_para.get("layer2"), ann_final_para.get("layer3"))])    
            if hypTune_method == "grid-search" or hypTune_method == "random grid-search":
                ann_tuning_results["hidden layer sizes"] = [ann_final_para.get("hidden_layer_sizes")]
            ann_tuning_results["learning rate"] = ann_final_para.get("learning_rate_init")
            #ann_tuning_results["learning rate schedule"] = ann_final_para.get("learning_rate")
            #ann_tuning_results["momentum"] = ann_final_para.get("momentum")
            ann_tuning_results["L² regularization"] = ann_final_para.get("alpha")
            #ann_tuning_results["epsilon"] = ann_final_para.get("epsilon")

            # Extract best score (mean cross-validated score of the best_estimator) and standard deviation
            ann_tuning_results["mean score"] = ann_grid_test_fit.best_score_
            ann_tuning_results["std score"] = ann_grid_test_fit.cv_results_["std_test_score"][ann_grid_test_fit.best_index_]

            # Extract parameters for testing on test data set
            params2 = {
                "solver": ann_tuning_results["weight optimization solver"][0],
                "max_iter": ann_tuning_results["maximum number of iterations"][0],
                "activation": ann_tuning_results["activation function"][0],
                "hidden_layer_sizes": ann_tuning_results["hidden layer sizes"][0],
                "learning_rate_init": ann_tuning_results["learning rate"][0],
                #"learning_rate": ann_tuning_results["learning rate schedule"][0],
                #"momentum": ann_tuning_results["momentum"][0],
                "alpha": ann_tuning_results["L² regularization"][0]
                #"epsilon": ann_tuning_results["epsilon"][0]
            }
        
        # Sequential model-based optimization results
        if hypTune_method == "sequential model-based optimization":
            ann_grid_test_fit = forest_minimize(objective1, param_space, n_calls = hypTune_iter, verbose = 1)#, n_jobs = -1)

            # Extract number of models
            ann_tuning_results["number of models"] = ann_grid_test_fit.func_vals.size

            # Extract best model parameters (order important!)
            if nhl_value == 1:
                ann_tuning_results["hidden layer sizes"] = [(ann_grid_test_fit.x[0],)]
                ann_tuning_results["activation function"] = ann_grid_test_fit.x[1]
                ann_tuning_results["weight optimization solver"] = ann_grid_test_fit.x[2]
                #ann_tuning_results["learning rate schedule"] = ann_grid_test_fit.x[3]
                ann_tuning_results["maximum number of iterations"] = ann_grid_test_fit.x[3]
                ann_tuning_results["learning rate"] = ann_grid_test_fit.x[4]
                #ann_tuning_results["momentum"] = ann_grid_test_fit.x[6]
                ann_tuning_results["L² regularization"] = ann_grid_test_fit.x[5]
                #ann_tuning_results["epsilon"] = ann_grid_test_fit.x[8]
            if nhl_value == 2:
                ann_tuning_results["hidden layer sizes"] = list([(ann_grid_test_fit.x[0], ann_grid_test_fit.x[1])])
                ann_tuning_results["activation function"] = ann_grid_test_fit.x[2]
                ann_tuning_results["weight optimization solver"] = ann_grid_test_fit.x[3]
                #ann_tuning_results["learning rate schedule"] = ann_grid_test_fit.x[4]
                ann_tuning_results["maximum number of iterations"] = ann_grid_test_fit.x[4]
                ann_tuning_results["learning rate"] = ann_grid_test_fit.x[5]
                #ann_tuning_results["momentum"] = ann_grid_test_fit.x[7]
                ann_tuning_results["L² regularization"] = ann_grid_test_fit.x[6]
                #ann_tuning_results["epsilon"] = ann_grid_test_fit.x[9]
            if nhl_value == 3:
                ann_tuning_results["hidden layer sizes"] = list([(ann_grid_test_fit.x[0], ann_grid_test_fit.x[1], ann_grid_test_fit.x[2])])
                ann_tuning_results["activation function"] = ann_grid_test_fit.x[3]
                ann_tuning_results["weight optimization solver"] = ann_grid_test_fit.x[4]
                #ann_tuning_results["learning rate schedule"] = ann_grid_test_fit.x[5]
                ann_tuning_results["maximum number of iterations"] = ann_grid_test_fit.x[5]
                ann_tuning_results["learning rate"] = ann_grid_test_fit.x[6]
                #ann_tuning_results["momentum"] = ann_grid_test_fit.x[8]
                ann_tuning_results["L² regularization"] = ann_grid_test_fit.x[7]
                #ann_tuning_results["epsilon"] = ann_grid_test_fit.x[10]

            # Extract parameters for testing on test data set
            params2 = {
                "solver": ann_tuning_results["weight optimization solver"][0],
                "max_iter": ann_tuning_results["maximum number of iterations"][0],
                "activation": ann_tuning_results["activation function"][0],
                "hidden_layer_sizes": ann_tuning_results["hidden layer sizes"][0],
                "learning_rate_init": ann_tuning_results["learning rate"][0],
                #"learning_rate": ann_tuning_results["learning rate schedule"][0],
                #"momentum": ann_tuning_results["momentum"][0],
                "alpha": ann_tuning_results["L² regularization"][0]
                #"epsilon": ann_tuning_results["epsilon"][0]
            }

            # Re-run cv for best paras in SMBO
            if response_var_type == "continuous":
                cv_best_para = cross_val_score(MLPRegressor(**params2), X_train_ann, Y_train, cv = cv, scoring = scoring_function)#, n_jobs = -1)
            if response_var_type == "binary":
                cv_best_para = cross_val_score(MLPClassifier(**params2), X_train_ann, Y_train, cv = cv, scoring = scoring_function)#, n_jobs = -1)
            if response_var_type == "multi-class":
                cv_best_para = cross_val_score(MLPClassifier(**params2), X_train_ann, Y_train, cv = cv, scoring = scoring_function)#, n_jobs = -1)

            # Extract best score (mean cross-validated score of the best_estimator) and standard deviation
            ann_tuning_results["mean score"] = np.mean(cv_best_para)
            ann_tuning_results["std score"] = np.std(cv_best_para)
        
        # Test final parameters on test dataset
        if response_var_type == "continuous":
            ann_final_model = MLPRegressor(**params2)
        if response_var_type == "binary":
            ann_final_model = MLPClassifier(**params2)
        if response_var_type == "multi-class":
            ann_final_model = MLPClassifier(**params2)
        ann_final_model.fit(X_train_ann, Y_train)
    
        # Prediction for Y_test (continuous)
        if response_var_type == "continuous":
            Y_test_pred = ann_final_model.predict(X_test_ann)
        # Prediction of probability for Y_test (binary)
        if response_var_type == "binary":
            Y_test_pred = ann_final_model.predict_proba(X_test_ann)[:, 1]
        # Prediction for Y_test (multi-class)
        if response_var_type == "multi-class":
            Y_test_pred = ann_final_model.predict(X_test_ann)

        # R² for test data (continuous)
        if response_var_type == "continuous":
            ann_tuning_results["test score"] = r2_score(Y_test, Y_test_pred)
        # AUC for test data (binary)
        if response_var_type == "binary":
            ann_tuning_results["test score"] = roc_auc_score(Y_test, Y_test_pred)
        # Accuracy for test data (multi-class)
        if response_var_type == "multi-class":
            ann_tuning_results["test score"] = accuracy_score(Y_test, Y_test_pred)

        # Save results
        tuning_results["ann tuning"] = ann_tuning_results

        progress += 1
        my_bar.progress(progress/algs_no)

    return tuning_results


#----------------------------------------------------------------------------------------------
#FUNCTION FOR MODEL VALIDATION RUNS
#@st.cache(suppress_st_warning = True)
def model_val(data, algorithms, MLR_model, train_frac, val_runs, response_var_type, response_var, expl_var, final_hyPara_values, gam_finalPara, MLR_finalPara, LR_finalPara):
    
    # Progress bar
    st.info("Validation progress")
    my_bar = st.progress(0.0)
    progress1 = 0

    #-----------------------------------------------------------------------------------------
    # Continuous response variables

    if response_var_type == "continuous":

        # Model validation
        # R²
        model_eval_r2 = pd.DataFrame(index = range(val_runs), columns = algorithms)
        # MSE
        model_eval_mse = pd.DataFrame(index = range(val_runs), columns = algorithms)
        # RMSE
        model_eval_rmse = pd.DataFrame(index = range(val_runs), columns = algorithms)
        # MAE
        model_eval_mae = pd.DataFrame(index = range(val_runs), columns = algorithms)
        # MaxERR
        model_eval_maxerr = pd.DataFrame(index = range(val_runs), columns = algorithms)
        # EVRS
        model_eval_evrs = pd.DataFrame(index = range(val_runs), columns = algorithms)
        # SSR
        model_eval_ssr = pd.DataFrame(index = range(val_runs), columns = algorithms)

        # Variable importance
        model_varImp = {}
        for var in expl_var:
            model_varImp[var] = []
        model_varImp_mean = pd.DataFrame(index = expl_var, columns = algorithms)
        model_varImp_sd = pd.DataFrame(index = expl_var, columns = algorithms)
        
        # Model validation summary
        model_eval_mean = pd.DataFrame(index = ["% VE", "MSE", "RMSE", "MAE", "MaxErr", "EVRS", "SSR"], columns = algorithms)
        model_eval_sd = pd.DataFrame(index = ["% VE", "MSE", "RMSE", "MAE", "MaxErr", "EVRS", "SSR"], columns = algorithms)
        resdiuals_allmodels_allruns = pd.DataFrame()
        
        for model in algorithms:

            # Collect all residuals in test runs
            resdiuals_allruns = {}

            for val in range(val_runs):
                
                # Split data into train/ test data
                Y_data = data[response_var]
                X_data = data[expl_var]
                X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, train_size = train_frac, random_state = val)

                # Train MLR model
                if model == "Multiple Linear Regression":

                    # Extract parameters
                    MLR_intercept = MLR_finalPara["intercept"][0]
                    MLR_cov_type = MLR_finalPara["covType"][0]

                    if MLR_intercept == "Yes":
                        X_test_mlr = sm.add_constant(X_test)
                        X_train_mlr = sm.add_constant(X_train)
                    if MLR_intercept == "No":
                        X_test_mlr = X_test
                        X_train_mlr = X_train
                    
                    if MLR_model == "OLS":
                        model_mlr = sm.OLS(Y_train, X_train_mlr)
                    # if MLR_model == "GLS":
                    #     ols_resid = sm.OLS(Y_train, X_train_mlr).fit().resid
                    #     res_fit = sm.OLS(np.array(ols_resid[1:]), np.array(ols_resid[:-1])).fit()
                    #     rho = res_fit.params
                    #     order = toeplitz(np.arange(X_train_mlr.shape[0]))
                    #     sigma = rho**order
                    #     model_mlr = sm.GLS(Y_train, X_train_mlr, sigma=sigma)
                    
                    # Prediction for Y_test
                    model_mlr_fit = model_mlr.fit()
                    Y_test_pred = model_mlr_fit.predict(X_test_mlr)
                    Y_test_pred = Y_test_pred.to_numpy()
                    # Y_test_pred = ml_model.predict(X_test)

                    # sklearn
                    if MLR_intercept == "Yes":
                        ml_model = LinearRegression(fit_intercept=True)
                    if MLR_intercept == "No":
                        ml_model = LinearRegression(fit_intercept=False)
                    ml_model.fit(X_train, Y_train)      

                # Train GAM model
                if model == "Generalized Additive Models":
                    if isinstance(gam_finalPara["spline order"][0], list):
                        nos = gam_finalPara["number of splines"][0]
                        so = gam_finalPara["spline order"][0]
                        lam = gam_finalPara["lambda"][0]
                    else:
                        nos = int(gam_finalPara["number of splines"][0])
                        so = int(gam_finalPara["spline order"][0])
                        lam = float(gam_finalPara["lambda"][0])
                    if gam_finalPara["intercept"][0] == "Yes":
                        ml_model = LinearGAM(n_splines = nos, spline_order = so, lam = lam, fit_intercept = True).fit(X_train, Y_train)
                    if gam_finalPara["intercept"][0] == "No":
                        ml_model = LinearGAM(n_splines = nos, spline_order = so, lam = lam, fit_intercept = False).fit(X_train, Y_train)

                    # Prediction for Y_test
                    Y_test_pred = ml_model.predict(X_test) 

                # Train RF model
                if model == "Random Forest":
                    rf_final_para = final_hyPara_values["rf"]
                    params = {'n_estimators': rf_final_para["number of trees"][0],
                    'max_depth': rf_final_para["maximum tree depth"][0],
                    'max_features': rf_final_para["maximum number of features"][0],
                    'max_samples': rf_final_para["sample rate"][0],
                    'bootstrap': True,
                    'oob_score': True,
                    }
                    ml_model = ensemble.RandomForestRegressor(**params)
                    ml_model.fit(X_train, Y_train)
                    
                    # Prediction for Y_test
                    Y_test_pred = ml_model.predict(X_test)          
                    
                # Train BRT model
                if model == "Boosted Regression Trees":
                    brt_final_para = final_hyPara_values["brt"]
                    params = {'n_estimators': brt_final_para["number of trees"][0],
                    'learning_rate': brt_final_para["learning rate"][0],
                    'max_depth': brt_final_para["maximum tree depth"][0],
                    'subsample': brt_final_para["sample rate"][0]
                    }
                    ml_model = ensemble.GradientBoostingRegressor(**params)
                    ml_model.fit(X_train, Y_train.values.ravel())
                    
                    # Prediction for Y_test
                    Y_test_pred = ml_model.predict(X_test)

                # Train ANN model
                if model == "Artificial Neural Networks":
                    # Standardize X_data
                    scaler = StandardScaler()
                    scaler.fit(X_train)
                    X_train_ann = scaler.transform(X_train)
                    X_test_ann = scaler.transform(X_test) 

                    ann_final_para = final_hyPara_values["ann"]
                    params = {"solver": ann_final_para["weight optimization solver"][0],
                    "max_iter": ann_final_para["maximum number of iterations"][0],
                    "activation": ann_final_para["activation function"][0],
                    "hidden_layer_sizes": ann_final_para["hidden layer sizes"][0],
                    "learning_rate_init": ann_final_para["learning rate"][0],
                    #"learning_rate": ann_final_para["learning rate schedule"][0],
                    #"momentum": ann_final_para["momentum"][0],
                    "alpha": ann_final_para["L² regularization"][0]
                    #"epsilon": ann_final_para["epsilon"][0]
                    }
                    ml_model = MLPRegressor(**params)
                    ml_model.fit(X_train_ann, Y_train)
                    
                    # Prediction for Y_test
                    Y_test_pred = ml_model.predict(X_test_ann)

                # Variable importance with test data (via permutation, order important)
                scoring_function = make_scorer(r2_score, greater_is_better = True)
                if model == "Multiple Linear Regression":
                    varImp = permutation_importance(ml_model , X_test, Y_test, n_repeats = 10, random_state = 0, scoring = scoring_function)
                elif model == "Generalized Additive Models":
                    varImp = permutation_importance(ml_model , X_test, Y_test, n_repeats = 10, random_state = 0, scoring = scoring_function)
                elif model == "Random Forest":
                    varImp = permutation_importance(ml_model , X_test, Y_test, n_repeats = 10, random_state = 0, scoring = scoring_function)
                elif model == "Boosted Regression Trees":
                    varImp = permutation_importance(ml_model , X_test, Y_test.values.ravel(), n_repeats = 10, random_state = 0, scoring = scoring_function)
                elif model == "Artificial Neural Networks":
                    varImp = permutation_importance(ml_model , X_test_ann, Y_test, n_repeats = 10, random_state = 0, scoring = scoring_function)
                for var in expl_var:
                    model_varImp[var] = np.append(model_varImp[var], varImp.importances[expl_var.index(var)])

                # Save R² for test data
                model_eval_r2.iloc[val][model] = r2_score(Y_test, Y_test_pred)

                # Save MSE for test data
                model_eval_mse.iloc[val][model] = mean_squared_error(Y_test, Y_test_pred, squared = True)

                # Save RMSE for test data
                model_eval_rmse.iloc[val][model] = mean_squared_error(Y_test, Y_test_pred, squared = False)

                # Save MAE for test data
                model_eval_mae.iloc[val][model] = mean_absolute_error(Y_test, Y_test_pred)

                # Save MaxERR for test data
                model_eval_maxerr.iloc[val][model] = max_error(Y_test, Y_test_pred)

                # Save explained variance regression score for test data
                model_eval_evrs.iloc[val][model] = explained_variance_score(Y_test, Y_test_pred)

                # Save sum of squared residuals for test data
                model_eval_ssr.iloc[val][model] = ((Y_test-Y_test_pred)**2).sum()

                # Save residual values for test data 
                res = Y_test-Y_test_pred
                resdiuals_allruns[val] = res

                progress1 += 1
                my_bar.progress(progress1/(len(algorithms)*val_runs))
            
            # Calculate mean performance statistics
            # Mean
            model_eval_mean.loc["% VE"][model] = model_eval_r2[model].mean()
            model_eval_mean.loc["MSE"][model] = model_eval_mse[model].mean()
            model_eval_mean.loc["RMSE"][model] = model_eval_rmse[model].mean()
            model_eval_mean.loc["MAE"][model] = model_eval_mae[model].mean()
            model_eval_mean.loc["MaxErr"][model] = model_eval_maxerr[model].mean()
            model_eval_mean.loc["EVRS"][model] = model_eval_evrs[model].mean()
            model_eval_mean.loc["SSR"][model] = model_eval_ssr[model].mean()
            # Sd
            model_eval_sd.loc["% VE"][model] = model_eval_r2[model].std()
            model_eval_sd.loc["MSE"][model] = model_eval_mse[model].std()
            model_eval_sd.loc["RMSE"][model] = model_eval_rmse[model].std()
            model_eval_sd.loc["MAE"][model] = model_eval_mae[model].std()
            model_eval_sd.loc["MaxErr"][model] = model_eval_maxerr[model].std()
            model_eval_sd.loc["EVRS"][model] = model_eval_evrs[model].std()
            model_eval_sd.loc["SSR"][model] = model_eval_ssr[model].std()
            # Residuals 
            residuals_collection = pd.DataFrame(columns =[response_var])
            for x in resdiuals_allruns: 
                #residuals_collection = residuals_collection.append(pd.DataFrame(resdiuals_allruns[x]), ignore_index = True)
                residuals_collection =pd.concat([residuals_collection,pd.DataFrame(resdiuals_allruns[x])])
            resdiuals_allmodels_allruns[model] = residuals_collection[response_var]
            # Variable importances (mean & sd)
            for v in expl_var:
                model_varImp_mean.loc[v][model] = model_varImp[v].mean()
                model_varImp_sd.loc[v][model] = model_varImp[v].std()
            
        # Collect results
        validation_results = {}
        validation_results["mean"] = model_eval_mean
        validation_results["sd"] = model_eval_sd
        validation_results["residuals"] = resdiuals_allmodels_allruns
        validation_results["variance explained"] = model_eval_r2
        validation_results["variable importance mean"] = model_varImp_mean
        validation_results["variable importance sd"] = model_varImp_sd

    #-----------------------------------------------------------------------------------------
    # Binary response variables

    if response_var_type == "binary":

        validation_results = {}

        # Exclude MLR
        if any(a for a in algorithms if a == "Multiple Linear Regression"):
            algo = []
            for i in algorithms:
                if i != "Multiple Linear Regression":
                    algo.append(i)
            algorithms = algo

        if algorithms is not None:
        
            # Model validation (threshold independent)
            # AUC ROC
            model_eval_auc_roc = pd.DataFrame(index = range(val_runs), columns = algorithms)
            # Average precision
            model_eval_ap = pd.DataFrame(index = range(val_runs), columns = algorithms)
            # AUC PRC
            model_eval_auc_prc = pd.DataFrame(index = range(val_runs), columns = algorithms)
            # Log-loss
            model_eval_log_loss = pd.DataFrame(index = range(val_runs), columns = algorithms)
            # Threshold
            model_eval_thres = pd.DataFrame(index = range(val_runs), columns = algorithms)
            # Model validation (threshold dependent)
            # TPR
            model_eval_TPR = pd.DataFrame(index = range(val_runs), columns = algorithms)
            # FNR
            model_eval_FNR = pd.DataFrame(index = range(val_runs), columns = algorithms)
            # TNR
            model_eval_TNR = pd.DataFrame(index = range(val_runs), columns = algorithms)
            # FPR
            model_eval_FPR = pd.DataFrame(index = range(val_runs), columns = algorithms)
            # TSS
            model_eval_TSS = pd.DataFrame(index = range(val_runs), columns = algorithms)
            # PREC
            model_eval_PREC = pd.DataFrame(index = range(val_runs), columns = algorithms)
            # F1
            model_eval_F1 = pd.DataFrame(index = range(val_runs), columns = algorithms)
            # KAPPA
            model_eval_kappa = pd.DataFrame(index = range(val_runs), columns = algorithms)
            # ACC
            model_eval_acc = pd.DataFrame(index = range(val_runs), columns = algorithms)
            # BAL ACC
            model_eval_bal_acc = pd.DataFrame(index = range(val_runs), columns = algorithms)

            # Variable importance
            model_varImp = {}
            for var in expl_var:
                model_varImp[var] = []
            model_varImp_mean = pd.DataFrame(index = expl_var, columns = algorithms)
            model_varImp_sd = pd.DataFrame(index = expl_var, columns = algorithms)

            # Model validation summary
            model_eval_mean_ind = pd.DataFrame(index = ["AUC ROC", "AP", "AUC PRC", "LOG-LOSS"], columns = algorithms)
            model_eval_sd_ind = pd.DataFrame(index = ["AUC ROC", "AP", "AUC PRC", "LOG-LOSS"], columns = algorithms)
            model_eval_mean_thres = pd.DataFrame(index = ["threshold"], columns = algorithms)
            model_eval_sd_thres = pd.DataFrame(index = ["threshold"], columns = algorithms)
            model_eval_mean_dep = pd.DataFrame(index = ["TPR", "FNR", "TNR", "FPR", "TSS", "PREC", "F1", "KAPPA", "ACC", "BAL ACC"], columns = algorithms)
            model_eval_sd_dep = pd.DataFrame(index = ["TPR", "FNR", "TNR", "FPR", "TSS", "PREC", "F1", "KAPPA", "ACC", "BAL ACC"], columns = algorithms)
            
            for model in algorithms:
                for val in range(val_runs):

                    # Split data into train/ test data
                    Y_data = data[response_var]
                    X_data = data[expl_var]
                    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, train_size = train_frac, random_state = val)#, stratify = Y_data)

                    # Train LR model
                    if model == "Logistic Regression":

                        # Extract parameters
                        LR_intercept = LR_finalPara["intercept"][0]
                        LR_cov_type = LR_finalPara["covType"][0]

                        # Train LR model (statsmodels)
                        if LR_intercept == "Yes":
                            X_train_lr = sm.add_constant(X_train)
                        if LR_intercept == "No":
                            X_train_lr = X_train
                        ml_model = sm.Logit(Y_train, X_train_lr)
                        ml_model_fit = ml_model.fit(method = "ncg", maxiter = 100)

                        # Prediction probability for Y_test
                        if LR_intercept == "Yes":
                            X_test_lr = sm.add_constant(X_test)
                        if LR_intercept == "No":
                            X_test_lr = X_test
                        Y_test_pred = 1-pd.DataFrame(ml_model_fit.predict(X_test_lr), columns = ["0"])
                        Y_test_pred["1"] = 1-Y_test_pred
                        Y_test_pred = Y_test_pred.to_numpy()

                        # Train LR model (sklearn)
                        if LR_intercept == "Yes":
                            ml_model_sk = LogisticRegression(fit_intercept = True, solver = "newton-cg", penalty = "none", tol = 1e-05)
                        if LR_intercept == "No":
                            ml_model_sk = LogisticRegression(fit_intercept = False, solver = "newton-cg", penalty = "none", tol = 1e-05)
                        ml_model_sk.fit(X_train, Y_train)

                    # Train GAM model
                    if model == "Generalized Additive Models":
                        if isinstance(gam_finalPara["spline order"][0], list):
                            nos = gam_finalPara["number of splines"][0]
                            so = gam_finalPara["spline order"][0]
                            lam = gam_finalPara["lambda"][0]
                        else:
                            nos = int(gam_finalPara["number of splines"][0])
                            so = int(gam_finalPara["spline order"][0])
                            lam = float(gam_finalPara["lambda"][0])
                        if gam_finalPara["intercept"][0] == "Yes":
                            ml_model = LogisticGAM(n_splines = nos, spline_order = so, lam = lam, fit_intercept = True).fit(X_train, Y_train)
                        if gam_finalPara["intercept"][0] == "No":
                            ml_model = LogisticGAM(n_splines = nos, spline_order = so, lam = lam, fit_intercept = False).fit(X_train, Y_train)

                        # Prediction for Y_test
                        Y_test_pred = 1-pd.DataFrame(ml_model.predict_proba(X_test), columns = ["0"])
                        Y_test_pred["1"] = 1-Y_test_pred
                        Y_test_pred = Y_test_pred.to_numpy() 

                    # Train RF model
                    if model == "Random Forest":
                        rf_final_para = final_hyPara_values["rf"]
                        params = {'n_estimators': rf_final_para["number of trees"][0],
                        'max_depth': rf_final_para["maximum tree depth"][0],
                        'max_features': rf_final_para["maximum number of features"][0],
                        'max_samples': rf_final_para["sample rate"][0],
                        'bootstrap': True,
                        'oob_score': True,
                        }
                        ml_model = ensemble.RandomForestClassifier(**params)
                        ml_model.fit(X_train, Y_train)
                        
                        # Prediction for Y_test
                        Y_test_pred = ml_model.predict_proba(X_test)         

                    # Train BRT model
                    if model == "Boosted Regression Trees":
                        brt_final_para = final_hyPara_values["brt"]
                        params = {'n_estimators': brt_final_para["number of trees"][0],
                        'learning_rate': brt_final_para["learning rate"][0],
                        'max_depth': brt_final_para["maximum tree depth"][0],
                        'subsample': brt_final_para["sample rate"][0]
                        }
                        ml_model = ensemble.GradientBoostingClassifier(**params)
                        ml_model.fit(X_train, Y_train.values.ravel())

                        # Prediction probability for Y_test
                        Y_test_pred = ml_model.predict_proba(X_test)

                    # Train ANN model
                    if model == "Artificial Neural Networks":
                        # Standardize X_data
                        scaler = StandardScaler()
                        scaler.fit(X_train)
                        X_train_ann = scaler.transform(X_train)
                        X_test_ann = scaler.transform(X_test) 

                        ann_final_para = final_hyPara_values["ann"]
                        params = {"solver": ann_final_para["weight optimization solver"][0],
                        "max_iter": ann_final_para["maximum number of iterations"][0],
                        "activation": ann_final_para["activation function"][0],
                        "hidden_layer_sizes": ann_final_para["hidden layer sizes"][0],
                        "learning_rate_init": ann_final_para["learning rate"][0],
                        #"learning_rate": ann_final_para["learning rate schedule"][0],
                        #"momentum": ann_final_para["momentum"][0],
                        "alpha": ann_final_para["L² regularization"][0]
                        #"epsilon": ann_final_para["epsilon"][0]
                        }
                        ml_model = MLPClassifier(**params)
                        ml_model.fit(X_train_ann, Y_train)

                        # Prediction probability for Y_test
                        Y_test_pred = ml_model.predict_proba(X_test_ann)

                    # Variable importance with test data (via permutation, order important)
                    scoring_function = make_scorer(roc_auc_score, greater_is_better = True)
                    if model == "Logistic Regression":
                        varImp = permutation_importance(ml_model_sk , X_test, Y_test, n_repeats = 10, random_state = 0, scoring = scoring_function)
                    elif model == "Generalized Additive Models":
                        varImp = permutation_importance(ml_model , X_test, Y_test, n_repeats = 10, random_state = 0, scoring = scoring_function)
                    elif model == "Random Forest":
                        varImp = permutation_importance(ml_model , X_test, Y_test, n_repeats = 10, random_state = 0, scoring = scoring_function)
                    elif model == "Boosted Regression Trees":
                        varImp = permutation_importance(ml_model , X_test, Y_test.values.ravel(), n_repeats = 10, random_state = 0, scoring = scoring_function)
                    elif model == "Artificial Neural Networks":
                        varImp = permutation_importance(ml_model , X_test_ann, Y_test, n_repeats = 10, random_state = 0, scoring = scoring_function)
                    for var in expl_var:
                        model_varImp[var] = np.append(model_varImp[var], varImp.importances[expl_var.index(var)])
                    
                    # Threshold independent metrics
                    # Save AUC ROC for test data
                    model_eval_auc_roc.iloc[val][model] = roc_auc_score(Y_test, Y_test_pred[:, 1])

                    # Save AP for test data
                    model_eval_ap.iloc[val][model] = average_precision_score(Y_test, Y_test_pred[:, 1])

                    # Save AUC PRC for test data
                    precision, recall, thresholds = precision_recall_curve(Y_test, Y_test_pred[:, 1])
                    model_eval_auc_prc.iloc[val][model] = auc(recall, precision)

                    # Save log-loss for test data
                    model_eval_log_loss.iloc[val][model] = log_loss(Y_test, Y_test_pred)

                    #------------------------------

                    # Threshold determination (according to Youden index)
                    FPR, TPR, thresholds = roc_curve(Y_test, Y_test_pred[:, 1])
                    thres_index = np.argmax(TPR - FPR)
                    thres = thresholds[thres_index]

                    # Threshold determination (minimizing abs. distance between SENS & SPEC)
                    # FPR, TPR, thresholds = roc_curve(Y_test, Y_test_pred[:, 1])
                    # thres_index = np.argmin(abs(TPR - (1-FPR)))
                    # thres = thresholds[thres_index]

                    model_eval_thres.iloc[val][model] = thres

                    # Create 0/1 prediction according to threshold
                    Y_test_pred_bin = np.array([1 if x >= thres else 0 for x in Y_test_pred[:, 1]])

                    #------------------------------

                    # Threshold dependent metrics
                    # Confusion matrix
                    TN, FP, FN, TP = confusion_matrix(Y_test, Y_test_pred_bin).ravel()

                    # Save TPR (sensitivity, recall) for test data
                    model_eval_TPR.iloc[val][model] = TP/(TP+FN)

                    # Save FNR for test data
                    model_eval_FNR.iloc[val][model] = FN/(TP+FN)
                    
                    # Save TNR (specificity) for test data
                    model_eval_TNR.iloc[val][model] = TN/(TN+FP)
                    
                    # Save TPR for test data
                    model_eval_FPR.iloc[val][model] = FP/(TN+FP)

                    # Save TSS for test data
                    model_eval_TSS.iloc[val][model] = (TP/(TP+FN))+(TN/(TN+FP))-1

                    # Save PREC for test data
                    model_eval_PREC.iloc[val][model] = TP/(TP+FP)
                    
                    # Save F1 for test data
                    model_eval_F1.iloc[val][model] = f1_score(Y_test, Y_test_pred_bin)

                    # Save Cohen's Kappa for test data
                    model_eval_kappa.iloc[val][model] = cohen_kappa_score(Y_test, Y_test_pred_bin)

                    # Save accuracy for test data
                    model_eval_acc.iloc[val][model] = accuracy_score(Y_test, Y_test_pred_bin)

                    # Save balanced accuracy for test data
                    model_eval_bal_acc.iloc[val][model] = balanced_accuracy_score(Y_test, Y_test_pred_bin)

                    progress1 += 1
                    my_bar.progress(progress1/(len(algorithms)*val_runs))

                # Calculate mean performance statistics
                # mean
                model_eval_mean_ind.loc["AUC ROC"][model] = model_eval_auc_roc[model].mean()
                model_eval_mean_ind.loc["AP"][model] = model_eval_ap[model].mean()
                model_eval_mean_ind.loc["AUC PRC"][model] = model_eval_auc_prc[model].mean()
                model_eval_mean_ind.loc["LOG-LOSS"][model] = model_eval_log_loss[model].mean()
                model_eval_mean_thres.loc["threshold"][model] = model_eval_thres[model].mean()
                model_eval_mean_dep.loc["TPR"][model] = model_eval_TPR[model].mean()
                model_eval_mean_dep.loc["FNR"][model] = model_eval_FNR[model].mean()
                model_eval_mean_dep.loc["TNR"][model] = model_eval_TNR[model].mean()
                model_eval_mean_dep.loc["FPR"][model] = model_eval_FPR[model].mean()
                model_eval_mean_dep.loc["TSS"][model] = model_eval_TSS[model].mean()
                model_eval_mean_dep.loc["PREC"][model] = model_eval_PREC[model].mean()
                model_eval_mean_dep.loc["F1"][model] = model_eval_F1[model].mean()
                model_eval_mean_dep.loc["KAPPA"][model] = model_eval_kappa[model].mean()
                model_eval_mean_dep.loc["ACC"][model] = model_eval_acc[model].mean()
                model_eval_mean_dep.loc["BAL ACC"][model] = model_eval_bal_acc[model].mean()
                # sd
                model_eval_sd_ind.loc["AUC ROC"][model] = model_eval_auc_roc[model].std()
                model_eval_sd_ind.loc["AP"][model] = model_eval_ap[model].std()
                model_eval_sd_ind.loc["AUC PRC"][model] = model_eval_auc_prc[model].std()
                model_eval_sd_ind.loc["LOG-LOSS"][model] = model_eval_log_loss[model].std()
                model_eval_sd_thres.loc["threshold"][model] = model_eval_thres[model].std()
                model_eval_sd_dep.loc["TPR"][model] = model_eval_TPR[model].std()
                model_eval_sd_dep.loc["FNR"][model] = model_eval_FNR[model].std()
                model_eval_sd_dep.loc["TNR"][model] = model_eval_TNR[model].std()
                model_eval_sd_dep.loc["TSS"][model] = model_eval_TSS[model].std()
                model_eval_sd_dep.loc["FPR"][model] = model_eval_FPR[model].std()
                model_eval_sd_dep.loc["PREC"][model] = model_eval_PREC[model].std()
                model_eval_sd_dep.loc["F1"][model] = model_eval_F1[model].std()
                model_eval_sd_dep.loc["KAPPA"][model] = model_eval_kappa[model].std()
                model_eval_sd_dep.loc["ACC"][model] = model_eval_acc[model].std()
                model_eval_sd_dep.loc["BAL ACC"][model] = model_eval_bal_acc[model].std()
                # Variable importances (mean & sd)
                for v in expl_var:
                    model_varImp_mean.loc[v][model] = model_varImp[v].mean()
                    model_varImp_sd.loc[v][model] = model_varImp[v].std()

            # Collect results
            validation_results["mean_ind"] = model_eval_mean_ind
            validation_results["mean_thres"] = model_eval_mean_thres
            validation_results["mean_dep"] = model_eval_mean_dep
            validation_results["sd_ind"] = model_eval_sd_ind
            validation_results["sd_thres"] = model_eval_sd_thres
            validation_results["sd_dep"] = model_eval_sd_dep
            validation_results["AUC ROC"] = model_eval_auc_roc
            validation_results["TSS"] = model_eval_TSS
            validation_results["variable importance mean"] = model_varImp_mean
            validation_results["variable importance sd"] = model_varImp_sd
        else:
            validation_results["mean_ind"] = pd.DataFrame(columns = algorithms)
            validation_results["mean_thres"] = pd.DataFrame(columns = algorithms)
            validation_results["mean_dep"] = pd.DataFrame(columns = algorithms)
            validation_results["sd_ind"] = pd.DataFrame(columns = algorithms)
            validation_results["sd_thres"] = pd.DataFrame(columns = algorithms)
            validation_results["sd_dep"] = pd.DataFrame(columns = algorithms)
            validation_results["AUC ROC"] = pd.DataFrame(columns = algorithms)
            validation_results["TSS"] = pd.DataFrame(columns = algorithms)
            validation_results["variable importance mean"] = pd.DataFrame(columns = algorithms)
            validation_results["variable importance sd"] = pd.DataFrame(columns = algorithms)
    
    #-----------------------------------------------------------------------------------------
    # Multi-class response variables

    if response_var_type == "multi-class":

        validation_results = {}

        if algorithms is not None:
        
            # Model validation
            # ACC
            model_eval_acc = pd.DataFrame(index = range(val_runs), columns = algorithms)
            # BAL ACC
            model_eval_bal_acc = pd.DataFrame(index = range(val_runs), columns = algorithms)

            # Variable importance
            model_varImp = {}
            for var in expl_var:
                model_varImp[var] = []
            model_varImp_mean = pd.DataFrame(index = expl_var, columns = algorithms)
            model_varImp_sd = pd.DataFrame(index = expl_var, columns = algorithms)

            # Model validation summary
            model_eval_mean = pd.DataFrame(index = ["ACC", "BAL ACC"], columns = algorithms)
            model_eval_sd = pd.DataFrame(index = ["ACC", "BAL ACC"], columns = algorithms)
            
            for model in algorithms:
                for val in range(val_runs):

                    # Split data into train/ test data
                    Y_data = data[response_var]
                    X_data = data[expl_var]
                    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, train_size = train_frac, random_state = val)#, stratify = Y_data)

                    # Train RF model
                    if model == "Random Forest":
                        rf_final_para = final_hyPara_values["rf"]
                        params = {'n_estimators': rf_final_para["number of trees"][0],
                        'max_depth': rf_final_para["maximum tree depth"][0],
                        'max_features': rf_final_para["maximum number of features"][0],
                        'max_samples': rf_final_para["sample rate"][0],
                        'bootstrap': True,
                        'oob_score': True,
                        }
                        ml_model = ensemble.RandomForestClassifier(**params)
                        ml_model.fit(X_train, Y_train)
                        
                        # Prediction for Y_test
                        Y_test_pred = ml_model.predict(X_test)         

                    # Train ANN model
                    if model == "Artificial Neural Networks":
                        # Standardize X_data
                        scaler = StandardScaler()
                        scaler.fit(X_train)
                        X_train_ann = scaler.transform(X_train)
                        X_test_ann = scaler.transform(X_test) 

                        ann_final_para = final_hyPara_values["ann"]
                        params = {"solver": ann_final_para["weight optimization solver"][0],
                        "max_iter": ann_final_para["maximum number of iterations"][0],
                        "activation": ann_final_para["activation function"][0],
                        "hidden_layer_sizes": ann_final_para["hidden layer sizes"][0],
                        "learning_rate_init": ann_final_para["learning rate"][0],
                        #"learning_rate": ann_final_para["learning rate schedule"][0],
                        #"momentum": ann_final_para["momentum"][0],
                        "alpha": ann_final_para["L² regularization"][0]
                        #"epsilon": ann_final_para["epsilon"][0]
                        }
                        ml_model = MLPClassifier(**params)
                        ml_model.fit(X_train_ann, Y_train)

                        # Prediction probability for Y_test
                        Y_test_pred = ml_model.predict(X_test_ann)

                    # Variable importance with test data (via permutation, order important)
                    scoring_function = make_scorer(accuracy_score, greater_is_better = True)
                    # if model == "Logistic Regression":
                    #     varImp = permutation_importance(ml_model_sk , X_test, Y_test, n_repeats = 10, random_state = 0, scoring = scoring_function)
                    if model == "Random Forest":
                        varImp = permutation_importance(ml_model , X_test, Y_test, n_repeats = 10, random_state = 0, scoring = scoring_function)
                    elif model == "Artificial Neural Networks":
                        varImp = permutation_importance(ml_model , X_test_ann, Y_test, n_repeats = 10, random_state = 0, scoring = scoring_function)
                    for var in expl_var:
                        model_varImp[var] = np.append(model_varImp[var], varImp.importances[expl_var.index(var)])

                    # Save accuracy for test data
                    model_eval_acc.iloc[val][model] = accuracy_score(Y_test, Y_test_pred)

                    # Save balanced accuracy for test data
                    model_eval_bal_acc.iloc[val][model] = balanced_accuracy_score(Y_test, Y_test_pred)

                    progress1 += 1
                    my_bar.progress(progress1/(len(algorithms)*val_runs))

                # Calculate mean performance statistics
                # mean
                model_eval_mean.loc["ACC"][model] = model_eval_acc[model].mean()
                model_eval_mean.loc["BAL ACC"][model] = model_eval_bal_acc[model].mean()
                # sd
                model_eval_sd.loc["ACC"][model] = model_eval_acc[model].std()
                model_eval_sd.loc["BAL ACC"][model] = model_eval_bal_acc[model].std()
                # Variable importances (mean & sd)
                for v in expl_var:
                    model_varImp_mean.loc[v][model] = model_varImp[v].mean()
                    model_varImp_sd.loc[v][model] = model_varImp[v].std()

            # Collect results
            validation_results["mean"] = model_eval_mean
            validation_results["sd"] = model_eval_sd
            validation_results["ACC"] = model_eval_acc
            validation_results["BAL ACC"] = model_eval_bal_acc
            validation_results["variable importance mean"] = model_varImp_mean
            validation_results["variable importance sd"] = model_varImp_sd
        else:
            validation_results["mean"] = pd.DataFrame(columns = algorithms)
            validation_results["sd"] = pd.DataFrame(columns = algorithms)
            validation_results["ACC"] = pd.DataFrame(columns = algorithms)
            validation_results["BAL ACC"] = pd.DataFrame(columns = algorithms)
            validation_results["variable importance mean"] = pd.DataFrame(columns = algorithms)
            validation_results["variable importance sd"] = pd.DataFrame(columns = algorithms)
            
    return validation_results


#----------------------------------------------------------------------------------------------
#FUNCTION FOR FULL MODEL
#@st.cache(suppress_st_warning = True, allow_output_mutation = True)
def model_full(data, data_new, algorithms, MLR_model, MLR_finalPara, LR_finalPara, response_var_type, response_var, expl_var, final_hyPara_values, gam_finalPara):
    
    # Progress bar
    st.info("Full model progress")
    my_bar = st.progress(0.0)
    progress2 = 0

    # Save results 
    full_model_results = {}

    # Prepare data
    Y_data = data[response_var]
    X_data = data[expl_var]
    if data_new.empty == False:
        X_data_new = data_new[expl_var]

    #-----------------------------------------------------------------------------------------
    # Continuous response variables

    if response_var_type == "continuous":

        # Collect performance results across all algorithms
        model_comparison = pd.DataFrame(index = ["% VE", "MSE", "RMSE", "MAE", "MaxErr", "EVRS", "SSR"], columns = algorithms)
        residuals_comparison = pd.DataFrame(columns = algorithms)

        # Multiple Linear Regression full model
        if any(a for a in algorithms if a == "Multiple Linear Regression"):

            # Extract parameters
            MLR_intercept = MLR_finalPara["intercept"][0]
            MLR_cov_type = MLR_finalPara["covType"][0]

            # Save results
            mlr_reg_inf = pd.DataFrame(index = ["Dep. variable", "Model", "Method", "No. observations", "DF residuals", "DF model", "Covariance type"], columns = ["Value"])
            mlr_reg_stats = pd.DataFrame(index = ["R²", "Adj. R²", "Mult. corr. coeff.", "Residual SE", "Log-likelihood", "AIC", "BIC"], columns = ["Value"])
            mlr_reg_anova = pd.DataFrame(index = ["Regression", "Residual", "Total"], columns = ["DF", "SS", "MS", "F-statistic", "p-value"])
            if MLR_intercept == "Yes":
                mlr_reg_coef = pd.DataFrame(index = ["const"]+ expl_var, columns = ["coeff", "std err", "t-statistic", "p-value", "lower 95%", "upper 95%"])
            if MLR_intercept == "No":
                mlr_reg_coef = pd.DataFrame(index = expl_var, columns = ["coeff", "std err", "t-statistic", "p-value", "lower 95%", "upper 95%"])
            mlr_reg_hetTest = pd.DataFrame(index = ["test statistic", "p-value"], columns = ["Breusch-Pagan test", "White test (without int.)", "White test (with int.)"])
            mlr_reg_varImp = pd.DataFrame(index = expl_var, columns = ["mean", "std"])

            # Train MLR model (statsmodels)
            if MLR_intercept == "Yes":
                X_data_mlr = sm.add_constant(X_data)
            if MLR_intercept == "No":
                X_data_mlr = X_data
            if MLR_model == "OLS":
                full_model_mlr = sm.OLS(Y_data, X_data_mlr)
            # if MLR_model == "GLS":
            #     ols_resid = sm.OLS(Y_data, X_data_mlr).fit().resid
            #     res_fit = sm.OLS(np.array(ols_resid[1:]), np.array(ols_resid[:-1])).fit()
            #     rho = res_fit.params
            #     order = toeplitz(np.arange(data.shape[0]))
            #     sigma = rho**order
            #     full_model_mlr = sm.GLS(Y_data, X_data_mlr, sigma=sigma)
            if MLR_cov_type == "non-robust":
                full_model_fit = full_model_mlr.fit()
            else:
                full_model_fit = full_model_mlr.fit().get_robustcov_results(cov_type = MLR_cov_type)
            Y_pred = full_model_fit.predict(X_data_mlr)
            Y_pred = Y_pred.to_numpy()
            if data_new.empty == False:
                if MLR_intercept == "Yes":
                    X_data_new_mlr = sm.add_constant(X_data_new)
                if MLR_intercept == "No":
                    X_data_new_mlr = X_data_new
                Y_pred_new = full_model_fit.predict(X_data_new_mlr)
                Y_pred_new = Y_pred_new.to_numpy()

            # Train MLR model (sklearn)
            if MLR_intercept == "Yes":
                full_model_mlr_sk = LinearRegression(fit_intercept=True)
            if MLR_intercept == "No":
                full_model_mlr_sk = LinearRegression(fit_intercept=False)    
            full_model_mlr_sk.fit(X_data, Y_data)
            # Y_pred = full_model_mlr_sk.predict(X_data)

            # Extract essential results from model
            # Information
            mlr_reg_inf.loc["Dep. variable"] = full_model_fit.model.endog_names
            mlr_reg_inf.loc["Model"] = MLR_model
            mlr_reg_inf.loc["Method"] = "Least squares"
            mlr_reg_inf.loc["No. observations"] = full_model_fit.model.nobs
            mlr_reg_inf.loc["DF residuals"] = full_model_fit.df_resid
            mlr_reg_inf.loc["DF model"] = full_model_fit.df_model
            mlr_reg_inf.loc["Covariance type"] = full_model_fit.cov_type
            # Statistics
            mlr_reg_stats.loc["R²"] = full_model_fit.rsquared
            mlr_reg_stats.loc["Adj. R²"] = full_model_fit.rsquared_adj
            mlr_reg_stats.loc["Mult. corr. coeff."] = np.sqrt(full_model_fit.rsquared)
            mlr_reg_stats.loc["Residual SE"] = np.sqrt(full_model_fit.mse_resid)
            mlr_reg_stats.loc["Log-likelihood"] = full_model_fit.llf
            mlr_reg_stats.loc["AIC"] = full_model_fit.aic
            mlr_reg_stats.loc["BIC"] = full_model_fit.bic
            # ANOVA
            mlr_reg_anova.loc["Regression"]["DF"] = full_model_fit.df_model
            mlr_reg_anova.loc["Regression"]["SS"] = full_model_fit.ess
            mlr_reg_anova.loc["Regression"]["MS"] = full_model_fit.ess/full_model_fit.df_model
            mlr_reg_anova.loc["Regression"]["F-statistic"] = full_model_fit.fvalue
            mlr_reg_anova.loc["Regression"]["p-value"] = full_model_fit.f_pvalue
            mlr_reg_anova.loc["Residual"]["DF"] = full_model_fit.df_resid
            mlr_reg_anova.loc["Residual"]["SS"] = full_model_fit.ssr
            mlr_reg_anova.loc["Residual"]["MS"] = full_model_fit.ssr/full_model_fit.df_resid
            mlr_reg_anova.loc["Residual"]["F-statistic"] = ""
            mlr_reg_anova.loc["Residual"]["p-value"] = ""
            mlr_reg_anova.loc["Total"]["DF"] = full_model_fit.df_resid + full_model_fit.df_model
            mlr_reg_anova.loc["Total"]["SS"] = full_model_fit.ssr + full_model_fit.ess
            mlr_reg_anova.loc["Total"]["MS"] = ""
            mlr_reg_anova.loc["Total"]["F-statistic"] = ""
            mlr_reg_anova.loc["Total"]["p-value"] = ""
            # Coefficients
            if MLR_intercept == "Yes":
                coef_list = ["const"]+ expl_var
            if MLR_intercept == "No":
                coef_list = expl_var
            for c in coef_list:
                mlr_reg_coef.loc[c]["coeff"] = full_model_fit.params[coef_list.index(c)]
                mlr_reg_coef.loc[c]["std err"] = full_model_fit.bse[coef_list.index(c)]
                mlr_reg_coef.loc[c]["t-statistic"] = full_model_fit.tvalues[coef_list.index(c)]
                mlr_reg_coef.loc[c]["p-value"] = full_model_fit.pvalues[coef_list.index(c)]
                if MLR_cov_type == "non-robust":
                    mlr_reg_coef.loc[c]["lower 95%"] = full_model_fit.conf_int(alpha = 0.05).loc[c][0]
                    mlr_reg_coef.loc[c]["upper 95%"] = full_model_fit.conf_int(alpha = 0.05).loc[c][1]
                else:
                    mlr_reg_coef.loc[c]["lower 95%"] = full_model_fit.conf_int(alpha = 0.05)[coef_list.index(c)][0]
                    mlr_reg_coef.loc[c]["upper 95%"] = full_model_fit.conf_int(alpha = 0.05)[coef_list.index(c)][1]
            if MLR_intercept == "Yes":
                # Breusch-Pagan heteroscedasticity test
                bp_result = sm.stats.diagnostic.het_breuschpagan(full_model_fit.resid, full_model_fit.model.exog) 
                mlr_reg_hetTest.loc["test statistic"]["Breusch-Pagan test"] = bp_result[0]
                mlr_reg_hetTest.loc["p-value"]["Breusch-Pagan test"] = bp_result[1]
                # White heteroscedasticity test with interaction
                white_int_result = sm.stats.diagnostic.het_white(full_model_fit.resid, full_model_fit.model.exog)
                mlr_reg_hetTest.loc["test statistic"]["White test (with int.)"] = white_int_result[0]
                mlr_reg_hetTest.loc["p-value"]["White test (with int.)"] = white_int_result[1]
                # White heteroscedasticity test without interaction
                X_data_mlr_white = X_data_mlr
                for i in expl_var: 
                    X_data_mlr_white[i+ "_squared"] = X_data_mlr_white[i]**2
                white = sm.OLS(full_model_fit.resid**2, X_data_mlr_white)
                del X_data_mlr_white
                white_fit = white.fit()
                white_statistic = white_fit.rsquared*data.shape[0]
                white_p_value = stats.chi2.sf(white_statistic,len(white_fit.model.exog_names)-1)
                mlr_reg_hetTest.loc["test statistic"]["White test (without int.)"] = white_statistic
                mlr_reg_hetTest.loc["p-value"]["White test (without int.)"] = white_p_value

            # Variable importance (via permutation, order important)
            scoring_function = make_scorer(r2_score, greater_is_better = True)
            mlr_varImp = permutation_importance(full_model_mlr_sk , X_data, Y_data, n_repeats = 10, random_state = 0, scoring = scoring_function)
            for varI in expl_var:
                mlr_reg_varImp.loc[varI]["mean"] = mlr_varImp.importances_mean[expl_var.index(varI)]
                mlr_reg_varImp.loc[varI]["std"] = mlr_varImp.importances_std[expl_var.index(varI)]
            mlr_reg_varImp = mlr_reg_varImp.sort_values(by=["mean"], ascending = False)

            # Save tables
            full_model_results["MLR information"] = mlr_reg_inf
            full_model_results["MLR statistics"] = mlr_reg_stats
            full_model_results["MLR ANOVA"] = mlr_reg_anova
            full_model_results["MLR coefficients"] = mlr_reg_coef
            full_model_results["MLR hetTest"] = mlr_reg_hetTest
            full_model_results["MLR fitted"] = Y_pred
            full_model_results["MLR Cooks distance"] = full_model_fit.get_influence().cooks_distance[0]
            full_model_results["MLR leverage"] = full_model_fit.get_influence().hat_matrix_diag
            full_model_results["MLR variable importance"] = mlr_reg_varImp 
            if data_new.empty == False:
                full_model_results["MLR prediction"] = Y_pred_new

            # Model comparison for MLR
            model_comparison.loc["% VE"]["Multiple Linear Regression"] =  r2_score(Y_data, Y_pred)
            model_comparison.loc["MSE"]["Multiple Linear Regression"] = mean_squared_error(Y_data, Y_pred, squared = True)
            model_comparison.loc["RMSE"]["Multiple Linear Regression"] = mean_squared_error(Y_data, Y_pred, squared = False)
            model_comparison.loc["MAE"]["Multiple Linear Regression"] = mean_absolute_error(Y_data, Y_pred)
            model_comparison.loc["MaxErr"]["Multiple Linear Regression"] = max_error(Y_data, Y_pred)
            model_comparison.loc["EVRS"]["Multiple Linear Regression"] = explained_variance_score(Y_data, Y_pred)
            model_comparison.loc["SSR"]["Multiple Linear Regression"] = ((Y_data-Y_pred)**2).sum()
            residuals_comparison["Multiple Linear Regression"] = Y_data-Y_pred

            progress2 += 1
            my_bar.progress(progress2/len(algorithms))

        # Generalized Additive Models full model
        if any(a for a in algorithms if a == "Generalized Additive Models"):

            # Save results
            gam_reg_inf = pd.DataFrame(index = ["Distribution", "Link function", "Terms", "Features", "No. observations", "Effective DF", ], columns = ["Value"])
            gam_reg_stats = pd.DataFrame(index = ["Log-likelihood", "AIC", "AICc", "GCV", "Scale", "Pseudo R²"], columns = ["Value"])
            if gam_finalPara["intercept"][0] == "Yes":
                gam_reg_featSign = pd.DataFrame(index = expl_var + ["const"], columns = ["feature function", "coeff", "lambda", "rank", "edof", "p-value"])
            if gam_finalPara["intercept"][0] == "No":
                gam_reg_featSign = pd.DataFrame(index = expl_var, columns = ["feature function", "lambda", "rank", "edof", "p-value"])
            gam_reg_varImp = pd.DataFrame(index = expl_var, columns = ["mean", "std"])
            
            # Train GAM model
            if isinstance(gam_finalPara["spline order"][0], list):
                nos = gam_finalPara["number of splines"][0]
                so = gam_finalPara["spline order"][0]
                lam = gam_finalPara["lambda"][0]
            else:
                nos = int(gam_finalPara["number of splines"][0])
                so = int(gam_finalPara["spline order"][0])
                lam = float(gam_finalPara["lambda"][0])
            if gam_finalPara["intercept"][0] == "Yes":
                gam = LinearGAM(n_splines = nos, spline_order = so, lam = lam, fit_intercept = True).fit(X_data, Y_data)
            if gam_finalPara["intercept"][0] == "No":
                gam = LinearGAM(n_splines = nos, spline_order = so, lam = lam, fit_intercept = False).fit(X_data, Y_data)

            Y_pred = gam.predict(X_data)
            if data_new.empty == False:
                Y_pred_new = gam.predict(X_data_new)

            # Extract essential results from model
            # Information
            gam_reg_inf.loc["Distribution"] = gam.distribution
            gam_reg_inf.loc["Link function"] = gam.link
            gam_reg_inf.loc["Terms"] = str(gam.terms)
            gam_reg_inf.loc["Features"] = gam.statistics_["m_features"]
            gam_reg_inf.loc["No. observations"] = gam.statistics_["n_samples"]
            gam_reg_inf.loc["Effective DF"] = gam.statistics_["edof"]
            # Statistics
            gam_reg_stats.loc["Log-likelihood"] = gam.statistics_["loglikelihood"]
            gam_reg_stats.loc["AIC"] = gam.statistics_["AIC"]
            gam_reg_stats.loc["AICc"] = gam.statistics_["AICc"]
            gam_reg_stats.loc["GCV"] = gam.statistics_["GCV"]
            gam_reg_stats.loc["Scale"] = gam.statistics_["scale"]
            gam_reg_stats.loc["Pseudo R²"] = gam.statistics_["pseudo_r2"]["explained_deviance"]
            # Feature significance
            index_save_start = 0
            index_save_end = 0
            if gam_finalPara["intercept"][0] == "Yes":
                coef_list = expl_var + ["const"]
            if gam_finalPara["intercept"][0] == "No":
                coef_list = expl_var 
            for c in coef_list:
                if c != "const":
                    gam_reg_featSign.loc[c]["feature function"] = "s(" + str(c) + ")"
                    gam_reg_featSign.loc[c]["lambda"] = gam.lam[coef_list.index(c)][0]
                    gam_reg_featSign.loc[c]["rank"] = gam.n_splines[coef_list.index(c)]
                else:
                    gam_reg_featSign.loc[c]["feature function"] = "intercept"
                    gam_reg_featSign.loc[c]["coeff"] = gam.coef_[-1]
                    gam_reg_featSign.loc[c]["rank"] = 1
                index_save_end = index_save_start + gam_reg_featSign.loc[c]["rank"]
                gam_reg_featSign.loc[c]["edof"] = sum(gam.statistics_["edof_per_coef"][index_save_start:index_save_end])
                index_save_start = index_save_end
                gam_reg_featSign.loc[c]["p-value"] = gam.statistics_["p_values"][coef_list.index(c)]
            # Variable importance (via permutation, order important)
            scoring_function = make_scorer(r2_score, greater_is_better = True)
            gam_varImp = permutation_importance(gam , X_data, Y_data, n_repeats = 10, random_state = 0, scoring = scoring_function)
            for varI in expl_var:
                gam_reg_varImp.loc[varI]["mean"] = gam_varImp.importances_mean[expl_var.index(varI)]
                gam_reg_varImp.loc[varI]["std"] = gam_varImp.importances_std[expl_var.index(varI)]
            gam_reg_varImp = gam_reg_varImp.sort_values(by = ["mean"], ascending = False)
            # Partial dependence (order important)
            gam_pd = {}
            gam_pd_min_max = pd.DataFrame(index = expl_var, columns = ["min", "max"])
            for varPd in expl_var:
                term_index = expl_var.index(varPd)
                XX = gam.generate_X_grid(term=term_index)
                pdep, confi = gam.partial_dependence(term=term_index, X=XX, width=0.95)
                PD_data = pd.DataFrame(XX[:, term_index], columns = ["x_values"])
                PD_data["pd_values"] = pd.DataFrame(pdep)
                PD_data["lower_95"] = pd.DataFrame(confi[:,0])
                PD_data["upper_95"] = pd.DataFrame(confi[:,1])
                gam_pd[varPd] = PD_data
                gam_pd_min_max.loc[varPd]["min"] = PD_data["lower_95"].min()
                gam_pd_min_max.loc[varPd]["max"] = PD_data["upper_95"].max()  
            
            # Save tables
            full_model_results["GAM information"] = gam_reg_inf
            full_model_results["GAM statistics"] = gam_reg_stats
            full_model_results["GAM feature significance"] = gam_reg_featSign
            full_model_results["GAM variable importance"] = gam_reg_varImp
            full_model_results["GAM fitted"] = Y_pred
            full_model_results["GAM partial dependence"] = gam_pd
            full_model_results["GAM partial dependence min/max"] = gam_pd_min_max
            full_model_results["GAM Residual SE"] = np.sqrt(sum((Y_data-Y_pred)**2)/(Y_data.shape[0]- len(expl_var)-1))
            if data_new.empty == False:
                full_model_results["GAM prediction"] = Y_pred_new

            # Model comparison for GAM
            model_comparison.loc["% VE"]["Generalized Additive Models"] =  r2_score(Y_data, Y_pred)
            model_comparison.loc["MSE"]["Generalized Additive Models"] = mean_squared_error(Y_data, Y_pred, squared = True)
            model_comparison.loc["RMSE"]["Generalized Additive Models"] = mean_squared_error(Y_data, Y_pred, squared = False)
            model_comparison.loc["MAE"]["Generalized Additive Models"] = mean_absolute_error(Y_data, Y_pred)
            model_comparison.loc["MaxErr"]["Generalized Additive Models"] = max_error(Y_data, Y_pred)
            model_comparison.loc["EVRS"]["Generalized Additive Models"] = explained_variance_score(Y_data, Y_pred)
            model_comparison.loc["SSR"]["Generalized Additive Models"] = ((Y_data-Y_pred)**2).sum()
            residuals_comparison["Generalized Additive Models"] = Y_data-Y_pred

            progress2 += 1
            my_bar.progress(progress2/len(algorithms))

        # Random Forest full model
        if any(a for a in algorithms if a == "Random Forest"):
            
            # Save results
            rf_reg_inf = pd.DataFrame(index = ["Base estimator", "Estimators", "Features", "OOB score"], columns = ["Value"])
            rf_reg_featImp = pd.DataFrame(index = expl_var, columns = ["Value"])
            rf_reg_varImp = pd.DataFrame(index = expl_var, columns = ["mean", "std"])
            
            # Train RF model
            rf_final_para = final_hyPara_values["rf"]
            params = {'n_estimators': rf_final_para["number of trees"][0],
            'max_depth': rf_final_para["maximum tree depth"][0],
            'max_features': rf_final_para["maximum number of features"][0],
            'max_samples': rf_final_para["sample rate"][0],
            'bootstrap': True,
            'oob_score': True,
            }
            full_model_rf_sk = ensemble.RandomForestRegressor(**params)
            full_model_rf_sk.fit(X_data, Y_data)
            Y_pred = full_model_rf_sk.predict(X_data)
            if data_new.empty == False:
                Y_pred_new = full_model_rf_sk.predict(X_data_new)

            # Extract essential results from model
            # Information
            rf_reg_inf.loc["Base estimator"] = full_model_rf_sk.base_estimator_
            rf_reg_inf.loc["Estimators"] = len(full_model_rf_sk.estimators_)
            rf_reg_inf.loc["Features"] = full_model_rf_sk.n_features_in_
            rf_reg_inf.loc["OOB score"] = full_model_rf_sk.oob_score_
            # Feature importances (RF method)
            rf_featImp = full_model_rf_sk.feature_importances_
            for varI in expl_var:
                rf_reg_featImp.loc[varI]["Value"] = rf_featImp[expl_var.index(varI)]
            rf_reg_featImp = rf_reg_featImp.sort_values(by = ["Value"], ascending = False)
            # Variable importance (via permutation, order important)
            scoring_function = make_scorer(r2_score, greater_is_better = True)
            rf_varImp = permutation_importance(full_model_rf_sk , X_data, Y_data, n_repeats = 10, random_state = 0, scoring = scoring_function)
            for varI in expl_var:
                rf_reg_varImp.loc[varI]["mean"] = rf_varImp.importances_mean[expl_var.index(varI)]
                rf_reg_varImp.loc[varI]["std"] = rf_varImp.importances_std[expl_var.index(varI)]
            rf_reg_varImp = rf_reg_varImp.sort_values(by = ["mean"], ascending = False)
            # Partial dependence
            rf_pd = {}
            rf_pd_min_max = pd.DataFrame(index = expl_var, columns = ["min", "max"])
            #rf_partDep = plot_partial_dependence(full_model_rf_sk, X = X_data, features = expl_var, percentiles =(0, 1), method = "brute").pd_results
            rf_partDep = PartialDependenceDisplay.from_estimator(full_model_rf_sk, X = X_data, features = expl_var, percentiles =(0, 1), method = "brute").pd_results

            for varPd in expl_var:
                rf_pd[varPd] = rf_partDep[expl_var.index(varPd)]
                rf_pd_min_max.loc[varPd]["min"] = rf_partDep[expl_var.index(varPd)]["average"].min()
                rf_pd_min_max.loc[varPd]["max"] = rf_partDep[expl_var.index(varPd)]["average"].max()              
            
            # Save tables
            full_model_results["RF information"] = rf_reg_inf
            full_model_results["RF variable importance"] = rf_reg_varImp
            full_model_results["RF feature importance"] = rf_reg_featImp
            full_model_results["RF fitted"] = Y_pred
            full_model_results["RF partial dependence"] = rf_pd
            full_model_results["RF partial dependence min/max"] = rf_pd_min_max
            full_model_results["RF Residual SE"] = np.sqrt(sum((Y_data-Y_pred)**2)/(Y_data.shape[0]- len(expl_var)-1))
            if data_new.empty == False:
                full_model_results["RF prediction"] = Y_pred_new

            # Model comparison for BRT
            model_comparison.loc["% VE"]["Random Forest"] =  r2_score(Y_data, Y_pred)
            model_comparison.loc["MSE"]["Random Forest"] = mean_squared_error(Y_data, Y_pred, squared = True)
            model_comparison.loc["RMSE"]["Random Forest"] = mean_squared_error(Y_data, Y_pred, squared = False)
            model_comparison.loc["MAE"]["Random Forest"] = mean_absolute_error(Y_data, Y_pred)
            model_comparison.loc["MaxErr"]["Random Forest"] = max_error(Y_data, Y_pred)
            model_comparison.loc["EVRS"]["Random Forest"] = explained_variance_score(Y_data, Y_pred)
            model_comparison.loc["SSR"]["Random Forest"] = ((Y_data-Y_pred)**2).sum()
            residuals_comparison["Random Forest"] = Y_data-Y_pred

            progress2 += 1
            my_bar.progress(progress2/len(algorithms))

        # Boosted Regression Trees full model
        if any(a for a in algorithms if a == "Boosted Regression Trees"):
            
            # Save results
            brt_reg_inf = pd.DataFrame(index = ["Classes", "Estimators", "Features", "Loss function"], columns = ["Value"])
            brt_reg_featImp = pd.DataFrame(index = expl_var, columns = ["Value"])
            brt_reg_varImp = pd.DataFrame(index = expl_var, columns = ["mean", "std"])
            
            # Train BRT model
            brt_final_para = final_hyPara_values["brt"]
            params = {'n_estimators': brt_final_para["number of trees"][0],
            'learning_rate': brt_final_para["learning rate"][0],
            'max_depth': brt_final_para["maximum tree depth"][0],
            'subsample': brt_final_para["sample rate"][0]
            }
            full_model_brt_sk = ensemble.GradientBoostingRegressor(**params)
            full_model_brt_sk.fit(X_data, Y_data.values.ravel())
            Y_pred = full_model_brt_sk.predict(X_data)
            if data_new.empty == False:
                Y_pred_new = full_model_brt_sk.predict(X_data_new)

            # Extract essential results from model
            # Information
            #brt_reg_inf.loc["Classes"] = full_model_brt_sk.n_classes_
            brt_reg_inf.loc["Estimators"] = full_model_brt_sk.n_estimators_
            brt_reg_inf.loc["Features"] = full_model_brt_sk.n_features_in_
            brt_reg_inf.loc["Loss function"] = full_model_brt_sk.loss
            # Feature importances (BRT method)
            brt_featImp = full_model_brt_sk.feature_importances_
            for varI in expl_var:
                brt_reg_featImp.loc[varI]["Value"] = brt_featImp[expl_var.index(varI)]
            brt_reg_featImp = brt_reg_featImp.sort_values(by = ["Value"], ascending = False)
            # Variable importance (via permutation, order important)
            scoring_function = make_scorer(r2_score, greater_is_better = True)
            brt_varImp = permutation_importance(full_model_brt_sk , X_data, Y_data.values.ravel(), n_repeats = 10, random_state = 0, scoring = scoring_function)
            for varI in expl_var:
                brt_reg_varImp.loc[varI]["mean"] = brt_varImp.importances_mean[expl_var.index(varI)]
                brt_reg_varImp.loc[varI]["std"] = brt_varImp.importances_std[expl_var.index(varI)]
            brt_reg_varImp = brt_reg_varImp.sort_values(by = ["mean"], ascending = False)
            # Partial dependence
            brt_pd = {}
            brt_pd_min_max = pd.DataFrame(index = expl_var, columns = ["min", "max"])
            #brt_partDep = plot_partial_dependence(full_model_brt_sk, X = X_data, features = expl_var, percentiles =(0, 1), method = "brute").pd_results
            brt_partDep = PartialDependenceDisplay.from_estimator(full_model_brt_sk, X_data, features=expl_var, percentiles=(0, 1), method="brute").pd_results
                        
            for varPd in expl_var:                              
                brt_pd[varPd] = brt_partDep[expl_var.index(varPd)] 
                brt_pd_min_max.loc[varPd]["min"] = brt_partDep[expl_var.index(varPd)]["average"].min()
                brt_pd_min_max.loc[varPd]["max"] = brt_partDep[expl_var.index(varPd)]["average"].max()              
            #######
            
            # Save tables
            full_model_results["BRT information"] = brt_reg_inf
            full_model_results["BRT variable importance"] = brt_reg_varImp
            full_model_results["BRT feature importance"] = brt_reg_featImp
            full_model_results["BRT fitted"] = Y_pred
            full_model_results["BRT partial dependence"] = brt_pd
            full_model_results["BRT partial dependence min/max"] = brt_pd_min_max
            full_model_results["BRT train score"] = full_model_brt_sk.train_score_
            full_model_results["BRT Residual SE"] = np.sqrt(sum((Y_data-Y_pred)**2)/(Y_data.shape[0]- len(expl_var)-1))
            if data_new.empty == False:
                full_model_results["BRT prediction"] = Y_pred_new

            # Model comparison for BRT
            model_comparison.loc["% VE"]["Boosted Regression Trees"] =  r2_score(Y_data, Y_pred)
            model_comparison.loc["MSE"]["Boosted Regression Trees"] = mean_squared_error(Y_data, Y_pred, squared = True)
            model_comparison.loc["RMSE"]["Boosted Regression Trees"] = mean_squared_error(Y_data, Y_pred, squared = False)
            model_comparison.loc["MAE"]["Boosted Regression Trees"] = mean_absolute_error(Y_data, Y_pred)
            model_comparison.loc["MaxErr"]["Boosted Regression Trees"] = max_error(Y_data, Y_pred)
            model_comparison.loc["EVRS"]["Boosted Regression Trees"] = explained_variance_score(Y_data, Y_pred)
            model_comparison.loc["SSR"]["Boosted Regression Trees"] = ((Y_data-Y_pred)**2).sum()
            residuals_comparison["Boosted Regression Trees"] = Y_data-Y_pred

            progress2 += 1
            my_bar.progress(progress2/len(algorithms))


        # Artificial Neural Networks full model
        if any(a for a in algorithms if a == "Artificial Neural Networks"):

            # Save results
            ann_reg_inf = pd.DataFrame(index = ["Outputs", "Layers", "Training samples", "Output activation", "Loss function"], columns = ["Value"])
            ann_reg_varImp = pd.DataFrame(index = expl_var, columns = ["mean", "std"])
            
            # Train ANN model
            scaler = StandardScaler()
            scaler.fit(X_data)
            X_data_ann = scaler.transform(X_data)
            X_data_ann = pd.DataFrame(X_data_ann, index = X_data.index, columns = X_data.columns)
            ann_final_para = final_hyPara_values["ann"]
            params = {"solver": ann_final_para["weight optimization solver"][0],
            "max_iter": ann_final_para["maximum number of iterations"][0],
            "activation": ann_final_para["activation function"][0],
            "hidden_layer_sizes": ann_final_para["hidden layer sizes"][0],
            "learning_rate_init": ann_final_para["learning rate"][0],
            #"learning_rate": ann_final_para["learning rate schedule"][0],
            #"momentum": ann_final_para["momentum"][0],
            "alpha": ann_final_para["L² regularization"][0],
            "random_state": 0
            #"epsilon": ann_final_para["epsilon"][0]
            }
            full_model_ann_sk = MLPRegressor(**params)
            full_model_ann_sk.fit(X_data_ann, Y_data)
            Y_pred = full_model_ann_sk.predict(X_data_ann)
            if data_new.empty == False:
                X_data_new_ann = scaler.transform(X_data_new)
                Y_pred_new = full_model_ann_sk.predict(X_data_new_ann)
            
            # Extract essential results from model
            # Regression information
            ann_reg_inf.loc["Outputs"] = full_model_ann_sk.n_outputs_
            ann_reg_inf.loc["Layers"] = full_model_ann_sk.n_layers_
            ann_reg_inf.loc["Training samples"] = full_model_ann_sk.t_
            ann_reg_inf.loc["Output activation"] = full_model_ann_sk.out_activation_
            ann_reg_inf.loc["Loss function"] = full_model_ann_sk.loss
            # Variable importance (via permutation, order important)
            scoring_function = make_scorer(r2_score, greater_is_better = True)
            ann_varImp = permutation_importance(full_model_ann_sk , X_data_ann, Y_data, n_repeats = 10, random_state = 0, scoring = scoring_function)
            for varI in expl_var:
                ann_reg_varImp.loc[varI]["mean"] = ann_varImp.importances_mean[expl_var.index(varI)]
                ann_reg_varImp.loc[varI]["std"] = ann_varImp.importances_std[expl_var.index(varI)]
            ann_reg_varImp = ann_reg_varImp.sort_values(by = ["mean"], ascending = False)
            # Partial dependence
            ann_pd = {}
            ann_pd_min_max = pd.DataFrame(index = expl_var, columns = ["min", "max"])
            #ann_partDep = plot_partial_dependence(full_model_ann_sk, X = X_data_ann, features = expl_var, percentiles =(0, 1), method = "brute").pd_results
            ann_partDep = PartialDependenceDisplay.from_estimator(full_model_ann_sk, X = X_data_ann, features = expl_var, percentiles =(0, 1), method = "brute").pd_results
            for varPd in expl_var:
                ann_pd[varPd] = ann_partDep[expl_var.index(varPd)]
                ann_pd_min_max.loc[varPd]["min"] = ann_partDep[expl_var.index(varPd)]["average"].min()
                ann_pd_min_max.loc[varPd]["max"] = ann_partDep[expl_var.index(varPd)]["average"].max()    

            # Save tables
            full_model_results["ANN information"] = ann_reg_inf
            full_model_results["ANN variable importance"] = ann_reg_varImp
            full_model_results["ANN fitted"] = Y_pred
            full_model_results["ANN partial dependence"] = ann_pd
            full_model_results["ANN partial dependence min/max"] = ann_pd_min_max
            if ann_final_para["weight optimization solver"][0] != "lbfgs":
                full_model_results["ANN loss curve"] = full_model_ann_sk.loss_curve_
                full_model_results["ANN loss"] = full_model_ann_sk.best_loss_
            full_model_results["ANN Residual SE"] = np.sqrt(sum((Y_data-Y_pred)**2)/(Y_data.shape[0]- len(expl_var)-1))
            if data_new.empty == False:
                full_model_results["ANN prediction"] = Y_pred_new

            # Model comparison for ANN
            model_comparison.loc["% VE"]["Artificial Neural Networks"] =  r2_score(Y_data, Y_pred)
            model_comparison.loc["MSE"]["Artificial Neural Networks"] = mean_squared_error(Y_data, Y_pred, squared = True)
            model_comparison.loc["RMSE"]["Artificial Neural Networks"] = mean_squared_error(Y_data, Y_pred, squared = False)
            model_comparison.loc["MAE"]["Artificial Neural Networks"] = mean_absolute_error(Y_data, Y_pred)
            model_comparison.loc["MaxErr"]["Artificial Neural Networks"] = max_error(Y_data, Y_pred)
            model_comparison.loc["EVRS"]["Artificial Neural Networks"] = explained_variance_score(Y_data, Y_pred)
            model_comparison.loc["SSR"]["Artificial Neural Networks"] = ((Y_data-Y_pred)**2).sum()
            residuals_comparison["Artificial Neural Networks"] = Y_data-Y_pred
            
            progress2 += 1
            my_bar.progress(progress2/len(algorithms))
        
        # Save model comparison
        full_model_results["model comparison"] = model_comparison
        full_model_results["residuals"] = residuals_comparison
        
    #-----------------------------------------------------------------------------------------
    # Binary response variables

    if response_var_type == "binary":

        # Remove MLR from typical binary outputs
        alg = algorithms.copy()
        if any(a for a in algorithms if a == "Multiple Linear Regression"):  
            alg.remove("Multiple Linear Regression")

        # Collect performance results across all algorithms
        model_comparison_thresInd = pd.DataFrame(index = ["AUC ROC", "AP", "AUC PRC", "LOG-LOSS"], columns = alg)
        model_comparison_thres = pd.DataFrame(index = ["threshold"], columns = alg)
        model_comparison_thresDep = pd.DataFrame(index = ["TPR", "FNR", "TNR", "FPR", "TSS", "PREC", "F1", "KAPPA", "ACC", "BAL ACC"], columns = alg)
        # MLR residuals
        residuals_comparison = pd.DataFrame(columns = ["Multiple Linear Regression"])

        # Multiple Linear Regression full model
        if any(a for a in algorithms if a == "Multiple Linear Regression"):
            
            # Extract parameters
            MLR_intercept = MLR_finalPara["intercept"][0]
            MLR_cov_type = MLR_finalPara["covType"][0]

            # Save results
            mlr_reg_inf = pd.DataFrame(index = ["Dep. variable", "Model", "Method", "No. observations", "DF residuals", "DF model", "Covariance type"], columns = ["Value"])
            mlr_reg_stats = pd.DataFrame(index = ["R²", "Adj. R²", "Mult. corr. coeff.", "Residual SE", "Log-likelihood", "AIC", "BIC"], columns = ["Value"])
            mlr_reg_anova = pd.DataFrame(index = ["Regression", "Residual", "Total"], columns = ["DF", "SS", "MS", "F-statistic", "p-value"])
            if MLR_intercept == "Yes":
                mlr_reg_coef = pd.DataFrame(index = ["const"]+ expl_var, columns = ["coeff", "std err", "t-statistic", "p-value", "lower 95%", "upper 95%"])
            if MLR_intercept == "No":
                mlr_reg_coef = pd.DataFrame(index = expl_var, columns = ["coeff", "std err", "t-statistic", "p-value", "lower 95%", "upper 95%"])
            mlr_reg_hetTest = pd.DataFrame(index = ["test statistic", "p-value"], columns = ["Breusch-Pagan test", "White test (without int.)", "White test (with int.)"])
            mlr_reg_varImp = pd.DataFrame(index = expl_var, columns = ["mean", "std"])

            # Train MLR model (statsmodels)
            if MLR_intercept == "Yes":
                X_data_mlr = sm.add_constant(X_data)
            if MLR_intercept == "No":
                X_data_mlr = X_data
            if MLR_model == "OLS":
                full_model_mlr = sm.OLS(Y_data, X_data_mlr)
            # if MLR_model == "GLS":
            #     ols_resid = sm.OLS(Y_data, X_data_mlr).fit().resid
            #     res_fit = sm.OLS(np.array(ols_resid[1:]), np.array(ols_resid[:-1])).fit()
            #     rho = res_fit.params
            #     order = toeplitz(np.arange(data.shape[0]))
            #     sigma = rho**order
            #     full_model_mlr = sm.GLS(Y_data, X_data_mlr, sigma=sigma)
            if MLR_cov_type == "non-robust":
                full_model_fit = full_model_mlr.fit()
            else:
                full_model_fit = full_model_mlr.fit().get_robustcov_results(cov_type = MLR_cov_type)
            Y_pred = full_model_fit.predict(X_data_mlr)
            Y_pred = Y_pred.to_numpy()
            if data_new.empty == False:
                if MLR_intercept == "Yes":
                    X_data_new_mlr = sm.add_constant(X_data_new)
                if MLR_intercept == "No":
                    X_data_new_mlr = X_data_new
                Y_pred_new = full_model_fit.predict(X_data_new_mlr)
                Y_pred_new = Y_pred_new.to_numpy()

            # Train MLR model (sklearn)
            if MLR_intercept == "Yes":
                full_model_mlr_sk = LinearRegression(fit_intercept=True)
            if MLR_intercept == "No":
                full_model_mlr_sk = LinearRegression(fit_intercept=False)    
            full_model_mlr_sk.fit(X_data, Y_data)
            # Y_pred = full_model_mlr_sk.predict(X_data)

            # Extract essential results from model
            # Information
            mlr_reg_inf.loc["Dep. variable"] = full_model_fit.model.endog_names
            mlr_reg_inf.loc["Model"] = MLR_model
            mlr_reg_inf.loc["Method"] = "Least squares"
            mlr_reg_inf.loc["No. observations"] = full_model_fit.model.nobs
            mlr_reg_inf.loc["DF residuals"] = full_model_fit.df_resid
            mlr_reg_inf.loc["DF model"] = full_model_fit.df_model
            mlr_reg_inf.loc["Covariance type"] = full_model_fit.cov_type
            # Statistics
            mlr_reg_stats.loc["R²"] = full_model_fit.rsquared
            mlr_reg_stats.loc["Adj. R²"] = full_model_fit.rsquared_adj
            mlr_reg_stats.loc["Mult. corr. coeff."] = np.sqrt(full_model_fit.rsquared)
            mlr_reg_stats.loc["Residual SE"] = np.sqrt(full_model_fit.mse_resid)
            mlr_reg_stats.loc["Log-likelihood"] = full_model_fit.llf
            mlr_reg_stats.loc["AIC"] = full_model_fit.aic
            mlr_reg_stats.loc["BIC"] = full_model_fit.bic
            # ANOVA
            mlr_reg_anova.loc["Regression"]["DF"] = full_model_fit.df_model
            mlr_reg_anova.loc["Regression"]["SS"] = full_model_fit.ess
            mlr_reg_anova.loc["Regression"]["MS"] = full_model_fit.ess/full_model_fit.df_model
            mlr_reg_anova.loc["Regression"]["F-statistic"] = full_model_fit.fvalue
            mlr_reg_anova.loc["Regression"]["p-value"] = full_model_fit.f_pvalue
            mlr_reg_anova.loc["Residual"]["DF"] = full_model_fit.df_resid
            mlr_reg_anova.loc["Residual"]["SS"] = full_model_fit.ssr
            mlr_reg_anova.loc["Residual"]["MS"] = full_model_fit.ssr/full_model_fit.df_resid
            mlr_reg_anova.loc["Residual"]["F-statistic"] = ""
            mlr_reg_anova.loc["Residual"]["p-value"] = ""
            mlr_reg_anova.loc["Total"]["DF"] = full_model_fit.df_resid + full_model_fit.df_model
            mlr_reg_anova.loc["Total"]["SS"] = full_model_fit.ssr + full_model_fit.ess
            mlr_reg_anova.loc["Total"]["MS"] = ""
            mlr_reg_anova.loc["Total"]["F-statistic"] = ""
            mlr_reg_anova.loc["Total"]["p-value"] = ""
            # Coefficients
            if MLR_intercept == "Yes":
                coef_list = ["const"]+ expl_var
            if MLR_intercept == "No":
                coef_list = expl_var
            for c in coef_list:
                mlr_reg_coef.loc[c]["coeff"] = full_model_fit.params[coef_list.index(c)]
                mlr_reg_coef.loc[c]["std err"] = full_model_fit.bse[coef_list.index(c)]
                mlr_reg_coef.loc[c]["t-statistic"] = full_model_fit.tvalues[coef_list.index(c)]
                mlr_reg_coef.loc[c]["p-value"] = full_model_fit.pvalues[coef_list.index(c)]
                if MLR_cov_type == "non-robust":
                    mlr_reg_coef.loc[c]["lower 95%"] = full_model_fit.conf_int(alpha = 0.05).loc[c][0]
                    mlr_reg_coef.loc[c]["upper 95%"] = full_model_fit.conf_int(alpha = 0.05).loc[c][1]
                else:
                    mlr_reg_coef.loc[c]["lower 95%"] = full_model_fit.conf_int(alpha = 0.05)[coef_list.index(c)][0]
                    mlr_reg_coef.loc[c]["upper 95%"] = full_model_fit.conf_int(alpha = 0.05)[coef_list.index(c)][1]
            if MLR_intercept == "Yes":
                # Breusch-Pagan heteroscedasticity test
                bp_result = sm.stats.diagnostic.het_breuschpagan(full_model_fit.resid, full_model_fit.model.exog) 
                mlr_reg_hetTest.loc["test statistic"]["Breusch-Pagan test"] = bp_result[0]
                mlr_reg_hetTest.loc["p-value"]["Breusch-Pagan test"] = bp_result[1]
                # White heteroscedasticity test with interaction
                white_int_result = sm.stats.diagnostic.het_white(full_model_fit.resid, full_model_fit.model.exog)
                mlr_reg_hetTest.loc["test statistic"]["White test (with int.)"] = white_int_result[0]
                mlr_reg_hetTest.loc["p-value"]["White test (with int.)"] = white_int_result[1]
                # White heteroscedasticity test without interaction
                X_data_mlr_white = X_data_mlr
                for i in expl_var: 
                    X_data_mlr_white[i+ "_squared"] = X_data_mlr_white[i]**2
                white = sm.OLS(full_model_fit.resid**2, X_data_mlr_white)
                del X_data_mlr_white
                white_fit = white.fit()
                white_statistic = white_fit.rsquared*data.shape[0]
                white_p_value = stats.chi2.sf(white_statistic,len(white_fit.model.exog_names)-1)
                mlr_reg_hetTest.loc["test statistic"]["White test (without int.)"] = white_statistic
                mlr_reg_hetTest.loc["p-value"]["White test (without int.)"] = white_p_value

            # Variable importance (via permutation, order important)
            scoring_function = make_scorer(r2_score, greater_is_better = True)
            mlr_varImp = permutation_importance(full_model_mlr_sk , X_data, Y_data, n_repeats = 10, random_state = 0, scoring = scoring_function)
            for varI in expl_var:
                mlr_reg_varImp.loc[varI]["mean"] = mlr_varImp.importances_mean[expl_var.index(varI)]
                mlr_reg_varImp.loc[varI]["std"] = mlr_varImp.importances_std[expl_var.index(varI)]
            mlr_reg_varImp = mlr_reg_varImp.sort_values(by=["mean"], ascending = False)

            # Save tables
            full_model_results["MLR information"] = mlr_reg_inf
            full_model_results["MLR statistics"] = mlr_reg_stats
            full_model_results["MLR ANOVA"] = mlr_reg_anova
            full_model_results["MLR coefficients"] = mlr_reg_coef
            full_model_results["MLR hetTest"] = mlr_reg_hetTest
            full_model_results["MLR fitted"] = Y_pred
            full_model_results["MLR Cooks distance"] = full_model_fit.get_influence().cooks_distance[0]
            full_model_results["MLR leverage"] = full_model_fit.get_influence().hat_matrix_diag
            full_model_results["MLR variable importance"] = mlr_reg_varImp
            if data_new.empty == False:
                full_model_results["MLR prediction"] = Y_pred_new 

            # Residuals MLR
            residuals_comparison["Multiple Linear Regression"] = Y_data-Y_pred

            progress2 += 1
            my_bar.progress(progress2/len(algorithms))

        # Logistic Regression full model
        if any(a for a in algorithms if a == "Logistic Regression"):

            # Extract parameters
            LR_intercept = LR_finalPara["intercept"][0]
            LR_cov_type = LR_finalPara["covType"][0]
            
            # Save results
            lr_reg_inf = pd.DataFrame(index = ["Dep. variable", "Model", "Method", "No. observations", "DF residuals", "DF model", "Converged", "Iterations", "Covariance type"], columns = ["Value"])
            lr_reg_stats = pd.DataFrame(index = ["AUC ROC", "Pseudo R²", "Log-Likelihood", "LL-Null", "Residual deviance", "Null deviance", "LLR", "LLR p-value", "AIC", "BIC"], columns = ["Value"])
            if LR_intercept == "Yes":
                lr_reg_coef = pd.DataFrame(index = ["const"]+ expl_var, columns = ["coeff", "std err", "t-statistic", "p-value", "lower 95%", "upper 95%"])
            if LR_intercept == "No":
                lr_reg_coef = pd.DataFrame(index = expl_var, columns = ["coeff", "std err", "t-statistic", "p-value", "lower 95%", "upper 95%"])
            lr_reg_varImp = pd.DataFrame(index = expl_var, columns = ["mean", "std"])

            # Train LR model (statsmodels)
            if LR_intercept == "Yes":
                X_data_lr = sm.add_constant(X_data)
            if LR_intercept == "No":
                X_data_lr = X_data
            full_model_lr = sm.Logit(Y_data, X_data_lr)
            if LR_cov_type == "non-robust":
                full_model_fit = full_model_lr.fit(method = "ncg", maxiter = 100)
            else:
                full_model_fit = full_model_lr.fit(method = "ncg", maxiter = 100, cov_type = LR_cov_type)
            Y_pred = 1-pd.DataFrame(full_model_fit.predict(X_data_lr), columns = ["0"])
            Y_pred["1"] = 1-Y_pred
            Y_pred = Y_pred.to_numpy()
            if data_new.empty == False:
                if LR_intercept == "Yes":
                    X_data_new_lr = sm.add_constant(X_data_new)
                if LR_intercept == "No":
                    X_data_new_lr = X_data_new
                Y_pred_new = 1-pd.DataFrame(full_model_fit.predict(X_data_new_lr), columns = ["0"])
                Y_pred_new["1"] = 1-Y_pred_new
                Y_pred_new = Y_pred_new.to_numpy()

            # Train LR model (sklearn)
            if LR_intercept == "Yes":
                full_model_lr_sk = LogisticRegression(fit_intercept = True, solver = "newton-cg", penalty = "none", tol = 1e-05)
            if LR_intercept == "No":
                full_model_lr_sk = LogisticRegression(fit_intercept = False, solver = "newton-cg", penalty = "none", tol = 1e-05)
            full_model_lr_sk.fit(X_data, Y_data)
            # Y_pred = full_model_lr_sk.predict_proba(X_data)

            # Extract essential results from model
            # Information
            lr_reg_inf.loc["Dep. variable"] = full_model_fit.model.endog_names
            lr_reg_inf.loc["Model"] = "Logit"
            lr_reg_inf.loc["Method"] = "MLE"
            lr_reg_inf.loc["No. observations"] = full_model_fit.nobs
            lr_reg_inf.loc["DF residuals"] = full_model_fit.df_resid
            lr_reg_inf.loc["DF model"] = full_model_fit.df_model
            lr_reg_inf.loc["Converged"] = full_model_fit.mle_retvals["converged"]
            lr_reg_inf.loc["Iterations"] = full_model_fit.mle_retvals["hcalls"]
            lr_reg_inf.loc["Covariance type"] = full_model_fit.cov_type
            # Statistics
            lr_reg_stats.loc["AUC ROC"] = roc_auc_score(Y_data, Y_pred[:, 1])
            lr_reg_stats.loc["Pseudo R²"] = full_model_fit.prsquared
            lr_reg_stats.loc["Log-Likelihood"] = full_model_fit.llf
            lr_reg_stats.loc["LL-Null"] = full_model_fit.llnull
            lr_reg_stats.loc["Residual deviance"] = -2*full_model_fit.llf
            lr_reg_stats.loc["Null deviance"] = -2*full_model_fit.llnull
            lr_reg_stats.loc["LLR"] = full_model_fit.llr
            lr_reg_stats.loc["LLR p-value"] = full_model_fit.llr_pvalue
            lr_reg_stats.loc["AIC"] = full_model_fit.aic
            lr_reg_stats.loc["BIC"] = full_model_fit.bic
            # Coefficients
            if LR_intercept == "Yes":
                coef_list = ["const"]+ expl_var
            if LR_intercept == "No":
                coef_list = expl_var
            for c in coef_list:
                lr_reg_coef.loc[c]["coeff"] = full_model_fit.params[c]
                lr_reg_coef.loc[c]["std err"] = full_model_fit.bse[c]
                lr_reg_coef.loc[c]["t-statistic"] = full_model_fit.tvalues[c]
                lr_reg_coef.loc[c]["p-value"] = full_model_fit.pvalues[c]
                lr_reg_coef.loc[c]["lower 95%"] = full_model_fit.conf_int(alpha = 0.05).loc[c][0]
                lr_reg_coef.loc[c]["upper 95%"] = full_model_fit.conf_int(alpha = 0.05).loc[c][1]
            # Variable importance (via permutation, order important)
            scoring_function = make_scorer(roc_auc_score, greater_is_better = True)
            lr_varImp = permutation_importance(full_model_lr_sk , X_data, Y_data, n_repeats = 10, random_state = 0, scoring = scoring_function)
            for varI in expl_var:
                lr_reg_varImp.loc[varI]["mean"] = lr_varImp.importances_mean[expl_var.index(varI)]
                lr_reg_varImp.loc[varI]["std"] = lr_varImp.importances_std[expl_var.index(varI)]
            lr_reg_varImp = lr_reg_varImp.sort_values(by=["mean"], ascending = False)
            # Partial probabilities
            lr_partial_probs = {}
            for varPP in expl_var:
                PP_data_lr = sm.add_constant(X_data[varPP])
                PP_model_lr = sm.Logit(Y_data, PP_data_lr)
                PP_model_fit = PP_model_lr.fit(method = "ncg", maxiter = 100)
                PP_pred = pd.DataFrame()
                PP_pred["prediction"] = PP_model_fit.predict(PP_data_lr)
                PP_pred[varPP] = X_data[varPP]
                lr_partial_probs[varPP] = PP_pred

            # Save tables
            full_model_results["LR information"] = lr_reg_inf
            full_model_results["LR statistics"] = lr_reg_stats
            full_model_results["LR coefficients"] = lr_reg_coef
            full_model_results["LR ROC curve"] = roc_curve(Y_data, Y_pred[:,1])
            full_model_results["LR fitted"] = Y_pred
            full_model_results["LR variable importance"] = lr_reg_varImp
            full_model_results["LR partial probabilities"] = lr_partial_probs
            if data_new.empty == False:
                full_model_results["LR prediction"] = Y_pred_new 

            # Model comparison for LR
            model_comparison_thresInd.loc["AUC ROC"]["Logistic Regression"] = roc_auc_score(Y_data, Y_pred[:, 1])
            model_comparison_thresInd.loc["AP"]["Logistic Regression"] = average_precision_score(Y_data, Y_pred[:, 1])
            precision, recall, thresholds = precision_recall_curve(Y_data, Y_pred[:, 1])
            model_comparison_thresInd.loc["AUC PRC"]["Logistic Regression"] =  auc(recall, precision)
            model_comparison_thresInd.loc["LOG-LOSS"]["Logistic Regression"] =  log_loss(Y_data, Y_pred)
            
            # Threshold according to Youden's index
            FPR, TPR, thresholds = roc_curve(Y_data, Y_pred[:, 1])
            thres_index = np.argmax(TPR - FPR)
            thres = thresholds[thres_index]
            model_comparison_thres.loc["threshold"]["Logistic Regression"] = thres
            
            # Threshold determination (minimizing abs. distance between SENS & SPEC)
            # FPR, TPR, thresholds = roc_curve(Y_test, Y_test_pred[:, 1])
            # thres_index = np.argmin(abs(TPR - (1-FPR)))
            # thres = thresholds[thres_index]

            Y_pred_bin = np.array([1 if x >= thres else 0 for x in Y_pred[:, 1]])
            full_model_results["LR fitted binary"] = Y_pred_bin
            if data_new.empty == False:
                Y_pred_new_bin = np.array([1 if x >= thres else 0 for x in Y_pred_new[:, 1]])
                full_model_results["LR prediction binary"] = Y_pred_new_bin 
            TN, FP, FN, TP = confusion_matrix(Y_data, Y_pred_bin).ravel()
            model_comparison_thresDep.loc["TPR"]["Logistic Regression"] =  TP/(TP+FN)
            model_comparison_thresDep.loc["FNR"]["Logistic Regression"] = FN/(TP+FN)
            model_comparison_thresDep.loc["TNR"]["Logistic Regression"] = TN/(TN+FP)
            model_comparison_thresDep.loc["FPR"]["Logistic Regression"] = FP/(TN+FP)
            model_comparison_thresDep.loc["TSS"]["Logistic Regression"] = (TP/(TP+FN))+(TN/(TN+FP))-1
            model_comparison_thresDep.loc["PREC"]["Logistic Regression"] = TP/(TP+FP)
            model_comparison_thresDep.loc["F1"]["Logistic Regression"] = f1_score(Y_data, Y_pred_bin)
            model_comparison_thresDep.loc["KAPPA"]["Logistic Regression"] = cohen_kappa_score(Y_data, Y_pred_bin)
            model_comparison_thresDep.loc["ACC"]["Logistic Regression"] = accuracy_score(Y_data, Y_pred_bin)
            model_comparison_thresDep.loc["BAL ACC"]["Logistic Regression"] = balanced_accuracy_score(Y_data, Y_pred_bin)

            progress2 += 1
            my_bar.progress(progress2/len(algorithms))

        # Generalized Additive Models full model
        if any(a for a in algorithms if a == "Generalized Additive Models"):

            # Save results
            gam_reg_inf = pd.DataFrame(index = ["Distribution", "Link function", "Terms", "Features", "No. observations", "Effective DF"], columns = ["Value"])
            gam_reg_stats = pd.DataFrame(index = ["Log-likelihood", "AIC", "AICc", "UBRE", "Scale", "Pseudo R²", "AUC ROC"], columns = ["Value"])
            if gam_finalPara["intercept"][0] == "Yes":
                gam_reg_featSign = pd.DataFrame(index = expl_var + ["const"], columns = ["feature function", "coeff", "lambda", "rank", "edof", "p-value"])
            if gam_finalPara["intercept"][0] == "No":
                gam_reg_featSign = pd.DataFrame(index = expl_var, columns = ["feature function", "lambda", "rank", "edof", "p-value"])
            gam_reg_varImp = pd.DataFrame(index = expl_var, columns = ["mean", "std"])
            
            # Train GAM model
            if isinstance(gam_finalPara["spline order"][0], list):
                nos = gam_finalPara["number of splines"][0]
                so = gam_finalPara["spline order"][0]
                lam = gam_finalPara["lambda"][0]
            else:
                nos = int(gam_finalPara["number of splines"][0])
                so = int(gam_finalPara["spline order"][0])
                lam = float(gam_finalPara["lambda"][0])
            if gam_finalPara["intercept"][0] == "Yes":
                gam = LogisticGAM(n_splines = nos, spline_order = so, lam = lam, fit_intercept = True).fit(X_data, Y_data)
            if gam_finalPara["intercept"][0] == "No":
                gam = LogisticGAM(n_splines = nos, spline_order = so, lam = lam, fit_intercept = False).fit(X_data, Y_data)
            Y_pred = gam.predict_proba(X_data)
            if data_new.empty == False:
                Y_pred_new = gam.predict_proba(X_data_new)

            # Extract essential results from model
            # Information
            gam_reg_inf.loc["Distribution"] = gam.distribution
            gam_reg_inf.loc["Link function"] = gam.link
            gam_reg_inf.loc["Terms"] = str(gam.terms)
            gam_reg_inf.loc["Features"] = gam.statistics_["m_features"]
            gam_reg_inf.loc["No. observations"] = gam.statistics_["n_samples"]
            gam_reg_inf.loc["Effective DF"] = gam.statistics_["edof"]
            # Statistics
            gam_reg_stats.loc["Log-likelihood"] = gam.statistics_["loglikelihood"]
            gam_reg_stats.loc["AIC"] = gam.statistics_["AIC"]
            gam_reg_stats.loc["AICc"] = gam.statistics_["AICc"]
            gam_reg_stats.loc["UBRE"] = gam.statistics_["UBRE"]
            gam_reg_stats.loc["Scale"] = gam.statistics_["scale"]
            gam_reg_stats.loc["Pseudo R²"] = gam.statistics_["pseudo_r2"]["explained_deviance"]
            gam_reg_stats.loc["AUC ROC"] = roc_auc_score(Y_data, Y_pred)
            # Feature significance
            index_save_start = 0
            index_save_end = 0
            if gam_finalPara["intercept"][0] == "Yes":
                coef_list = expl_var + ["const"]
            if gam_finalPara["intercept"][0] == "No":
                coef_list = expl_var 
            for c in coef_list:
                if c != "const":
                    gam_reg_featSign.loc[c]["feature function"] = "s(" + str(c) + ")"
                    gam_reg_featSign.loc[c]["lambda"] = gam.lam[coef_list.index(c)][0]
                    gam_reg_featSign.loc[c]["rank"] = gam.n_splines[coef_list.index(c)]
                else:
                    gam_reg_featSign.loc[c]["feature function"] = "intercept"
                    gam_reg_featSign.loc[c]["coeff"] = gam.coef_[-1]
                    gam_reg_featSign.loc[c]["rank"] = 1
                index_save_end = index_save_start + gam_reg_featSign.loc[c]["rank"]
                gam_reg_featSign.loc[c]["edof"] = sum(gam.statistics_["edof_per_coef"][index_save_start:index_save_end])
                index_save_start = index_save_end
                gam_reg_featSign.loc[c]["p-value"] = gam.statistics_["p_values"][coef_list.index(c)]
            # Variable importance (via permutation, order important)
            scoring_function = make_scorer(roc_auc_score, greater_is_better = True)
            gam_varImp = permutation_importance(gam , X_data, Y_data, n_repeats = 10, random_state = 0, scoring = scoring_function)
            for varI in expl_var:
                gam_reg_varImp.loc[varI]["mean"] = gam_varImp.importances_mean[expl_var.index(varI)]
                gam_reg_varImp.loc[varI]["std"] = gam_varImp.importances_std[expl_var.index(varI)]
            gam_reg_varImp = gam_reg_varImp.sort_values(by = ["mean"], ascending = False)
            # Partial dependence (order important)
            gam_pd = {}
            gam_pd_min_max = pd.DataFrame(index = expl_var, columns = ["min", "max"])
            for varPd in expl_var:
                term_index = expl_var.index(varPd)
                XX = gam.generate_X_grid(term=term_index)
                pdep, confi = gam.partial_dependence(term=term_index, X=XX, width=0.95)
                PD_data = pd.DataFrame(XX[:, term_index], columns = ["x_values"])
                PD_data["pd_values"] = pd.DataFrame(pdep)
                PD_data["lower_95"] = pd.DataFrame(confi[:,0])
                PD_data["upper_95"] = pd.DataFrame(confi[:,1])
                gam_pd[varPd] = PD_data
                gam_pd_min_max.loc[varPd]["min"] = PD_data["lower_95"].min()
                gam_pd_min_max.loc[varPd]["max"] = PD_data["upper_95"].max()  
            
            # Save tables
            full_model_results["GAM information"] = gam_reg_inf
            full_model_results["GAM statistics"] = gam_reg_stats
            full_model_results["GAM feature significance"] = gam_reg_featSign
            full_model_results["GAM variable importance"] = gam_reg_varImp
            full_model_results["GAM fitted"] = Y_pred
            full_model_results["GAM partial dependence"] = gam_pd
            full_model_results["GAM partial dependence min/max"] = gam_pd_min_max
            full_model_results["GAM ROC curve"] = roc_curve(Y_data, Y_pred)
            if data_new.empty == False:
                full_model_results["GAM prediction"] = Y_pred_new

            # Model comparison for GAM
            model_comparison_thresInd.loc["AUC ROC"]["Generalized Additive Models"] = roc_auc_score(Y_data, Y_pred)
            model_comparison_thresInd.loc["AP"]["Generalized Additive Models"] = average_precision_score(Y_data, Y_pred)
            precision, recall, thresholds = precision_recall_curve(Y_data, Y_pred)
            model_comparison_thresInd.loc["AUC PRC"]["Generalized Additive Models"] =  auc(recall, precision)
            model_comparison_thresInd.loc["LOG-LOSS"]["Generalized Additive Models"] =  log_loss(Y_data, Y_pred)
            
            # Threshold according to Youden's index
            FPR, TPR, thresholds = roc_curve(Y_data, Y_pred)
            thres_index = np.argmax(TPR - FPR)
            thres = thresholds[thres_index]
            model_comparison_thres.loc["threshold"]["Generalized Additive Models"] = thres

            # Threshold determination (minimizing abs. distance between SENS & SPEC)
            # FPR, TPR, thresholds = roc_curve(Y_test, Y_test_pred)
            # thres_index = np.argmin(abs(TPR - (1-FPR)))
            # thres = thresholds[thres_index]

            Y_pred_bin = np.array([1 if x >= thres else 0 for x in Y_pred])
            full_model_results["GAM fitted binary"] = Y_pred_bin
            if data_new.empty == False:
                Y_pred_new_bin = np.array([1 if x >= thres else 0 for x in Y_pred_new])
                full_model_results["GAM prediction binary"] = Y_pred_new_bin
            TN, FP, FN, TP = confusion_matrix(Y_data, Y_pred_bin).ravel()
            model_comparison_thresDep.loc["TPR"]["Generalized Additive Models"] =  TP/(TP+FN)
            model_comparison_thresDep.loc["FNR"]["Generalized Additive Models"] = FN/(TP+FN)
            model_comparison_thresDep.loc["TNR"]["Generalized Additive Models"] = TN/(TN+FP)
            model_comparison_thresDep.loc["FPR"]["Generalized Additive Models"] = FP/(TN+FP)
            model_comparison_thresDep.loc["TSS"]["Generalized Additive Models"] = (TP/(TP+FN))+(TN/(TN+FP))-1
            model_comparison_thresDep.loc["PREC"]["Generalized Additive Models"] = TP/(TP+FP)
            model_comparison_thresDep.loc["F1"]["Generalized Additive Models"] = f1_score(Y_data, Y_pred_bin)
            model_comparison_thresDep.loc["KAPPA"]["Generalized Additive Models"] = cohen_kappa_score(Y_data, Y_pred_bin)
            model_comparison_thresDep.loc["ACC"]["Generalized Additive Models"] = accuracy_score(Y_data, Y_pred_bin)
            model_comparison_thresDep.loc["BAL ACC"]["Generalized Additive Models"] = balanced_accuracy_score(Y_data, Y_pred_bin)

            progress2 += 1
            my_bar.progress(progress2/len(algorithms))
        
        # Random Forest full model
        if any(a for a in algorithms if a == "Random Forest"):

            # Save results
            rf_reg_inf = pd.DataFrame(index = ["Base estimator", "Estimators", "Features", "OOB score"], columns = ["Value"])
            rf_reg_featImp = pd.DataFrame(index = expl_var, columns = ["Value"])
            rf_reg_varImp = pd.DataFrame(index = expl_var, columns = ["mean", "std"])
            
            # Train RF model
            rf_final_para = final_hyPara_values["rf"]
            params = {'n_estimators': rf_final_para["number of trees"][0],
            'max_depth': rf_final_para["maximum tree depth"][0],
            'max_features': rf_final_para["maximum number of features"][0],
            'max_samples': rf_final_para["sample rate"][0],
            'bootstrap': True,
            'oob_score': True,
            }
            full_model_rf_sk = ensemble.RandomForestClassifier(**params)
            full_model_rf_sk.fit(X_data, Y_data)
            Y_pred = full_model_rf_sk.predict_proba(X_data)
            if data_new.empty == False:
                Y_pred_new = full_model_rf_sk.predict_proba(X_data_new)

            # Extract essential results from model
            # Information
            rf_reg_inf.loc["Base estimator"] = full_model_rf_sk.base_estimator_
            rf_reg_inf.loc["Estimators"] = len(full_model_rf_sk.estimators_)
            rf_reg_inf.loc["Features"] = full_model_rf_sk.n_features_in_
            rf_reg_inf.loc["OOB score"] = full_model_rf_sk.oob_score_
            # Feature importances (RF method)
            rf_featImp = full_model_rf_sk.feature_importances_
            for varI in expl_var:
                rf_reg_featImp.loc[varI]["Value"] = rf_featImp[expl_var.index(varI)]
            rf_reg_featImp = rf_reg_featImp.sort_values(by = ["Value"], ascending = False)
            # Variable importance (via permutation, order important)
            scoring_function = make_scorer(roc_auc_score, greater_is_better = True)
            rf_varImp = permutation_importance(full_model_rf_sk , X_data, Y_data, n_repeats = 10, random_state = 0, scoring = scoring_function)
            for varI in expl_var:
                rf_reg_varImp.loc[varI]["mean"] = rf_varImp.importances_mean[expl_var.index(varI)]
                rf_reg_varImp.loc[varI]["std"] = rf_varImp.importances_std[expl_var.index(varI)]
            rf_reg_varImp = rf_reg_varImp.sort_values(by = ["mean"], ascending = False)
            # Partial dependence
            rf_pd = {}
            rf_pd_min_max = pd.DataFrame(index = expl_var, columns = ["min", "max"])
            #rf_partDep = plot_partial_dependence(full_model_rf_sk, X = X_data, features = expl_var, percentiles =(0, 1), method = "brute").pd_results
            rf_partDep = PartialDependenceDisplay.from_estimator(full_model_rf_sk, X = X_data, features = expl_var, percentiles =(0, 1), method = "brute").pd_results

            
            for varPd in expl_var:
                rf_pd[varPd] = rf_partDep[expl_var.index(varPd)]
                rf_pd_min_max.loc[varPd]["min"] = rf_partDep[expl_var.index(varPd)]["average"].min()
                #rf_pd_min_max.loc[varPd]["min"] = rf_partDep[expl_var.index(varPd)]["average"][0].min()
                rf_pd_min_max.loc[varPd]["max"] = rf_partDep[expl_var.index(varPd)]["average"].max()   
                #rf_pd_min_max.loc[varPd]["max"] = rf_partDep[expl_var.index(varPd)][0].max()                    
            
            # Save tables
            full_model_results["RF information"] = rf_reg_inf
            full_model_results["RF variable importance"] = rf_reg_varImp
            full_model_results["RF feature importance"] = rf_reg_featImp
            full_model_results["RF fitted"] = Y_pred
            full_model_results["RF partial dependence"] = rf_pd
            full_model_results["RF partial dependence min/max"] = rf_pd_min_max
            full_model_results["RF ROC curve"] = roc_curve(Y_data, Y_pred[:,1])
            if data_new.empty == False:
                full_model_results["RF prediction"] = Y_pred_new

            # Model comparison for RF
            model_comparison_thresInd.loc["AUC ROC"]["Random Forest"] = roc_auc_score(Y_data, Y_pred[:, 1])
            model_comparison_thresInd.loc["AP"]["Random Forest"] = average_precision_score(Y_data, Y_pred[:, 1])
            precision, recall, thresholds = precision_recall_curve(Y_data, Y_pred[:, 1])
            model_comparison_thresInd.loc["AUC PRC"]["Random Forest"] =  auc(recall, precision)
            model_comparison_thresInd.loc["LOG-LOSS"]["Random Forest"] =  log_loss(Y_data, Y_pred)
            
            # Threshold according to Youden's index
            FPR, TPR, thresholds = roc_curve(Y_data, Y_pred[:, 1])
            thres_index = np.argmax(TPR - FPR)
            thres = thresholds[thres_index]
            model_comparison_thres.loc["threshold"]["Random Forest"] = thres

            # Threshold determination (minimizing abs. distance between SENS & SPEC)
            # FPR, TPR, thresholds = roc_curve(Y_test, Y_test_pred[:, 1])
            # thres_index = np.argmin(abs(TPR - (1-FPR)))
            # thres = thresholds[thres_index]

            Y_pred_bin = np.array([1 if x >= thres else 0 for x in Y_pred[:, 1]])
            full_model_results["RF fitted binary"] = Y_pred_bin
            if data_new.empty == False:
                Y_pred_new_bin = np.array([1 if x >= thres else 0 for x in Y_pred_new[:, 1]])
                full_model_results["RF prediction binary"] = Y_pred_new_bin
            TN, FP, FN, TP = confusion_matrix(Y_data, Y_pred_bin).ravel()
            model_comparison_thresDep.loc["TPR"]["Random Forest"] =  TP/(TP+FN)
            model_comparison_thresDep.loc["FNR"]["Random Forest"] = FN/(TP+FN)
            model_comparison_thresDep.loc["TNR"]["Random Forest"] = TN/(TN+FP)
            model_comparison_thresDep.loc["FPR"]["Random Forest"] = FP/(TN+FP)
            model_comparison_thresDep.loc["TSS"]["Random Forest"] = (TP/(TP+FN))+(TN/(TN+FP))-1
            model_comparison_thresDep.loc["PREC"]["Random Forest"] = TP/(TP+FP)
            model_comparison_thresDep.loc["F1"]["Random Forest"] = f1_score(Y_data, Y_pred_bin)
            model_comparison_thresDep.loc["KAPPA"]["Random Forest"] = cohen_kappa_score(Y_data, Y_pred_bin)
            model_comparison_thresDep.loc["ACC"]["Random Forest"] = accuracy_score(Y_data, Y_pred_bin)
            model_comparison_thresDep.loc["BAL ACC"]["Random Forest"] = balanced_accuracy_score(Y_data, Y_pred_bin)

            progress2 += 1
            my_bar.progress(progress2/len(algorithms))

        # Boosted Regression Trees full model
        if any(a for a in algorithms if a == "Boosted Regression Trees"):

            # Save results
            brt_reg_inf = pd.DataFrame(index = ["Classes", "Estimators", "Features", "Loss function"], columns = ["Value"])
            brt_reg_featImp = pd.DataFrame(index = expl_var, columns = ["Value"])
            brt_reg_varImp = pd.DataFrame(index = expl_var, columns = ["mean", "std"])
            
            # Train BRT model
            brt_final_para = final_hyPara_values["brt"]
            params = {'n_estimators': brt_final_para["number of trees"][0],
            'learning_rate': brt_final_para["learning rate"][0],
            'max_depth': brt_final_para["maximum tree depth"][0],
            'subsample': brt_final_para["sample rate"][0]
            }
            full_model_brt_sk = ensemble.GradientBoostingClassifier(**params)
            full_model_brt_sk.fit(X_data, Y_data.values.ravel())
            Y_pred = full_model_brt_sk.predict_proba(X_data)
            if data_new.empty == False:
                Y_pred_new = full_model_brt_sk.predict_proba(X_data_new)

            # Extract essential results from model
            # Information
            brt_reg_inf.loc["Classes"] = full_model_brt_sk.n_classes_
            brt_reg_inf.loc["Estimators"] = full_model_brt_sk.n_estimators_
            brt_reg_inf.loc["Features"] = full_model_brt_sk.n_features_in_
            brt_reg_inf.loc["Loss function"] = full_model_brt_sk.loss
            # Feature importances (BRT method)
            brt_featImp = full_model_brt_sk.feature_importances_
            for varI in expl_var:
                brt_reg_featImp.loc[varI]["Value"] = brt_featImp[expl_var.index(varI)]
            brt_reg_featImp = brt_reg_featImp.sort_values(by = ["Value"], ascending = False)
            # Variable importance (via permutation, order important)
            scoring_function = make_scorer(roc_auc_score, greater_is_better = True)
            brt_varImp = permutation_importance(full_model_brt_sk , X_data, Y_data.values.ravel(), n_repeats = 10, random_state = 0, scoring = scoring_function)
            for varI in expl_var:
                brt_reg_varImp.loc[varI]["mean"] = brt_varImp.importances_mean[expl_var.index(varI)]
                brt_reg_varImp.loc[varI]["std"] = brt_varImp.importances_std[expl_var.index(varI)]
            brt_reg_varImp = brt_reg_varImp.sort_values(by = ["mean"], ascending = False)
            # Partial probabilities
            brt_partial_probs = {}
            for varPP in expl_var:
                PP_model_brt = ensemble.GradientBoostingClassifier(**params)
                PP_X_data = pd.DataFrame(X_data[varPP])
                PP_model_brt.fit(PP_X_data, Y_data.values.ravel())
                PP_pred = pd.DataFrame()
                PP_pred["prediction"] = PP_model_brt.predict_proba(PP_X_data)[:,1]
                PP_pred[varPP] = X_data[varPP]
                brt_partial_probs[varPP] = PP_pred
            # Partial dependence
            brt_pd = {}
            brt_pd_min_max = pd.DataFrame(index = expl_var, columns = ["min", "max"])
            #brt_partDep = plot_partial_dependence(full_model_brt_sk, X = X_data, features = expl_var, percentiles = (0, 1), method = "brute", response_method = "predict_proba").pd_results
            brt_partDep = PartialDependenceDisplay.from_estimator(full_model_brt_sk, X = X_data, features = expl_var, percentiles = (0, 1), method = "brute", response_method = "predict_proba").pd_results

            for varPd in expl_var:
                brt_pd[varPd] = brt_partDep[expl_var.index(varPd)]
                brt_pd_min_max.loc[varPd]["min"] = brt_partDep[expl_var.index(varPd)]["average"].min()
                brt_pd_min_max.loc[varPd]["max"] = brt_partDep[expl_var.index(varPd)]["average"].max()  
                #brt_pd_min_max.loc[varPd]["min"] = brt_partDep[expl_var.index(varPd)][0].min()
                #brt_pd_min_max.loc[varPd]["max"] = brt_partDep[expl_var.index(varPd)][0].max()              
            
            # Save tables
            full_model_results["BRT information"] = brt_reg_inf
            full_model_results["BRT variable importance"] = brt_reg_varImp
            full_model_results["BRT feature importance"] = brt_reg_featImp
            full_model_results["BRT fitted"] = Y_pred
            full_model_results["BRT partial dependence"] = brt_pd
            full_model_results["BRT partial dependence min/max"] = brt_pd_min_max
            full_model_results["BRT train score"] = full_model_brt_sk.train_score_
            full_model_results["BRT ROC curve"] = roc_curve(Y_data, Y_pred[:,1])
            full_model_results["BRT partial probabilities"] = brt_partial_probs
            if data_new.empty == False:
                full_model_results["BRT prediction"] = Y_pred_new

            # Model comparison for BRT
            model_comparison_thresInd.loc["AUC ROC"]["Boosted Regression Trees"] = roc_auc_score(Y_data, Y_pred[:, 1])
            model_comparison_thresInd.loc["AP"]["Boosted Regression Trees"] = average_precision_score(Y_data, Y_pred[:, 1])
            precision, recall, thresholds = precision_recall_curve(Y_data, Y_pred[:, 1])
            model_comparison_thresInd.loc["AUC PRC"]["Boosted Regression Trees"] =  auc(recall, precision)
            model_comparison_thresInd.loc["LOG-LOSS"]["Boosted Regression Trees"] =  log_loss(Y_data, Y_pred)
            
            # Threshold according to Youden's index
            FPR, TPR, thresholds = roc_curve(Y_data, Y_pred[:, 1])
            thres_index = np.argmax(TPR - FPR)
            thres = thresholds[thres_index]
            model_comparison_thres.loc["threshold"]["Boosted Regression Trees"] = thres

            # Threshold determination (minimizing abs. distance between SENS & SPEC)
            # FPR, TPR, thresholds = roc_curve(Y_test, Y_test_pred[:, 1])
            # thres_index = np.argmin(abs(TPR - (1-FPR)))
            # thres = thresholds[thres_index]

            Y_pred_bin = np.array([1 if x >= thres else 0 for x in Y_pred[:, 1]])
            full_model_results["BRT fitted binary"] = Y_pred_bin
            if data_new.empty == False:
                Y_pred_new_bin = np.array([1 if x >= thres else 0 for x in Y_pred_new[:, 1]])
                full_model_results["BRT prediction binary"] = Y_pred_new_bin
            TN, FP, FN, TP = confusion_matrix(Y_data, Y_pred_bin).ravel()
            model_comparison_thresDep.loc["TPR"]["Boosted Regression Trees"] =  TP/(TP+FN)
            model_comparison_thresDep.loc["FNR"]["Boosted Regression Trees"] = FN/(TP+FN)
            model_comparison_thresDep.loc["TNR"]["Boosted Regression Trees"] = TN/(TN+FP)
            model_comparison_thresDep.loc["FPR"]["Boosted Regression Trees"] = FP/(TN+FP)
            model_comparison_thresDep.loc["TSS"]["Boosted Regression Trees"] = (TP/(TP+FN))+(TN/(TN+FP))-1
            model_comparison_thresDep.loc["PREC"]["Boosted Regression Trees"] = TP/(TP+FP)
            model_comparison_thresDep.loc["F1"]["Boosted Regression Trees"] = f1_score(Y_data, Y_pred_bin)
            model_comparison_thresDep.loc["KAPPA"]["Boosted Regression Trees"] = cohen_kappa_score(Y_data, Y_pred_bin)
            model_comparison_thresDep.loc["ACC"]["Boosted Regression Trees"] = accuracy_score(Y_data, Y_pred_bin)
            model_comparison_thresDep.loc["BAL ACC"]["Boosted Regression Trees"] = balanced_accuracy_score(Y_data, Y_pred_bin)

            progress2 += 1
            my_bar.progress(progress2/len(algorithms))

        # Artificial Neural Networks full model
        if any(a for a in algorithms if a == "Artificial Neural Networks"):

            # Save results
            ann_reg_inf = pd.DataFrame(index = ["Outputs", "Layers", "Training samples", "Output activation", "Loss function"], columns = ["Value"])
            ann_reg_varImp = pd.DataFrame(index = expl_var, columns = ["mean", "std"])
            
            # Train ANN model
            scaler = StandardScaler()
            scaler.fit(X_data)
            X_data_ann = scaler.transform(X_data)
            X_data_ann = pd.DataFrame(X_data_ann, index = X_data.index, columns = X_data.columns)
            ann_final_para = final_hyPara_values["ann"]
            params2 = {"solver": ann_final_para["weight optimization solver"][0],
            "max_iter": ann_final_para["maximum number of iterations"][0],
            "activation": ann_final_para["activation function"][0],
            "hidden_layer_sizes": ann_final_para["hidden layer sizes"][0],
            "learning_rate_init": ann_final_para["learning rate"][0],
            #"learning_rate": ann_final_para["learning rate schedule"][0],
            #"momentum": ann_final_para["momentum"][0],
            "alpha": ann_final_para["L² regularization"][0]
            #"epsilon": ann_final_para["epsilon"][0]
            }
            full_model_ann_sk = MLPClassifier(**params2)
            full_model_ann_sk.fit(X_data_ann, Y_data)
            Y_pred = full_model_ann_sk.predict_proba(X_data_ann)
            if data_new.empty == False:
                X_data_new_ann = scaler.transform(X_data_new)
                Y_pred_new = full_model_ann_sk.predict_proba(X_data_new_ann)
            
            # Extract essential results from model
            # Regression information
            ann_reg_inf.loc["Outputs"] = full_model_ann_sk.n_outputs_
            ann_reg_inf.loc["Layers"] = full_model_ann_sk.n_layers_
            ann_reg_inf.loc["Training samples"] = full_model_ann_sk.t_
            ann_reg_inf.loc["Output activation"] = full_model_ann_sk.out_activation_
            ann_reg_inf.loc["Loss function"] = full_model_ann_sk.loss
            # Variable importance (via permutation, order important)
            scoring_function = make_scorer(roc_auc_score, greater_is_better = True)
            ann_varImp = permutation_importance(full_model_ann_sk , X_data_ann, Y_data, n_repeats = 10, random_state = 0, scoring = scoring_function)
            for varI in expl_var:
                ann_reg_varImp.loc[varI]["mean"] = ann_varImp.importances_mean[expl_var.index(varI)]
                ann_reg_varImp.loc[varI]["std"] = ann_varImp.importances_std[expl_var.index(varI)]
            ann_reg_varImp = ann_reg_varImp.sort_values(by = ["mean"], ascending = False)
            
            # Partial probabilities
            ann_partial_probs = {}
            for varPP in expl_var:
                PP_scaler = StandardScaler()
                PP_X_data = pd.DataFrame(X_data[varPP])
                PP_scaler.fit(PP_X_data)
                PP_scaler_X_data_ann = PP_scaler.transform(PP_X_data)
                PP_scaler_X_data_ann = pd.DataFrame(PP_scaler_X_data_ann, index = PP_X_data.index, columns = PP_X_data.columns)
                PP_model_ann = MLPClassifier(**params2)
                PP_model_ann.fit(PP_scaler_X_data_ann, Y_data)
                PP_pred = pd.DataFrame()
                PP_pred["prediction"] = PP_model_ann.predict_proba(PP_X_data)[:,1]
                PP_pred[varPP] = PP_scaler_X_data_ann
                ann_partial_probs[varPP] = PP_pred
            
            # Partial dependence
            ann_pd = {}
            ann_pd_min_max = pd.DataFrame(index = expl_var, columns = ["min", "max"])
            #ann_partDep = plot_partial_dependence(full_model_ann_sk, X = X_data_ann, features = expl_var, percentiles = (0, 1), method = "brute", response_method = "predict_proba").pd_results
            ann_partDep = PartialDependenceDisplay.from_estimator(full_model_ann_sk, X = X_data_ann, features = expl_var, percentiles = (0, 1), method = "brute", response_method = "predict_proba").pd_results

            for varPd in expl_var:
                ann_pd[varPd] = ann_partDep[expl_var.index(varPd)]
                ann_pd_min_max.loc[varPd]["min"] = ann_partDep[expl_var.index(varPd)]["average"].min()
                ann_pd_min_max.loc[varPd]["max"] = ann_partDep[expl_var.index(varPd)]["average"].max()  
                #ann_pd_min_max.loc[varPd]["min"] = ann_partDep[expl_var.index(varPd)][0].min()
                #ann_pd_min_max.loc[varPd]["max"] = ann_partDep[expl_var.index(varPd)][0].max()    

            # Save tables
            full_model_results["ANN information"] = ann_reg_inf
            full_model_results["ANN variable importance"] = ann_reg_varImp
            full_model_results["ANN fitted"] = Y_pred
            full_model_results["ANN partial dependence"] = ann_pd
            full_model_results["ANN partial dependence min/max"] = ann_pd_min_max
            if ann_final_para["weight optimization solver"][0] != "lbfgs":
                full_model_results["ANN loss curve"] = full_model_ann_sk.loss_curve_
                full_model_results["ANN loss"] = full_model_ann_sk.best_loss_
            full_model_results["ANN ROC curve"] = roc_curve(Y_data, Y_pred[:,1])
            full_model_results["ANN partial probabilities"] = ann_partial_probs
            if data_new.empty == False:
                full_model_results["ANN prediction"] = Y_pred_new

            # Model comparison for ANN
            model_comparison_thresInd.loc["AUC ROC"]["Artificial Neural Networks"] = roc_auc_score(Y_data, Y_pred[:, 1])
            model_comparison_thresInd.loc["AP"]["Artificial Neural Networks"] = average_precision_score(Y_data, Y_pred[:, 1])
            precision, recall, thresholds = precision_recall_curve(Y_data, Y_pred[:, 1])
            model_comparison_thresInd.loc["AUC PRC"]["Artificial Neural Networks"] =  auc(recall, precision)
            model_comparison_thresInd.loc["LOG-LOSS"]["Artificial Neural Networks"] =  log_loss(Y_data, Y_pred)
            
            # Threshold according to Youden's index
            FPR, TPR, thresholds = roc_curve(Y_data, Y_pred[:, 1])
            thres_index = np.argmax(TPR - FPR)
            thres = thresholds[thres_index]
            model_comparison_thres.loc["threshold"]["Artificial Neural Networks"] = thres

            # Threshold determination (minimizing abs. distance between SENS & SPEC)
            # FPR, TPR, thresholds = roc_curve(Y_test, Y_test_pred[:, 1])
            # thres_index = np.argmin(abs(TPR - (1-FPR)))
            # thres = thresholds[thres_index]

            Y_pred_bin = np.array([1 if x >= thres else 0 for x in Y_pred[:, 1]])
            full_model_results["ANN fitted binary"] = Y_pred_bin
            if data_new.empty == False:
                Y_pred_new_bin = np.array([1 if x >= thres else 0 for x in Y_pred_new[:, 1]])
                full_model_results["ANN prediction binary"] = Y_pred_new_bin
            TN, FP, FN, TP = confusion_matrix(Y_data, Y_pred_bin).ravel()
            model_comparison_thresDep.loc["TPR"]["Artificial Neural Networks"] =  TP/(TP+FN)
            model_comparison_thresDep.loc["FNR"]["Artificial Neural Networks"] = FN/(TP+FN)
            model_comparison_thresDep.loc["TNR"]["Artificial Neural Networks"] = TN/(TN+FP)
            model_comparison_thresDep.loc["FPR"]["Artificial Neural Networks"] = FP/(TN+FP)
            model_comparison_thresDep.loc["TSS"]["Artificial Neural Networks"] = (TP/(TP+FN))+(TN/(TN+FP))-1
            model_comparison_thresDep.loc["PREC"]["Artificial Neural Networks"] = TP/(TP+FP)
            model_comparison_thresDep.loc["F1"]["Artificial Neural Networks"] = f1_score(Y_data, Y_pred_bin)
            model_comparison_thresDep.loc["KAPPA"]["Artificial Neural Networks"] = cohen_kappa_score(Y_data, Y_pred_bin)
            model_comparison_thresDep.loc["ACC"]["Artificial Neural Networks"] = accuracy_score(Y_data, Y_pred_bin)
            model_comparison_thresDep.loc["BAL ACC"]["Artificial Neural Networks"] = balanced_accuracy_score(Y_data, Y_pred_bin)

            progress2 += 1
            my_bar.progress(progress2/len(algorithms))
            
        # Save model comparison
        full_model_results["model comparison thresInd"] = model_comparison_thresInd
        full_model_results["model comparison thres"] = model_comparison_thres
        full_model_results["model comparison thresDep"] = model_comparison_thresDep
        full_model_results["residuals"] = residuals_comparison

    #-----------------------------------------------------------------------------------------
    # Multi-class response variables

    if response_var_type == "multi-class":

        # Collect performance results across all algorithms
        model_comparison = pd.DataFrame(index = ["ACC", "BAL ACC", "macro avg PREC", "macro avg RECALL", "macro avg F1", "weighted avg PREC", "weighted avg RECALL", "weighted avg F1"], columns = algorithms)
        
        # Random Forest full model
        if any(a for a in algorithms if a == "Random Forest"):

            # Save results
            rf_reg_inf = pd.DataFrame(index = ["Base estimator", "Estimators", "Features", "OOB score"], columns = ["Value"])
            rf_reg_featImp = pd.DataFrame(index = expl_var, columns = ["Value"])
            rf_reg_varImp = pd.DataFrame(index = expl_var, columns = ["mean", "std"])
            rf_class_rep = pd.DataFrame(index = np.unique(Y_data), columns = ["PREC", "RECALL", "F1", "SUPPORT"])

            # Train RF model
            rf_final_para = final_hyPara_values["rf"]
            params = {'n_estimators': rf_final_para["number of trees"][0],
            'max_depth': rf_final_para["maximum tree depth"][0],
            'max_features': rf_final_para["maximum number of features"][0],
            'max_samples': rf_final_para["sample rate"][0],
            'bootstrap': True,
            'oob_score': True,
            }
            full_model_rf_sk = ensemble.RandomForestClassifier(**params)
            full_model_rf_sk.fit(X_data, Y_data)
            Y_pred = full_model_rf_sk.predict(X_data)
            Y_pred_proba = full_model_rf_sk.predict_proba(X_data)
            if data_new.empty == False:
                Y_pred_new = full_model_rf_sk.predict(X_data_new)
                Y_pred_new_proba = full_model_rf_sk.predict_proba(X_data_new)

            # Extract essential results from model
            # Information
            rf_reg_inf.loc["Base estimator"] = full_model_rf_sk.base_estimator_
            rf_reg_inf.loc["Estimators"] = len(full_model_rf_sk.estimators_)
            rf_reg_inf.loc["Features"] = full_model_rf_sk.n_features_in_
            rf_reg_inf.loc["OOB score"] = full_model_rf_sk.oob_score_
            # Feature importances (RF method)
            rf_featImp = full_model_rf_sk.feature_importances_
            for varI in expl_var:
                rf_reg_featImp.loc[varI]["Value"] = rf_featImp[expl_var.index(varI)]
            rf_reg_featImp = rf_reg_featImp.sort_values(by = ["Value"], ascending = False)
            # Variable importance (via permutation, order important)
            scoring_function = make_scorer(accuracy_score, greater_is_better = True)
            rf_varImp = permutation_importance(full_model_rf_sk , X_data, Y_data, n_repeats = 10, random_state = 0, scoring = scoring_function)
            for varI in expl_var:
                rf_reg_varImp.loc[varI]["mean"] = rf_varImp.importances_mean[expl_var.index(varI)]
                rf_reg_varImp.loc[varI]["std"] = rf_varImp.importances_std[expl_var.index(varI)]
            rf_reg_varImp = rf_reg_varImp.sort_values(by = ["mean"], ascending = False)
            # Partial dependence
            # rf_pd = {}
            # rf_pd_min_max = pd.DataFrame(index = expl_var, columns = ["min", "max"])
            # rf_partDep = plot_partial_dependence(full_model_rf_sk, X = X_data, features = expl_var, percentiles =(0, 1), method = "brute").pd_results
            # for varPd in expl_var:
            #     rf_pd[varPd] = rf_partDep[expl_var.index(varPd)]
            #     rf_pd_min_max.loc[varPd]["min"] = rf_partDep[expl_var.index(varPd)][0].min()
            #     rf_pd_min_max.loc[varPd]["max"] = rf_partDep[expl_var.index(varPd)][0].max() 
            # Classification report
            for i in range(len(np.unique(Y_data))):
                rf_class_rep.loc[np.unique(Y_data)[i]]["PREC"] =  classification_report(Y_data, Y_pred, digits = 10, output_dict = True)[str(np.unique(Y_data)[i])]["precision"]
                rf_class_rep.loc[np.unique(Y_data)[i]]["RECALL"] =  classification_report(Y_data, Y_pred, digits = 10, output_dict = True)[str(np.unique(Y_data)[i])]["recall"]
                rf_class_rep.loc[np.unique(Y_data)[i]]["F1"] =  classification_report(Y_data, Y_pred, digits = 10, output_dict = True)[str(np.unique(Y_data)[i])]["f1-score"]
                rf_class_rep.loc[np.unique(Y_data)[i]]["SUPPORT"] =  classification_report(Y_data, Y_pred, digits = 10, output_dict = True)[str(np.unique(Y_data)[i])]["support"]
            macro_avg_values = classification_report(Y_data, Y_pred, digits = 10, output_dict = True)["macro avg"]
            macro_avg = pd.DataFrame(index = ["macro avg"], columns = ["PREC", "RECALL", "F1", "SUPPORT"])
            macro_avg.loc["macro avg"]["PREC"] = macro_avg_values["precision"]
            macro_avg.loc["macro avg"]["RECALL"] = macro_avg_values["recall"]
            macro_avg.loc["macro avg"]["F1"] = macro_avg_values["f1-score"]
            macro_avg.loc["macro avg"]["SUPPORT"] = macro_avg_values["support"]
            weigh_avg_values = classification_report(Y_data, Y_pred, digits = 10, output_dict = True)["weighted avg"]
            weigh_avg = pd.DataFrame(index = ["weighted avg"], columns = ["PREC", "RECALL", "F1", "SUPPORT"])
            weigh_avg.loc["weighted avg"]["PREC"] = weigh_avg_values["precision"]
            weigh_avg.loc["weighted avg"]["RECALL"] = weigh_avg_values["recall"]
            weigh_avg.loc["weighted avg"]["F1"] = weigh_avg_values["f1-score"]
            weigh_avg.loc["weighted avg"]["SUPPORT"] = weigh_avg_values["support"]
            rf_class_rep = pd.concat([rf_class_rep,macro_avg])
            #rf_class_rep = rf_class_rep.append(macro_avg, ignore_index=False)
            rf_class_rep = pd.concat([rf_class_rep,weigh_avg]) 
            #rf_class_rep = rf_class_rep.append(weigh_avg, ignore_index=False)                     
            
            # Save tables
            full_model_results["RF information"] = rf_reg_inf
            full_model_results["RF variable importance"] = rf_reg_varImp
            full_model_results["RF feature importance"] = rf_reg_featImp
            full_model_results["RF fitted"] = Y_pred
            full_model_results["RF fitted proba"] = Y_pred_proba
            full_model_results["RF confusion"] = pd.DataFrame(confusion_matrix(Y_data, Y_pred, labels = np.unique(Y_data)), index=np.unique(Y_data), columns=np.unique(Y_data))
            full_model_results["RF classification report"] = rf_class_rep
            #full_model_results["RF partial dependence"] = rf_pd
            #full_model_results["RF partial dependence min/max"] = rf_pd_min_max
            if data_new.empty == False:
                full_model_results["RF prediction"] = Y_pred_new
                full_model_results["RF prediction proba"] = Y_pred_new_proba

            # Model comparison for RF
            model_comparison.loc["ACC"]["Random Forest"] = accuracy_score(Y_data, Y_pred)
            model_comparison.loc["BAL ACC"]["Random Forest"] = balanced_accuracy_score(Y_data, Y_pred)
            model_comparison.loc["macro avg PREC"]["Random Forest"] = rf_class_rep.loc["macro avg"]["PREC"]
            model_comparison.loc["macro avg RECALL"]["Random Forest"] = rf_class_rep.loc["macro avg"]["RECALL"]
            model_comparison.loc["macro avg F1"]["Random Forest"] = rf_class_rep.loc["macro avg"]["F1"]
            model_comparison.loc["weighted avg PREC"]["Random Forest"] = rf_class_rep.loc["weighted avg"]["PREC"]
            model_comparison.loc["weighted avg RECALL"]["Random Forest"] = rf_class_rep.loc["weighted avg"]["RECALL"]
            model_comparison.loc["weighted avg F1"]["Random Forest"] = rf_class_rep.loc["weighted avg"]["F1"]

            progress2 += 1
            my_bar.progress(progress2/len(algorithms))

        # Artificial Neural Networks full model
        if any(a for a in algorithms if a == "Artificial Neural Networks"):

            # Save results
            ann_reg_inf = pd.DataFrame(index = ["Outputs", "Layers", "Training samples", "Output activation", "Loss function"], columns = ["Value"])
            ann_reg_varImp = pd.DataFrame(index = expl_var, columns = ["mean", "std"])
            ann_class_rep = pd.DataFrame(index = np.unique(Y_data), columns = ["PREC", "RECALL", "F1", "SUPPORT"])
            
            # Train ANN model
            scaler = StandardScaler()
            scaler.fit(X_data)
            X_data_ann = scaler.transform(X_data)
            X_data_ann = pd.DataFrame(X_data_ann, index = X_data.index, columns = X_data.columns)
            ann_final_para = final_hyPara_values["ann"]
            params2 = {"solver": ann_final_para["weight optimization solver"][0],
            "max_iter": ann_final_para["maximum number of iterations"][0],
            "activation": ann_final_para["activation function"][0],
            "hidden_layer_sizes": ann_final_para["hidden layer sizes"][0],
            "learning_rate_init": ann_final_para["learning rate"][0],
            #"learning_rate": ann_final_para["learning rate schedule"][0],
            #"momentum": ann_final_para["momentum"][0],
            "alpha": ann_final_para["L² regularization"][0]
            #"epsilon": ann_final_para["epsilon"][0]
            }
            full_model_ann_sk = MLPClassifier(**params2)
            full_model_ann_sk.fit(X_data_ann, Y_data)
            Y_pred = full_model_ann_sk.predict(X_data_ann)
            Y_pred_proba = full_model_ann_sk.predict_proba(X_data_ann)
            if data_new.empty == False:
                X_data_new_ann = scaler.transform(X_data_new)
                Y_pred_new = full_model_ann_sk.predict(X_data_new_ann)
                Y_pred_new_proba = full_model_ann_sk.predict_proba(X_data_new_ann)
            
            # Extract essential results from model
            # Regression information
            ann_reg_inf.loc["Outputs"] = full_model_ann_sk.n_outputs_
            ann_reg_inf.loc["Layers"] = full_model_ann_sk.n_layers_
            ann_reg_inf.loc["Training samples"] = full_model_ann_sk.t_
            ann_reg_inf.loc["Output activation"] = full_model_ann_sk.out_activation_
            ann_reg_inf.loc["Loss function"] = full_model_ann_sk.loss
            
            # Variable importance (via permutation, order important)
            scoring_function = make_scorer(accuracy_score, greater_is_better = True)
            ann_varImp = permutation_importance(full_model_ann_sk , X_data_ann, Y_data, n_repeats = 10, random_state = 0, scoring = scoring_function)
            for varI in expl_var:
                ann_reg_varImp.loc[varI]["mean"] = ann_varImp.importances_mean[expl_var.index(varI)]
                ann_reg_varImp.loc[varI]["std"] = ann_varImp.importances_std[expl_var.index(varI)]
            ann_reg_varImp = ann_reg_varImp.sort_values(by = ["mean"], ascending = False)
            # Partial dependence
            # ann_pd = {}
            # ann_pd_min_max = pd.DataFrame(index = expl_var, columns = ["min", "max"])
            # ann_partDep = plot_partial_dependence(full_model_ann_sk, X = X_data_ann, features = expl_var, percentiles = (0, 1), method = "brute", response_method = "predict_proba").pd_results
            # for varPd in expl_var:
            #     ann_pd[varPd] = ann_partDep[expl_var.index(varPd)]
            #     ann_pd_min_max.loc[varPd]["min"] = ann_partDep[expl_var.index(varPd)][0].min()
            #     ann_pd_min_max.loc[varPd]["max"] = ann_partDep[expl_var.index(varPd)][0].max() 
            # Classification report
            for i in range(len(np.unique(Y_data))):
                ann_class_rep.loc[np.unique(Y_data)[i]]["PREC"] =  classification_report(Y_data, Y_pred, digits = 10, output_dict = True)[str(np.unique(Y_data)[i])]["precision"]
                ann_class_rep.loc[np.unique(Y_data)[i]]["RECALL"] =  classification_report(Y_data, Y_pred, digits = 10, output_dict = True)[str(np.unique(Y_data)[i])]["recall"]
                ann_class_rep.loc[np.unique(Y_data)[i]]["F1"] =  classification_report(Y_data, Y_pred, digits = 10, output_dict = True)[str(np.unique(Y_data)[i])]["f1-score"]
                ann_class_rep.loc[np.unique(Y_data)[i]]["SUPPORT"] =  classification_report(Y_data, Y_pred, digits = 10, output_dict = True)[str(np.unique(Y_data)[i])]["support"]
            macro_avg_values = classification_report(Y_data, Y_pred, digits = 10, output_dict = True)["macro avg"]
            macro_avg = pd.DataFrame(index = ["macro avg"], columns = ["PREC", "RECALL", "F1", "SUPPORT"])
            macro_avg.loc["macro avg"]["PREC"] = macro_avg_values["precision"]
            macro_avg.loc["macro avg"]["RECALL"] = macro_avg_values["recall"]
            macro_avg.loc["macro avg"]["F1"] = macro_avg_values["f1-score"]
            macro_avg.loc["macro avg"]["SUPPORT"] = macro_avg_values["support"]
            weigh_avg_values = classification_report(Y_data, Y_pred, digits = 10, output_dict = True)["weighted avg"]
            weigh_avg = pd.DataFrame(index = ["weighted avg"], columns = ["PREC", "RECALL", "F1", "SUPPORT"])
            weigh_avg.loc["weighted avg"]["PREC"] = weigh_avg_values["precision"]
            weigh_avg.loc["weighted avg"]["RECALL"] = weigh_avg_values["recall"]
            weigh_avg.loc["weighted avg"]["F1"] = weigh_avg_values["f1-score"]
            weigh_avg.loc["weighted avg"]["SUPPORT"] = weigh_avg_values["support"]
            #ann_class_rep = ann_class_rep.append(macro_avg, ignore_index=False)
            #ann_class_rep = ann_class_rep.append(weigh_avg, ignore_index=False)
            ann_class_rep = pd.concat([ann_class_rep,macro_avg])
            ann_class_rep = pd.concat([ann_class_rep,weigh_avg])

            # Save tables
            full_model_results["ANN information"] = ann_reg_inf
            full_model_results["ANN variable importance"] = ann_reg_varImp
            full_model_results["ANN fitted"] = Y_pred
            full_model_results["ANN fitted proba"] = Y_pred_proba
            full_model_results["ANN confusion"] = pd.DataFrame(confusion_matrix(Y_data, Y_pred, labels = np.unique(Y_data)), index=np.unique(Y_data), columns=np.unique(Y_data))
            full_model_results["ANN classification report"] = ann_class_rep
            # full_model_results["ANN partial dependence"] = ann_pd
            # full_model_results["ANN partial dependence min/max"] = ann_pd_min_max
            if ann_final_para["weight optimization solver"][0] != "lbfgs":
                full_model_results["ANN loss curve"] = full_model_ann_sk.loss_curve_
                full_model_results["ANN loss"] = full_model_ann_sk.best_loss_
            if data_new.empty == False:
                full_model_results["ANN prediction"] = Y_pred_new
                full_model_results["ANN prediction proba"] = Y_pred_new_proba

            # Model comparison for ANN
            model_comparison.loc["ACC"]["Artificial Neural Networks"] = accuracy_score(Y_data, Y_pred)
            model_comparison.loc["BAL ACC"]["Artificial Neural Networks"] = balanced_accuracy_score(Y_data, Y_pred)
            model_comparison.loc["macro avg PREC"]["Artificial Neural Networks"] = ann_class_rep.loc["macro avg"]["PREC"]
            model_comparison.loc["macro avg RECALL"]["Artificial Neural Networks"] = ann_class_rep.loc["macro avg"]["RECALL"]
            model_comparison.loc["macro avg F1"]["Artificial Neural Networks"] = ann_class_rep.loc["macro avg"]["F1"]
            model_comparison.loc["weighted avg PREC"]["Artificial Neural Networks"] = ann_class_rep.loc["weighted avg"]["PREC"]
            model_comparison.loc["weighted avg RECALL"]["Artificial Neural Networks"] = ann_class_rep.loc["weighted avg"]["RECALL"]
            model_comparison.loc["weighted avg F1"]["Artificial Neural Networks"] = ann_class_rep.loc["weighted avg"]["F1"]

            progress2 += 1
            my_bar.progress(progress2/len(algorithms))
            
        # Save model comparison
        full_model_results["model comparison"] = model_comparison
    
    if 'full_model_ann_sk' in locals():
        return full_model_results, full_model_ann_sk
    else:
        return full_model_results 

#---------------------------------------------------------------
# WEIGHT MATRIX FOR ARTIFICIAL NEURAL NETWORKS
#---------------------------------------------------------------
def weight_matrix_func(output, expl_var, wei_matrix, coef_list):
    # 1 hidden layer
    if len(coef_list) == 2:
        df_in_hi = pd.DataFrame(coef_list[0])
        df_in_hi.index = pd.MultiIndex.from_product([["Input Layer"], expl_var])
        df_in_hi.columns = pd.MultiIndex.from_product([["Hidden Layer 1"], list(range(1, df_in_hi.shape[1]+1))])
        df_hi_out = pd.DataFrame(coef_list[1])
        df_hi_out.index = pd.MultiIndex.from_product([["Hidden Layer 1"], list(range(1, df_hi_out.shape[0]+1))])
        df_hi_out.columns = pd.MultiIndex.from_product([["Output Layer"], output])
        if wei_matrix == "Input Layer <-> Hidden Layer 1":
            if df_in_hi.shape[0] < df_in_hi.shape[1]:
                df_in_hi = df_in_hi.transpose()
            return df_in_hi
        if wei_matrix == "Hidden Layer 1 <-> Output Layer":
            if df_hi_out.shape[0] < df_hi_out.shape[1]:
                df_hi_out = df_hi_out.transpose()
            return df_hi_out
    # 2 hidden layer
    elif len(coef_list) == 3:
        df_in_hi = pd.DataFrame(coef_list[0])
        df_in_hi.index = pd.MultiIndex.from_product([["Input Layer"], expl_var])
        df_in_hi.columns = pd.MultiIndex.from_product([["Hidden Layer 1"], list(range(1, df_in_hi.shape[1]+1))])
        df_hi1_hi2 = pd.DataFrame(coef_list[1])
        df_hi1_hi2.index = pd.MultiIndex.from_product([["Hidden Layer 1"], list(range(1, df_hi1_hi2.shape[0]+1))])
        df_hi1_hi2.columns = pd.MultiIndex.from_product([["Hidden Layer 2"], list(range(1, df_hi1_hi2.shape[1]+1))])
        df_hi_out = pd.DataFrame(coef_list[2])
        df_hi_out.index = pd.MultiIndex.from_product([["Hidden Layer 2"], list(range(1, df_hi_out.shape[0]+1))])
        df_hi_out.columns = pd.MultiIndex.from_product([["Output Layer"], output])
        if wei_matrix == "Input Layer <-> Hidden Layer 1":
            if df_in_hi.shape[0] < df_in_hi.shape[1]:
                df_in_hi = df_in_hi.transpose()
            return df_in_hi
        elif wei_matrix == "Hidden Layer 1 <-> Hidden Layer 2":
            if df_hi1_hi2.shape[0] < df_hi1_hi2.shape[1]:
                df_hi1_hi2 = df_hi1_hi2.transpose()
            return df_hi1_hi2
        elif wei_matrix == "Hidden Layer 2 <-> Output Layer":
            if df_hi_out.shape[0] < df_hi_out.shape[1]:
                df_hi_out = df_hi_out.transpose()
            return df_hi_out
    # 3 hidden layer
    elif len(coef_list) == 4:
        df_in_hi = pd.DataFrame(coef_list[0])
        df_in_hi.index = pd.MultiIndex.from_product([["Input Layer"], expl_var])
        df_in_hi.columns = pd.MultiIndex.from_product([["Hidden Layer 1"], list(range(1, df_in_hi.shape[1]+1))])
        df_hi1_hi2 = pd.DataFrame(coef_list[1])
        df_hi1_hi2.index = pd.MultiIndex.from_product([["Hidden Layer 1"], list(range(1, df_hi1_hi2.shape[0]+1))])
        df_hi1_hi2.columns = pd.MultiIndex.from_product([["Hidden Layer 2"], list(range(1, df_hi1_hi2.shape[1]+1))])
        df_hi2_hi3 = pd.DataFrame(coef_list[2])
        df_hi2_hi3.index = pd.MultiIndex.from_product([["Hidden Layer 2"], list(range(1, df_hi2_hi3.shape[0]+1))])
        df_hi2_hi3.columns = pd.MultiIndex.from_product([["Hidden Layer 3"], list(range(1, df_hi2_hi3.shape[1]+1))])
        df_hi_out = pd.DataFrame(coef_list[3])
        df_hi_out.index = pd.MultiIndex.from_product([["Hidden Layer 3"], list(range(1, df_hi_out.shape[0]+1))])
        df_hi_out.columns = pd.MultiIndex.from_product([["Output Layer"], output])
        if wei_matrix == "Input Layer <-> Hidden Layer 1":
            if df_in_hi.shape[0] < df_in_hi.shape[1]:
                df_in_hi = df_in_hi.transpose()
            return df_in_hi
        elif wei_matrix == "Hidden Layer 1 <-> Hidden Layer 2":
            if df_hi1_hi2.shape[0] < df_hi1_hi2.shape[1]:
                df_hi1_hi2 = df_hi1_hi2.transpose()
            return df_hi1_hi2
        elif wei_matrix == "Hidden Layer 2 <-> Hidden Layer 3":
            if df_hi2_hi3.shape[0] < df_hi2_hi3.shape[1]:
                df_hi2_hi3 = df_hi2_hi3.transpose()
            return df_hi2_hi3
        elif wei_matrix == "Hidden Layer 3 <-> Output Layer":
            if df_hi_out.shape[0] < df_hi_out.shape[1]:
                df_hi_out = df_hi_out.transpose()
            return df_hi_out

#---------------------------------------------------------------
# TIME SERIES DECOMPOSITION
#---------------------------------------------------------------
def decompose_plots(ts_decomp,ts_decom_name,df,ts,ts_var,ts_time):  

    decomposition = seasonal_decompose(ts[ts_var])

    if ts_decomp=="detrending":
        ts['trend']=decomposition.trend 
        ts[ts_decom_name]=ts[ts_var]-decomposition.trend
        ts_toplot=decomposition.trend
    elif ts_decomp=="seasonal adjustment":   
        ts['seasonal comp']=decomposition.seasonal 
        ts[ts_decom_name]=ts[ts_var]-decomposition.seasonal
        ts_toplot=decomposition.seasonal
    else:
        ts['trend']=decomposition.trend
        ts['seasonal comp']=decomposition.seasonal
        ts[ts_decom_name]=ts[ts_var]-decomposition.trend-decomposition.seasonal
        

    fig = go.Figure()
    #fig = px.area(x=df[ts_time], y=df[ts_var], color_discrete_sequence=['rgba(55, 126, 184, 0.7)'])
    fig.add_trace(go.Scatter(x=df[ts_time], y=df[ts_var], line=dict(color='royalblue', width=2)))
    if  ts_decom_name=='detrended and seasonally adjusted':
        fig.add_trace(go.Scatter(x=df[ts_time], y=ts['trend'], line=dict(color='firebrick', width=2)))
        fig.add_trace(go.Scatter(x=df[ts_time], y=ts['seasonal comp'], line=dict(color='green', width=2)))
    else:
            fig.add_trace(go.Scatter(x=df[ts_time], y=ts_toplot, line=dict(color='firebrick', width=2)))
    fig.update_layout(showlegend=False)
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',}) 
    fig.update_layout(yaxis=dict(title=ts_var, titlefont_size=12, tickfont_size=14,),)
    fig.update_layout(xaxis=dict(title="", titlefont_size=12, tickfont_size=14,),)
    #fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig,use_container_width=True) 
    
    ts_show_components=st.checkbox('Show time series components?', value=False)
    if ts_show_components:
        st.write(ts.drop(ts_time, axis = 1))
        st.write("")  
    
  
    return ts 

#--------------------------------------------------------
# ACF, PACF calculation and plots
#--------------------------------------------------------
def series_acf_pacf_plot(plot_var_name,plot_data):

    fig, (ax1, ax2, ax3)  = plt.subplots(1, 3, sharex=False)
    fig.set_figheight(2)
    plt.rcParams.update({'font.size': 8})
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.markersize']= 1
    ax1.plot(plot_data)
    ax1.set_title(plot_var_name)
    nlab=len(ax1.get_xticklabels())
    for i, label in enumerate(ax1.get_xticklabels()):
        if i > 1 and i < (nlab-1):
            label.set_visible(False)
    
    
    plot_acf(plot_data, ax=ax2)
    ax2.set_title('ACF')
    plot_pacf(plot_data, ax=ax3)
    ax3.set_title('PACF')

    for k in [ax1,ax2,ax3]:
        k.spines['top'].set_visible(False)
        k.spines['right'].set_visible(False)
        k.spines['bottom'].set_visible(False)
        k.spines['left'].set_visible(False)              
    st.pyplot(fig)
       
    return plot_data 

#--------------------------------------------------------
# Augmented Dickey Fuller Test
#--------------------------------------------------------
def adf_test(adf_var_name,ts,ts_var):
    
    adf_test=pd.DataFrame(index= ['raw series', adf_var_name],columns=['ADF', 'p-value'])
    adf_test['ADF'][0] = adfuller(ts[ts_var])[0]
    adf_test['p-value'][0] = adfuller(ts[ts_var])[1]
    adf_test['ADF'][1] = adfuller(ts[adf_var_name].dropna())[0]
    adf_test['p-value'][1] = adfuller(ts[adf_var_name].dropna())[1]
    st.table(adf_test)
    
    return adf_test

#----------------------------------------------------------
# Time series model evaluation (test dataset vs. observed)
#----------------------------------------------------------

def ts_model_evaluation(Y_test, Y_pred):

    stats_name="Val. measures"
    ts_model_eval_stats = pd.DataFrame(index = ["% VE", "MSE", "RMSE", "MAE", "MaxErr", "EVRS"], columns = [stats_name])
    ts_model_eval_stats[stats_name]["% VE"] = r2_score(Y_test, Y_pred)
    ts_model_eval_stats[stats_name]["MSE"] = mean_squared_error(Y_test, Y_pred, squared = True)
    ts_model_eval_stats[stats_name]["RMSE"] = mean_squared_error(Y_test, Y_pred, squared = False)
    ts_model_eval_stats[stats_name]["MAE"] = mean_absolute_error(Y_test, Y_pred)
    ts_model_eval_stats[stats_name]["MaxErr"] = max_error(Y_test, Y_pred)
    ts_model_eval_stats[stats_name]["EVRS"] = explained_variance_score(Y_test, Y_pred)
    
    return ts_model_eval_stats

