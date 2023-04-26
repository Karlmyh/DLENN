#!usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd
from time import time
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
import random
from sklearn.metrics import mean_squared_error
from FLENN import FLENN
from sklearn.neighbors import KNeighborsRegressor


data_file_dir = "./data/"
repeat_times = 20
portion_vec = [0.4,0.3,0.2,0.1]

portion_vec_list = [
                   [0.8, 0.2], 
                   [0.7, 0.2, 0.1],
                   [0.4, 0.3, 0.2, 0.1],
                   [0.6, 0.3, 0.1],
                   [0.2, 0.17, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05, 0.03]]

data_file_name_seq = [
    'housing_scale.csv',
 'mpg_scale.csv',
 'airfoil.csv',
 'space_ga_scale.csv',
 'whitewine.csv',
 'dakbilgic.csv',
 'mg_scale.csv',
 'bias.csv',
 'cpusmall_scale.csv',                     
 'aquatic.csv',
 'music.csv',
 'redwine.csv',
 'ccpp.csv',
 'concrete.csv',
 'portfolio.csv',
 'building.csv',
 'yacht.csv',
 'abalone.csv',
 'algerian.csv',
 'fish.csv',
 'communities.csv',
 'forestfires.csv',
 'cbm.csv']

log_file = "./results/realdata.csv"


for idx_portion_vec, portion_vec  in enumerate(portion_vec_list):
    if idx_portion_vec < 4:
        continue
    for data_file_name in data_file_name_seq:

        path = os.path.join(data_file_dir, data_file_name)
        data = pd.read_csv(path, header=None)
        data = np.array(data, dtype = "float")
        X = data[:,1:]
        y = data[:,0]
        scalar = MinMaxScaler()
        X = scalar.fit_transform(X)
        y = y[~np.isnan(X).any(axis = 1)]
        X = X[~np.isnan(X).any(axis = 1)]

        dim = X.shape[1]

        for iterate in range(repeat_times):

            np.random.seed(iterate)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = iterate)

            # uniform aggregation
            params = {
                 "portion_vec" : [portion_vec],
                 "weighting_scheme" : ["uniform"],
                 "smooth_order" : [0, 1, 2]}
            cv_regressor = GridSearchCV(estimator = FLENN(), param_grid = params, cv = 10, n_jobs = 30)
            _ = cv_regressor.fit(X_train, y_train)
            model = cv_regressor.best_estimator_

            time_start = time()
            score = mean_squared_error(model.predict(X_test), y_test)
            time_end = time()
            time_test= time_end - time_start
            with open(log_file, 'a') as f:
                logs= "{},{},{:.10f},{},{},{},uniform,{}\n".format(iterate,time_test,
                                                score,X_train.shape[0], X_test.shape[0], data_file_name, idx_portion_vec)
                f.writelines(logs)


            # cauchy aggregation   
            weights_vec = np.array(portion_vec)**(1 / ( dim / 2 +  1))
            weights_vec = weights_vec / weights_vec.sum()
            model.weights  = weights_vec

            time_start = time()
            score = mean_squared_error(model.predict(X_test), y_test)
            time_end = time()
            time_test= time_end - time_start
            with open(log_file, 'a') as f:
                logs= "{},{},{:.10f},{},{},{},cauchy,{}\n".format(iterate,time_test,
                                                score,X_train.shape[0], X_test.shape[0], data_file_name, idx_portion_vec)
                f.writelines(logs)

            # federated aggregation   
            weights_vec = np.array(portion_vec)
            weights_vec = weights_vec / weights_vec.sum()
            model.weights  = weights_vec
            time_start = time()
            score = mean_squared_error(model.predict(X_test), y_test)
            time_end = time()
            time_test= time_end - time_start
            with open(log_file, 'a') as f:
                logs= "{},{},{:.10f},{},{},{},federated,{}\n".format(iterate,time_test,
                                                score,X_train.shape[0], X_test.shape[0], data_file_name,idx_portion_vec)
                f.writelines(logs)


            # extrapolation aggregation
            params = {"lamda" : [0.0001,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5],
                 "portion_vec" : [portion_vec],
                 "weighting_scheme" : ["extrapolated"],
                 "smooth_order" : [0, 1, 2]}
            cv_regressor = GridSearchCV(estimator = FLENN(), param_grid = params, cv = 10, n_jobs = 60)
            _ = cv_regressor.fit(X_train, y_train)
            model = cv_regressor.best_estimator_
            time_start = time()
            score = mean_squared_error(model.predict(X_test), y_test)
            time_end = time()
            time_test= time_end - time_start
            with open(log_file, 'a') as f:
                logs= "{},{},{:.10f},{},{},{},extrapolation,{}\n".format(iterate,time_test,
                                                score,X_train.shape[0], X_test.shape[0], data_file_name,idx_portion_vec)
                f.writelines(logs)


            # knn
            params = {"n_neighbors":[40,60,80,100,120,150,200,250,300,350,400,450]}
            cv_regressor = GridSearchCV(estimator = KNeighborsRegressor(), param_grid = params, cv = 10)
            _ = cv_regressor.fit(X_train, y_train)
            model = cv_regressor.best_estimator_
            time_start = time()
            score = mean_squared_error(model.predict(X_test), y_test)
            time_end = time()
            time_test= time_end - time_start
            with open(log_file, 'a') as f:
                logs= "{},{},{:.10f},{},{},{},knn,{}\n".format(iterate,time_test,
                                                score,X_train.shape[0], X_test.shape[0], data_file_name, idx_portion_vec)
                f.writelines(logs)