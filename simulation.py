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
from DLENN import DLENN
from sklearn.neighbors import KNeighborsRegressor
from distribution import TestDistribution


repeat_times = 20



log_file = "./results/simulation.csv"


n_machine_per_order_vec = [2, 4, 6, 8]
order_vec = [1, 2]
distribution_vec = [1,2,3,4]

for n_machine_per_order in n_machine_per_order_vec:
    for order in order_vec:
        for distribution in distribution_vec:

            sample_generator = TestDistribution( distribution ).returnDistribution()

            

            for iterate in range(repeat_times):
                

                np.random.seed(iterate)
                X, y = sample_generator.generate(30000)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = iterate)

                # uniform aggregation
                params = {
                     "n_machine_per_order" : [n_machine_per_order],
                     "weighting_scheme" : ["uniform"],
                     "order" : [order],
                     "k": [2,5,10,15,20,30,40,50,75,100]}
                cv_regressor = GridSearchCV(estimator = DLENN(), param_grid = params, cv = 5, n_jobs = 10)
                _ = cv_regressor.fit(X_train, y_train)
                model = cv_regressor.best_estimator_

                time_start = time()
                score = mean_squared_error(model.predict(X_test), y_test)
                time_end = time()
                time_test= time_end - time_start
                with open(log_file, 'a') as f:
                    logs= "{},{},{:.10f},{},{},{},uniform,{},{}\n".format(iterate,time_test,
                                                    score,X_train.shape[0], X_test.shape[0], distribution, n_machine_per_order,order)
                    f.writelines(logs)




                # extrapolation aggregation
                params = {
                     "n_machine_per_order" : [n_machine_per_order],
                     "weighting_scheme" : ["extrapolated"],
                     "order" : [order],
                     "lamda": [0.0001,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5],
                     "k": [2,5,10,15,20,30,40,50,75,100]}
                cv_regressor = GridSearchCV(estimator = DLENN(), param_grid = params, cv = 5, n_jobs = 10)
                _ = cv_regressor.fit(X_train, y_train)
                model = cv_regressor.best_estimator_
                time_start = time()
                score = mean_squared_error(model.predict(X_test), y_test)
                time_end = time()
                time_test= time_end - time_start
                with open(log_file, 'a') as f:
                    logs= "{},{},{:.10f},{},{},{},extrapolated,{},{}\n".format(iterate,time_test,
                                                    score,X_train.shape[0], X_test.shape[0], distribution, n_machine_per_order,order)
                    f.writelines(logs)


                # knn
                params = {"n_neighbors":[5,10,15,20,30,40,60,80,100,120,150,200,250,300,350,400,450]}
                cv_regressor = GridSearchCV(estimator = KNeighborsRegressor(), param_grid = params, cv = 5)
                _ = cv_regressor.fit(X_train, y_train)
                model = cv_regressor.best_estimator_
                time_start = time()
                score = mean_squared_error(model.predict(X_test), y_test)
                time_end = time()
                time_test= time_end - time_start
                with open(log_file, 'a') as f:
                    logs= "{},{},{:.10f},{},{},{},knn,{},{}\n".format(iterate,time_test,
                                                    score,X_train.shape[0], X_test.shape[0], distribution, n_machine_per_order,order)
                    f.writelines(logs)