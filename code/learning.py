from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_validate, cross_val_predict, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from structure import *
from config import *
from feature import *

import numpy as np
import time, re, os
from random import shuffle
import pandas as pd

# map feature to get feature function
func_map = {
    "feature_topo_compo": get_feature_topo_compo, 
    "feature_add_s_nobin": get_feature_with_s_nobin, 
    "feature_add_s_nobin_Bar0": get_feature_with_s_nobin_Bar0, 
    "feature_composition": get_feature_composition
}


def learning_cv(data_dir):
    n_kfolds = 10
    crossvalidation = KFold(n_splits=n_kfolds, shuffle=True, random_state=1)
    X, y = get_data_X_y(data_dir, fname)
    print("get data")
    
    # normalization
    min_max_scaler = MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    clf = GradientBoostingRegressor(loss='ls', learning_rate=0.001, n_estimators=300000, max_depth=7, min_samples_split=5, subsample=0.85, max_features='sqrt', random_state = 0)

    # scores = cross_validate(clf, X_minmax, y, scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'], n_jobs=10, cv=crossvalidation)
    predicted = cross_val_predict(clf, X_minmax, y, cv=crossvalidation, n_jobs=10)
    with open(data_dir + '/predict_' + fname + '_cv.npz', 'wb') as out_file:
        np.savez(out_file, predicted=predicted, y=y)
    r2 = r2_score(y, predicted)
    mse = mean_squared_error(y, predicted)
    rmse = np.sqrt(abs(mse))
    mae = mean_absolute_error(y, predicted)

    print(fname + ", cv=10, n_estimator=300000, max_depth=7, min_samples_split=5, subsample=0.85 remove >=2.0 data " + str(cut))

    print("r2_score: {0}, rmse: {1}, mae: {2}".format(r2, rmse, mae))


def learning_cv_repeated(data_dir, fname, times):
    n_kfolds = 10
    crossvalidation = KFold(n_splits=n_kfolds, shuffle=True, random_state=1)
    X, y = get_data_X_y(data_dir, fname)
    print("get data")
    min_max_scaler = MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    for i in range(times):
        clf = GradientBoostingRegressor(loss='ls', learning_rate=0.001, n_estimators=300000, max_depth=7, min_samples_split=5, subsample=0.85, max_features='sqrt', random_state = i)

        # scores = cross_validate(clf, X_minmax, y, scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'], n_jobs=10, cv=crossvalidation)
        predicted = cross_val_predict(clf, X_minmax, y, cv=crossvalidation, n_jobs=10)
        with open(data_dir + '/predict_' + fname + str(i) + '_rep_cv.npz', 'wb') as out_file:
            np.savez(out_file, predicted=predicted, y=y)
        r2 = r2_score(y, predicted)
        mse = mean_squared_error(y, predicted)
        rmse = np.sqrt(abs(mse))
        mae = mean_absolute_error(y, predicted)

        print(fname + ", cv=10, n_estimator=300000, max_depth=7, min_samples_split=5, subsample=0.85 remove >=2.0 data " + str(cut))
        print("times: " + str(i))
        print("r2_score: {0}, rmse: {1}, mae: {2}".format(r2, rmse, mae))

def get_data_X_y(data_dir, fname):
    with open(data_dir + '/properties.txt', 'r') as f:
        lines = f.read().splitlines()
    X = []; y = []
    for line in lines:
        id, delta_e = line.split()[0], line.split()[1]
        with open(data_dir + '/' + fname + '/' + id + '_feature.npy', 'rb') as fe:
            feature = np.load(fe)

        X.append(feature)
        y.append(delta_e)

    X = np.asarray(X)
    y = np.asarray(y, float)
    # print(X.shape, y.shape)
    return X, y


def get_id_list(data_dir):
    with open(data_dir + '/properties.txt', 'r') as f:
        lines = f.read().splitlines()
    id_list = []
    for line in lines[1:]:
        id = line.split()[0]
        id_list.append(id)
    return id_list


def batch_handle(id_list):
    for id in id_list:
        get_prim_structure_info(data_dir, id)
        enlarge_cell(data_dir, id)
        get_betti_num(data_dir, id)
        func_map[fname](data_dir, id)


def split_list(all_id_list):
    id_list_splited = []
    step = math.ceil(len(all_id_list)/10.0)
    for i in range(0, len(all_id_list), step):
        start = i
        end = min(i + step, len(all_id_list))
        id_list_splited.append(all_id_list[start: end])
    # print(id_list_splited[0][0], id_list_splited[1][0])
    return id_list_splited
