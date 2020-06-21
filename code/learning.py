import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_validate, cross_val_predict, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from data_analysis import *
from get_structure import *
from resource import *
from get_feature import *
import time, re, os
from random import shuffle
import pandas as pd

def cv(X, y):
    n_kfolds = 10
    crossvalidation = KFold(n_splits=n_kfolds, shuffle=False, random_state=1)
    # param_test = {'n_estimators':range(60, 200, 20), 'min_samples_split':range(15,25,5)}
    clf = GradientBoostingRegressor(loss='ls', learning_rate=0.001, n_estimators=15000, max_depth=7, min_samples_split=5, subsample=1.0, max_features='sqrt', random_state = 0)
    # grsearch = GridSearchCV(estimator=clf, param_grid=param_test, scoring=['r2', 'neg_mean_squared_error'], n_jobs=4, cv=crossvalidation, refit='r2')
    # grsearch.fit(X,y)
    # print(grsearch.grid_scores_, grsearch.best_params_, grsearch.best_score_)
    cv_results = cross_validate(clf, X, y, cv=crossvalidation, n_jobs=-1, scoring=['r2', 'neg_mean_squared_error'])
    predictions = cross_val_predict(clf, X, y, cv=crossvalidation, n_jobs=-1)

    return cv_results, predictions

def learning(X_train, y_train, X_test, y_test, filename):
    clf = GradientBoostingRegressor(loss='ls', learning_rate=0.001, n_estimators=15000, max_depth=7, min_samples_split=5, subsample=1.0, max_features='sqrt', random_state = 0)
    clf.fit(X_train, y_train)
    joblib.dump(clf, filename + '.pkl') 
    predict = clf.predict(X_test)
    print("r2_score: {0}, mse: {1}".format(r2_score(y_test, predict), mean_squared_error(y_test, predict)))

def get_learning_data(data_dir, ldata_dir):
    gap_file = open(data_dir + '/gaps', 'r')
    train_list, test_list = split_data(data_dir)
    X_train = np.zeros((len(train_list), feature_len), float)
    X_test = np.zeros((len(test_list), feature_len), float)
    y1_train = np.zeros(len(train_list), float)
    y1_test = np.zeros(len(test_list), float)
    y2_train = np.zeros(len(train_list), float)
    y2_test = np.zeros(len(test_list), float)
    lines = gap_file.read().splitlines()
    gap_file.close()

    i_train = 0; i_test = 0
    for i in range(len(lines[1:])):
        line = lines[i+1]
        id, hse, pbe = line.split()[:3]
        if id in train_list:
            y1_train[i_train] = hse
            y2_train[i_train] = pbe
            with open(ldata_dir + '/feature/' + id + '_feature.npy', 'rb') as f:
                data = np.load(f)
                X_train[i_train][:] = data
                i_train += 1
        if id in test_list:
            y1_test[i_test] = hse
            y2_test[i_test] = pbe
            with open(ldata_dir + '/feature/' + id + '_feature.npy', 'rb') as f:
                data = np.load(f)
                X_test[i_test][:] = data
                i_test += 1
    return X_train, y1_train, y2_train, X_test, y1_test, y2_test

def minmax_preprocess_data(X_train, X_test):
    min_max_scaler = MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    X_test_minmax = min_max_scaler.transform(X_test)
    return X_train_minmax, X_test_minmax

def standard_preprocess_data(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train_stand = scaler.transform(X_train)
    X_test_stand = scaler.transform(X_test)
    return X_train_stand, X_test_stand

def get_X_y(data_dir):
    with open(data_dir + '/gaps', 'r') as f:
        lines = f.read().splitlines()
    X = []; y_hse = []; y_pbe = []
    for line in lines[1:]:
        id, hse, pbe = line.split()[:3]
        if float(hse) == 0.0: continue
        with open(data_dir + '/feature_add_coul_withoutbin/' + id + '_feature.npy', 'rb') as fe:
            feature = np.load(fe)
            X.append(feature)
        y_hse.append(hse)
        y_pbe.append(pbe)
    X = np.asarray(X)
    y_hse = np.asarray(y_hse, float)
    y_pbe = np.asarray(y_pbe, float)
    return X, y_hse, y_pbe

def get_icsd_X_y_remove_unusual(data_dir, fname, lname):
    unusual_list = icsd_unusual_data(data_dir)
    # elementary_list = remove_elementary(data_dir)
    with open(data_dir + '/' + lname + '.txt', 'r') as f:
        lines = f.read().splitlines()
    # shuffle(lines)
    X = []; y = []
    # err_list = ['3831-C1Cd1O3', '3584-Cd1O3Ti1', '2465-Cs1I1O3', '8011-C1Gd3Pb1']
    err_list = []
    for line in lines:
        id, delta_e = line.split()[0], line.split()[1]
        if id in unusual_list: continue
        with open(data_dir + '/' + fname + '/' + id + '_feature.npy', 'rb') as fe:
            feature = np.load(fe)
        # with open(data_dir + '/' + fname + '/' + id, 'r') as f:
        #     feature = f.read().splitlines()[0].split(",")[1:]
            # if feature.shape[0] != 2835:
            #     get_feature_with_s_nobin(data_dir, id)
            #     with open(data_dir + '/' + fname + '/' + id + '_feature.npy', 'rb') as ofe:
            #         feature = np.load(ofe)
        X.append(feature)
        y.append(delta_e)

    X = np.asarray(X)
    y = np.asarray(y, float)
    print(X.shape, y.shape)
    return X, y

def get_icsd_X_y(data_dir, fname):
    with open(data_dir + '/properties.txt', 'r') as f:
        lines = f.read().splitlines()
    X = []; y = []; id_list = []
    for line in lines[1:]:
        id, delta_e = line.split()[:2]
        if 'Xe' in id or 'Kr' in id or 'He' in id or 'Ne' in id or 'Ar' in id or 'Ac' in id or 'Pm' in id or 'Pa' in id or 'Tc' in id: continue
        with open(data_dir + '/' + fname + '/' + id + '_feature.npy', 'rb') as fe:
            feature = np.load(fe)
            X.append(feature)
        y.append(delta_e)
        id_list.append(id)
    X = np.asarray(X)
    y = np.asarray(y, float)
    return X, y

def get_X_y_basic(data_dir):
    with open(data_dir + '/basic_infor', 'r') as f:
        lines = f.read().splitlines()
    X = []; y_hse = []; y_pbe = []
    for line in lines[1:]:
        hse, id, pbe = line.split(',')[1:]
        try:
            with open(data_dir + '/feature_add_coul_withoutbin/' + id + '_feature.npy', 'rb') as fe:
                feature = np.load(fe)
        except:
            get_feature(data_dir, id)
            with open(data_dir + '/feature_add_coul_withoutbin/' + id + '_feature.npy', 'rb') as fe:
                feature = np.load(fe)
        X.append(feature)         
        y_hse.append(hse)
        y_pbe.append(pbe)
    X = np.asarray(X)
    y_hse = np.asarray(y_hse, float)
    y_pbe = np.asarray(y_pbe, float)
    return X, y_hse, y_pbe

def icsd_unusual_data(icsd_dir):
    with open(icsd_dir + '/properties.txt', 'r') as f:
        lines = f.read().splitlines()
    unusual_list = []
    for line in lines:
        id, delta_e = line.split()[:2]
        if float(delta_e) > 2.0:
            unusual_list.append(id)
        if float(delta_e) <= 2.0 and ('Xe' in id or 'Kr' in id or 'He' in id or 'Ne' in id or 'Ar' in id or 'Ac' in id or 'Pm' in id or 'Pa' in id or 'Tc' in id):
            unusual_list.append(id)
    return unusual_list

def rm_unusual(icsd_dir):
    unusual_list = icsd_unusual_data(icsd_dir)
    n = 0
    for id in unusual_list:
        os.system("rm ../voro-ml-si/datasets/icsd-all/" + id)
        n += 1
    print(n)

def remove_elementary(icsd_dir):
    elementary = []
    pattern = re.compile("[A-Z]{1}[a-z]{0,1}")
    # substance to all pair
    sub_dict = {}
    with open(icsd_dir + '/properties.txt', 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        sub = line.split()[0]
        eles = pattern.findall(sub)
        if len(eles) == 1: elementary.append(sub)
    return elementary

def get_voro_X_y(icsd_dir):
    csv_data = pd.read_csv(icsd_dir + '/icsd-all.csv')
    csv_data.fillna(0, inplace=True)
    # csv_data.drop(drop_index, inplace=True)
    columns_list = csv_data.columns.values.tolist()
    with open(icsd_dir + '/voro_feature', 'w') as f:
        for c in columns_list:
            f.write(c + '\n')
    X = np.asarray(csv_data.loc[:, columns_list[:-1]])
    y = np.asarray(csv_data.loc[:, columns_list[-1]])
    # y_set = set()
    # for i in y:
    #     y_set.add(i)
    # print(len(y_set))
    return X, y

def test(icsd_dir):
    csv_data = pd.read_csv(icsd_dir + '/icsd-all.csv')
    csv_data.fillna(0, inplace=True)
    columns_list = csv_data.columns.values.tolist()
    y = np.asarray(csv_data.loc[:, columns_list[-1]])

    prop_dict = {}
    with open(icsd_dir + '/properties.txt') as f:
        lines = f.read().splitlines()
    for l in lines:
        id, deltaE = l.split()[:2]
        prop_dict[id] = deltaE
    
    path = "/udata/yjiang/Topology_ML/voro-ml-si/datasets/icsd-all"
    file_list = []
    for file in os.listdir(path):
        if file in prop_dict:
            file_list.append(file)
    with open("11.txt", 'w') as f:
        for i in range(len(file_list)):
            file = file_list[i]
            f.write(file + " " + prop_dict[file] + " " + str(y[i]) + '\n')


if __name__ == '__main__':
    get_icsd_X_y_remove_unusual(icsd_dir, "feature_topo_compo", "properties")