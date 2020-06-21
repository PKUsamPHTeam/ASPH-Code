from data_analysis import *
from get_feature import *
from learning import *
from resource import *
from get_structure import *
from enlarge_cell import *
import numpy as np
import os
from multiprocessing import Pool

def icsd_learning_30_1(icsd_dir, fname, lname):
    # n_kfolds = 10
    # crossvalidation = KFold(n_splits=n_kfolds, shuffle=True, random_state=1)
    X, y = get_icsd_X_y_remove_unusual(icsd_dir, fname, lname)
    min_max_scaler = MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=1000, train_size=30000, random_state=0)
    clf = GradientBoostingRegressor(loss='ls', learning_rate=0.001, n_estimators=300000, max_depth=7, min_samples_split=5, subsample=0.85, max_features='sqrt', random_state = 0)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    # score = r2_score(y_test, clf.predict(X_test))
    feature_importance = clf.feature_importances_
    with open(icsd_dir + '/feature_importance_' + fname + '.npy', 'wb') as out_file:
        np.save(out_file, feature_importance)
    # print("r2_score: " + score)
    # scores = cross_validate(clf, X_minmax, y, scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'], n_jobs=10, cv=crossvalidation)
    # predicted = cross_val_predict(clf, X_minmax, y, cv=10, n_jobs=10)
    with open(icsd_dir + '/predict_icsd_' + fname + '_cutoff_12_nocv.npz', 'wb') as out_file:
        np.savez(out_file, predicted=predicted, y=y_test)
    r2 = r2_score(y_test, predicted)
    mse = mean_squared_error(y_test, predicted)
    rmse = np.sqrt(abs(mse))
    mae = mean_absolute_error(y_test, predicted)

    print(fname + ", 30_1, n_estimator=300000, max_depth=7, min_samples_split=5, subsample=0.85 remove >=2.0 data " + str(cut))
    # rmse_scores = [np.sqrt(abs(s)) for s in scores['test_neg_mean_squared_error']]
    # r2_scores = scores['test_r2']
    # mae_scores = scores['test_neg_mean_absolute_error']

    # print(rmse_scores)
    print("r2_score: {0}, rmse: {1}, mae: {2}".format(r2, rmse, mae))
    #print("r2_score: {0}, rmse: {1}, mae: {2}".format(np.mean(np.abs(r2_scores)), np.mean(np.abs(rmse_scores)), np.abs(mae_scores)))


def icsd_learning_cv(icsd_dir, fname, lname):
    n_kfolds = 10
    crossvalidation = KFold(n_splits=n_kfolds, shuffle=True, random_state=1)
    X, y = get_icsd_X_y_remove_unusual(icsd_dir, fname, lname)
    print("get data")
    min_max_scaler = MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    clf = GradientBoostingRegressor(loss='ls', learning_rate=0.001, n_estimators=300000, max_depth=7, min_samples_split=5, subsample=0.85, max_features='sqrt', random_state = 0)

    # scores = cross_validate(clf, X_minmax, y, scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'], n_jobs=10, cv=crossvalidation)
    predicted = cross_val_predict(clf, X_minmax, y, cv=crossvalidation, n_jobs=10)
    with open(icsd_dir + '/predict_icsd_' + fname + '_cutoff_12_cv.npz', 'wb') as out_file:
        np.savez(out_file, predicted=predicted, y=y)
    r2 = r2_score(y, predicted)
    mse = mean_squared_error(y, predicted)
    rmse = np.sqrt(abs(mse))
    mae = mean_absolute_error(y, predicted)

    print(fname + ", cv=10, n_estimator=300000, max_depth=7, min_samples_split=5, subsample=0.85 remove >=2.0 data " + str(cut))

    print("r2_score: {0}, rmse: {1}, mae: {2}".format(r2, rmse, mae))


def ortho_learning(data_dir, fname, lname):
    n_kfolds = 10
    crossvalidation = KFold(n_splits=n_kfolds, shuffle=False, random_state=1)
    X, y = get_icsd_X_y_remove_unusual(data_dir, fname, lname)
    min_max_scaler = MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    print(1)
    clf = GradientBoostingRegressor(loss='ls', learning_rate=0.001, n_estimators=300000, max_depth=7, min_samples_split=5, subsample=0.85, max_features='sqrt', random_state = 0)
    predicted = cross_val_predict(clf, X_minmax, y, cv = crossvalidation, n_jobs=10)
    with open(data_dir + '/predict_' + fname + '.npz', 'wb') as out_file:
        np.savez(out_file, predicted=predicted, y=y)
    r2 = r2_score(y, predicted)
    mse = mean_squared_error(y, predicted)
    rmse = np.sqrt(abs(mse))
    mae = mean_absolute_error(y, predicted)
    print(fname + ", ortho_perovskite, n_estimator=300000, max_depth=7, min_samples_split=5, subsample=0.85 remove unusual data " + str(cut))
    # rmse_scores = [np.sqrt(abs(s)) for s in scores['test_neg_mean_squared_error']]
    # r2_scores = scores['test_r2']
    # mae_scores = scores['test_neg_mean_absolute_error']
    print("r2_score: {0}, rmse: {1}, mae: {2}".format(r2, rmse, mae))

def gap_learning(data_dir):
    X, y_hse, y_pbe = get_X_y_basic(data_dir)
    min_max_scaler = MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    clf = GradientBoostingRegressor(loss='ls', learning_rate=0.001, n_estimators=100000, max_depth=7, min_samples_split=5, subsample=0.85, max_features='sqrt', random_state = 0)
    score = cross_val_score(clf, X_minmax, y_hse, scoring='r2', n_jobs=-1, cv=50)
    # predicted = cross_val_predict(clf, X_minmax, y_hse, n_jobs=10, cv=10)
    # with open(data_dir + '/predict_basic.npz', 'wb') as out_file:
    #     np.savez(out_file, predicted=predicted, y_hse=y_hse)
    # feature_importance = clf.feature_importances_
    # with open(data_dir + '/feature_importance.npy', 'wb') as out_file:
    #     np.save(out_file, feature_importance=feature_importance)
    print(score)
    print(np.mean(score))
    print("======================")

def random_learning(seed):
    X, y = get_icsd_X_y_remove_unusual(icsd_dir, 'feature_add_s_nobin', 'properties')
    min_max_scaler = MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=1000, train_size=30000, random_state=seed)
    n = 100000
    mdepth = 8
    clf = RandomForestRegressor(n_estimators=n, max_depth=mdepth, min_samples_split=5, max_features='sqrt', random_state=0)
    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)
    with open(icsd_dir + '/predict_randomforest' + seed + '.npz', 'wb') as out_file:
        np.savez(out_file, predicted=predicted, y=y_test)
    r2 = r2_score(y_test, predicted)
    mse = mean_squared_error(y_test, predicted)
    rmse = np.sqrt(abs(mse))
    mae = mean_absolute_error(y_test, predicted)
    print("r2_score: {0}, rmse: {1}, mae: {2}, seed: {3}, n_estimators: {4}, max_depth: {5}".format(r2, rmse, mae, seed, n, mdepth))


def gap_learning_random(data_dir):
    X, y_hse, y_pbe = get_X_y_basic(data_dir)
    min_max_scaler = MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_minmax, y_hse, test_size=0.1)
    clf = GradientBoostingRegressor(loss='ls', learning_rate=0.001, n_estimators=100000, max_depth=7, min_samples_split=5, subsample=0.85, max_features='sqrt', random_state = 0)
    clf.fit(X_train, y_train)
    feature_importance = clf.feature_importances_
    with open(data_dir + '/feature_importance.npy', 'wb') as out_file:
        np.save(out_file, feature_importance)

def voro_ml_learning_rf(icsd_dir):
    n_kfolds = 10
    crossvalidation = KFold(n_splits=n_kfolds, shuffle=False, random_state=1)
    X, y = get_voro_X_y(icsd_dir)
    min_max_scaler = MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    # X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=1000, train_size=30000, random_state=0)
    print(1)
    clf = RandomForestRegressor(n_estimators=50000, max_depth=7, min_samples_split=5, max_features='sqrt', random_state=0)
    # clf.fit(X_train, y_train)
    predicted = cross_val_predict(clf, X_minmax, y, cv = crossvalidation, n_jobs=10)
    # predicted = clf.predict(X_test)
    with open(icsd_dir + '/predict_rf_voro_cv' + '.npz', 'wb') as out_file:
        np.savez(out_file, predicted=predicted, y=y)
    r2 = r2_score(y, predicted)
    mse = mean_squared_error(y, predicted)
    rmse = np.sqrt(abs(mse))
    mae = mean_absolute_error(y, predicted)
    print("rf_voro_cv, n_estimator=50000, max_depth=7, min_samples_split=5")
    print("r2_score: {0}, rmse: {1}, mae: {2}".format(r2, rmse, mae))

    # feature_importance = clf.feature_importances_
    # with open(icsd_dir + '/feature_importance_rf_voro.npy', 'wb') as out_file:
    #     np.save(out_file, feature_importance)

def voro_ml_learning_rf_30_1(icsd_dir):
    X, y = get_voro_X_y(icsd_dir)
    min_max_scaler = MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=1000, train_size=30000, random_state=0)
    clf = RandomForestRegressor(n_estimators=50000, max_depth=7, min_samples_split=5, max_features='sqrt', random_state=0)
    print(1)
    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)
    with open(icsd_dir + '/predict_rf_voro_30_1' + '.npz', 'wb') as out_file:
        np.savez(out_file, predicted=predicted, y=y_test)
    r2 = r2_score(y_test, predicted)
    mse = mean_squared_error(y_test, predicted)
    rmse = np.sqrt(abs(mse))
    mae = mean_absolute_error(y_test, predicted)
    print("rf_voro_cv, n_estimator=50000, max_depth=7, min_samples_split=5")
    print("r2_score: {0}, rmse: {1}, mae: {2}".format(r2, rmse, mae))

    feature_importance = clf.feature_importances_
    with open(icsd_dir + '/feature_importance_rf_voro.npy', 'wb') as out_file:
        np.save(out_file, feature_importance)

def voro_ml_learning_gbr_30_1(icsd_dir):
    X, y = get_voro_X_y(icsd_dir)
    min_max_scaler = MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=1000, train_size=30000, random_state=0)
    clf = GradientBoostingRegressor(loss='ls', learning_rate=0.001, n_estimators=300000, max_depth=7, min_samples_split=5, subsample=0.85, max_features='sqrt', random_state = 0)
    print(1)
    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)

    with open(icsd_dir + '/predict_gbr_voro_30-1' + '.npz', 'wb') as out_file:
        np.savez(out_file, predicted=predicted, y=y_test)
    r2 = r2_score(y_test, predicted)
    mse = mean_squared_error(y_test, predicted)
    rmse = np.sqrt(abs(mse))
    mae = mean_absolute_error(y_test, predicted)
    print("gbr_voro_30-1, n_estimator=300000, max_depth=7, min_samples_split=5")
    print("r2_score: {0}, rmse: {1}, mae: {2}".format(r2, rmse, mae))

    feature_importance = clf.feature_importances_
    with open(icsd_dir + '/feature_importance_gbr_voro.npy', 'wb') as out_file:
        np.save(out_file, feature_importance)

def voro_ml_learning_gbr(icsd_dir):
    n_kfolds = 10
    crossvalidation = KFold(n_splits=n_kfolds, shuffle=False, random_state=1)
    X, y = get_voro_X_y(icsd_dir)
    min_max_scaler = MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    print(1)
    clf = GradientBoostingRegressor(loss='ls', learning_rate=0.001, n_estimators=300000, max_depth=7, min_samples_split=5, subsample=0.85, max_features='sqrt', random_state = 0)

    predicted = cross_val_predict(clf, X_minmax, y, cv = crossvalidation, n_jobs=10)

    with open(icsd_dir + '/predict_gbr_voro_cv' + '.npz', 'wb') as out_file:
        np.savez(out_file, predicted=predicted, y=y)
    r2 = r2_score(y, predicted)
    mse = mean_squared_error(y, predicted)
    rmse = np.sqrt(abs(mse))
    mae = mean_absolute_error(y, predicted)
    print("gbr_voro_cv, n_estimator=300000, max_depth=7, min_samples_split=5")
    print("r2_score: {0}, rmse: {1}, mae: {2}".format(r2, rmse, mae))


def batch_handle(id_list):
    for id in id_list:
        # get_prim_structure_info(data_dir, id)
        # enlarge_cell(data_dir, id)
        get_betti_num_prop(icsd_dir, id, 'Electronegativity')
        # need to change
        # get_feature_s_Bar0bin_othernon(icsd_dir, id)

def split_list(all_id_list):
    id_list_splited = []
    step = math.ceil(len(all_id_list)/10.0)
    for i in range(0, len(all_id_list), step):
        start = i
        end = min(i + step, len(all_id_list))
        id_list_splited.append(all_id_list[start: end])
    # print(id_list_splited[0][0], id_list_splited[1][0])
    return id_list_splited



if __name__ == '__main__':
    lname = 'properties'
    # with open(icsd_dir + '/' + lname + '.txt', 'r') as f:
    #     lines = f.read().splitlines()
    # id_list = []
    # for line in lines:
    #     id = line.split()[0]
    #     id_list.append(id)
    # id_list_splited = split_list(id_list)
    # pool = Pool(10)
    # pool.map(batch_handle, id_list_splited)
    # fnames = ['feature_0.25', 'feature_0.02', 'feature_add_coul_withoutbin', 'feature_add_sp', 'feature_element', 'feature_ph_withoutbin', 'feature_withoutbin', 'feature_add_spnobin', 'feature_add_spnobin_one', 'feature_add_spnobin_two', 'feature_add_spnobin_thr', 
    #         "feature_add_spnobin_Bar0", "feature_coulomb_matrix", "feature_add_spnobin_Bar0_1", "feature_add_p_nobin", 'feature_add_p_nobin_Bar0', 'feature_add_p_nobin_1', 'feature_add_spnobin_1', 'feature_add_p_nobin_oxide', 'feature_add_p_nobin_oxide_1', 'feature_add_sp_nobin_zip', 
    #         'feature_add_s_nobin', 'feature_add_s_nobin_Bar0', 'feature_add_s_0bin_othernon', 'feature_add_s_0bin_othernon_1']
    # ortho_learning(data_dir, fnames[12], lname)
    icsd_learning_cv(icsd_dir, "feature_prdf_origin", lname)
    # random_learning(5)
    # print(len(fnames))
    # for i in range(len(fnames)):
    #     if fnames[i] == "feature_coulomb_matrix":
    #         print(i)

    # test feature
    # for id in ["mp_549706_P63mc.vasp", "mp_549706_P63mc22.vasp", "mp_775808_P3m1.vasp"]:
    #     get_prim_structure_info(test_dir, id)
    #     enlarge_cell(test_dir, id)
    #     get_betti_num(test_dir, id)
    #     get_feature_with_s_nobin(test_dir, id)

    # voro reproduction
    # voro_ml_learning_gbr(icsd_dir)