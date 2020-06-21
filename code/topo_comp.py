from data_analysis import *
from get_feature import *
from learning import *
from resource import *
from get_structure import *
from enlarge_cell import *
import numpy as np
import os

def icsd_learning_cv_repeated(icsd_dir, fname, lname, times):
    n_kfolds = 10
    crossvalidation = KFold(n_splits=n_kfolds, shuffle=True, random_state=1)
    X, y = get_icsd_X_y_remove_unusual(icsd_dir, fname, lname)
    print("get data")
    min_max_scaler = MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    for i in range(times):
        clf = GradientBoostingRegressor(loss='ls', learning_rate=0.001, n_estimators=300000, max_depth=7, min_samples_split=5, subsample=0.85, max_features='sqrt', random_state = i)

        # scores = cross_validate(clf, X_minmax, y, scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'], n_jobs=10, cv=crossvalidation)
        predicted = cross_val_predict(clf, X_minmax, y, cv=crossvalidation, n_jobs=10)
        with open(icsd_dir + '/predict_icsd_' + fname + str(i) + '_cutoff_12_rep_cv.npz', 'wb') as out_file:
            np.savez(out_file, predicted=predicted, y=y)
        r2 = r2_score(y, predicted)
        mse = mean_squared_error(y, predicted)
        rmse = np.sqrt(abs(mse))
        mae = mean_absolute_error(y, predicted)

        print(fname + ", cv=10, n_estimator=300000, max_depth=7, min_samples_split=5, subsample=0.85 remove >=2.0 data " + str(cut))
        print("times: " + str(i))
        print("r2_score: {0}, rmse: {1}, mae: {2}".format(r2, rmse, mae))

if __name__ == "__main__":
    f_list = ["feature_topo_compo", "feature_coulomb_sine_matrix", "feature_add_s_nobin", "feature_add_s_nobin_Bar0"]
    icsd_learning_cv_repeated(icsd_dir, f_list[3], "properties", 20)