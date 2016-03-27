import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import cross_validation

def main(argv, estimators = [5, 10, 30, 50, 100, 150, 200, 250, 300, 400, 500, 1000], rep = 20):
    for train_file, test_file in argv:
        train = pd.read_csv(train_file, index_col=0)
        test = pd.read_csv(test_file, index_col=0)
        test = test.drop('target', axis=1)
        X = np.array(train.drop('target', axis=1))
        y = np.array(train['target'])
        dtrain = xgb.DMatrix(X, label=y)

        param = {'eta':0.3, 'max_depth':8, 'min_child_weight':4, 'colsample_bytree':0.4,
                 'objective':'binary:logistic', 'eval_metric':'logloss'}
        num_round = 2

        # do cross validation, this will print result out as
        # [iteration]  metric_name:mean_value+std_value
        # std_value is standard deviation of the metric
        xgb.cv(param, dtrain, num_round, nfold=5,
               metrics={'error'}, seed = 0)

        # np.savetxt(test_file+'.xgb.txt', y_test)

if __name__=='__main__':
    # argv = [('train.t0.has.v8.csv', 'test.t0.has.v8.csv'), ('train.t0.no.v8.csv', 'test.t0.no.v8.csv')]
    # main(argv, [1000])

    argv = [('train.t1.csv', 'test.t1.csv')]
    main(argv, [100, 200, 300, 500, 750, 1000])