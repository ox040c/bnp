import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import cross_validation

def transTrainData(argv):
    data = pd.read_csv(argv, index_col=0)
    X = np.array(data.drop('target', axis=1))
    y = np.array(data['target'])
    dtrain = xgb.DMatrix(X, label=y)
    dtrain.save_binary(argv+'.buffer')

def transTestData(argv):
    data = pd.read_csv(argv, index_col=0)
    X = np.array(data.drop('target', axis=1))
    dtrain = xgb.DMatrix(X)
    dtrain.save_binary(argv+'.buffer')

def main(argv):
    for train_file, test_file in argv:
        # train = pd.read_csv(train_file, index_col=0)
        test = pd.read_csv(test_file, index_col=0)
        test = np.array(test.drop('target', axis=1))
        dtest = xgb.DMatrix(test)
        # X = np.array(train.drop('target', axis=1))
        # y = np.array(train['target'])
        # dtrain = xgb.DMatrix(X, label=y)
        dtrain = xgb.DMatrix(train_file)
        param = {'eta':0.05, 'max_depth':8, 'min_child_weight':4, 'colsample_bytree':0.8,
                 'objective':'binary:logistic', 'eval_metric':'logloss'}
        num_round = 280
        model = xgb.train(param, dtrain, num_round)
        pred = model.predict(dtest)
        return pred

        # do cross validation, this will print result out as
        # [iteration]  metric_name:mean_value+std_value
        # std_value is standard deviation of the metric
        # print xgb.cv(param, dtrain, num_round, nfold=5,
        #        metrics={'error','logloss'}, seed = 0)

        # np.savetxt(test_file+'.xgb.txt', y_test)

if __name__=='__main__':
    # argv = [('train.t0.has.v8.csv', 'test.t0.has.v8.csv'), ('train.t0.no.v8.csv', 'test.t0.no.v8.csv')]
    # main(argv, [1000])

    argv = [('train.xgb.buffer', 'test.bayes.csv')]
    main(argv)