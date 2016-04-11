import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import BernoulliNB


def Binarize(columnName, df, features=None):
    if(features is None):
        features = np.unique(df[columnName].values)
    print(features)
    d = pd.DataFrame()
    for x in features:
        d[columnName+'_' + str(x)] = df[columnName].map(lambda y: 1 if y == x else 0)
    df.drop(columnName, inplace=True, axis=1)
    return d, features


def MungeData(train, test):
    scaler = preprocessing.MinMaxScaler((0, 20))
    cat_alpha = ['v3', 'v24', 'v30', 'v31', 'v47', 'v52', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110',
                 'v112', 'v56', 'v113', 'v125']

    features = train.columns[2:]
    for col in features:
        if col == 'v22':
            total = len(train) + len(test)
            features = np.unique(np.concatenate([train[col].values, test[col].values]))
            count = {}
            for x in features:
                count[x] = (np.sum(train[col] == x) + np.sum(test[col] == x))*1.0 / total
            train['v22'] = train[col].map(lambda x: count[x]).reshape(-1, 1)*20
            test['v22'] = test[col].map(lambda x: count[x]).reshape(-1, 1)*20

        elif (col in cat_alpha):
            print(col)
            bin_train, bin_features = Binarize(col, train)
            bin_test, _ = Binarize(col, test, bin_features)
            nb = BernoulliNB()
            nb.fit(bin_train, train['target'])
            train[col] = nb.predict_proba(bin_train)[:, 1].reshape(-1, 1)*20
            test[col] = nb.predict_proba(bin_test)[:, 1].reshape(-1, 1)*20

    return train, test

if __name__ == "__main__":
    na_flag = '.na-mean'
    print('Start')
    print('Importing Data')
    train = pd.read_csv('train.t1'+na_flag+'.csv', index_col=0)
    test = pd.read_csv('test.t1'+na_flag+'.csv', index_col=0)
    print(train[train.target == 0].shape[0])
    print(train[train.target == 1].shape[0])
    print('Munge Data')
    train, test = MungeData(train, test)
    train.to_csv('train.bayes'+na_flag+'.csv')
    test.to_csv('test.bayes'+na_flag+'.csv')
    print('Finish')