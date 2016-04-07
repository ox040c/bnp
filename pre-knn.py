import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
import sys

def cv(argv, ks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]):
    data = pd.read_csv(argv, index_col=0)
    y = data['target']
    X = data.drop(['target'], axis=1)
    # for i in ['v22', 'v3', 'v24', 'v30', 'v31', 'v47', 'v52', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110',
    #              'v112', 'v56', 'v113', 'v125']:
    #     try:
    #         X.drop([i], axis=1, inplace=True)
    #     except:
    #         pass
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4)
    for k in ks:
        knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=k)
        knn.fit(X_train, y_train)
        pred = knn.predict_proba(X_test)[:, 1]
        print 'K=%d, acc=%f, log=%f' % (k, accuracy_score(y_test, pred>0.5), log_loss(y_test, pred))

if __name__=='__main__':
    if len(sys.argv)==2:
        cv(sys.argv[1])
