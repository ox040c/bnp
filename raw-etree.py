import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation

def main(argv, estimators = [5, 10, 30, 50, 100, 150, 200, 250, 300, 400, 500, 1000]):
    for train_file, test_file in argv:
        train = pd.read_csv(train_file, index_col=0)
        test = pd.read_csv(test_file, index_col=0)
        X = train.drop('target', axis=1)
        y = train['target']
        cv_scores = []
        for estimator in estimators:
            etree = ExtraTreesClassifier(n_estimators=estimator, n_jobs=1)
            score = np.mean(cross_validation.cross_val_score(etree, X, y, n_jobs=-1))
            cv_scores.append(score)
            print "Estimator = %d, score = %f" % (estimator, score)

        cv_best_index = np.argmax(cv_scores)
        etree = ExtraTreesClassifier(n_estimators=estimators[cv_best_index], n_jobs=-1)
        etree.fit(X, y)
        y_test = etree.predict(test)
        y_test.to_csv(test_file+'.predict.csv')

if __name__=='__main__':
    argv = [('train-t0-has-v8.csv', 'test-t0-has-v8.csv'), ('train-t0-no-v8.csv', 'test-t0-no-v8.csv')]
    main(argv)
