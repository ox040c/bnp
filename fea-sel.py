import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib

def elasticNet(argv):
    data = pd.read_csv(argv, index_col=0)
    y = data['target']
    X = data.drop('target', axis=1)
    featureNames = X.columns.values
    enet = ElasticNetCV(n_jobs=-1, normalize=True)
    enet.fit(X, y)
    dropIdx = featureNames[enet.coef_ < 1e-5]
    print "Elastic Net drop: %d" % len(dropIdx)
    data.drop(dropIdx, axis=1, inplace=True)
    data.to_csv(argv+'.enet.csv')
    return enet

def extraTrees(argv):
    data = pd.read_csv(argv, index_col=0)
    y = data['target']
    X = data.drop('target', axis=1)
    featureNames = X.columns.values
    etree = ExtraTreesClassifier(n_jobs=-1)
    etree.fit(X, y)
    dropIdx = featureNames[etree.feature_importances_ < np.mean(etree.feature_importances_)]
    print "ExtraTrees drop: %d" % len(dropIdx)
    data.drop(dropIdx, axis=1, inplace=True)
    data.to_csv(argv+'.etree.csv')
    return etree

def main(argv):
    enet = elasticNet(argv)
    etree = extraTrees(argv)
    # joblib.dump(enet, argv+'.enet.pkl')
    # joblib.dump(etree, argv+'.etree.pkl')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Data should be given.'
    else:
        main(sys.argv[1])