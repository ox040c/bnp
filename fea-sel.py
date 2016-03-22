import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib

def main(argv):
    data = pd.read_csv(argv, index_col=0)
    y = data['target']
    X = data.drop('target', axis=1)
    # elastic net
    enet = ElasticNetCV(n_jobs=-1, normalize=True)
    enet.fit(X, y)
    joblib.dump(enet, argv+'.enet.pkl')
    # extreme random tree
    etree = ExtraTreesClassifier(n_jobs=-1)
    etree.fit(X, y)
    joblib.dump(etree, argv+'.etree.pkl')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Data should be given.'
    else:
        main(sys.argv[1])