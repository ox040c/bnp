import numpy as np
import pandas as pd
from sklearn import preprocessing

prefixs = ['train', 'test']

def main(fst, snd, dim):
    scaler = preprocessing.MinMaxScaler((0, 20))
    for prefix in prefixs:
        data = pd.read_csv('.'.join([prefix,fst,'csv']), index_col=0)
        features = np.loadtxt('.'.join([prefix,snd,'txt']))
        for i in range(dim):
            feature = scaler.fit_transform(features[:, i])
            data[snd+'.'+str(i)] = feature
        data.to_csv('.'.join([prefix,'t2','csv']))

if __name__ == '__main__':
    main('t1', 't1.csv.tsne3d', 3)