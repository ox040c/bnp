import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

indices = ['v23', 'v10', 'v50', 'v114', 'v12', 'v14', 'v21', 'v34', 'v40']

def main(filename):
    data = pd.read_csv(filename, index_col=0)
    X = np.array(data[indices])/20.
    y = np.array(data['target'])
    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(X)
    np.savetxt(filename+'.tsne2d.txt', reduced)
    plt.scatter(reduced[:, 0], reduced[:, 1], cmap=y)
    plt.savefig('tsne.png')

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main('train.t1.csv')