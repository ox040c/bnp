import sys
import numpy as np
import pandas as pd

def combine(argv, pivot_attr ='v8'):
    if 'no' in argv[2]:
        file_has_attr, file_no_attr = argv[1], argv[2]
    else:
        file_has_attr, file_no_attr = argv[2], argv[1]
    result_has_attr = np.loadtxt(file_has_attr).tolist()
    result_no_attr = np.loadtxt(file_no_attr).tolist()
    result_has_attr.reverse()
    result_no_attr.reverse()
    test_raw = pd.read_csv('test-raw.csv')
    ids = []
    preds = []
    for i in range(len(test_raw)):
        ids.append(test_raw['ID'][i])
        if pd.isnull(test_raw[pivot_attr][i]):
            preds.append(result_no_attr.pop())
        else:
            preds.append(result_has_attr.pop())

    result = pd.DataFrame({'ID':ids,'PredictedProb':preds})
    result.to_csv('result.csv', index=False)

def generate(argv):
    preds = np.loadtxt(argv[1]).tolist()
    test_raw = pd.read_csv('test-raw.csv')
    ids = test_raw['ID'].tolist()
    result = pd.DataFrame({'ID':ids,'PredictedProb':preds})
    result.to_csv(argv[1]+'.result.csv', index=False)

if __name__ == '__main__':
    if  (len(sys.argv) == 3):
        combine(sys.argv)
    elif len(sys.argv) == 2:
        generate(sys.argv)
    else:
        print 'File names should be given.'
