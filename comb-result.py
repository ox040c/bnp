import sys
import numpy as np
import pandas as pd

def main(argv, pivot_attr ='v8'):
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

if __name__ == '__main__':
    if  (len(sys.argv) == 3):
        main(sys.argv)
    else:
        print 'File names should be given.'
