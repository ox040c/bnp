import numpy as np
import pandas as pd
import sys

def main(argv):
    for i in range(len(argv)/2):
        filename = argv[i*2+1]
        weight = float(argv[i*2+2])
        data = pd.read_csv(filename)
        if i == 0:
            result = pd.DataFrame({'ID':data['ID'],'PredictedProb':np.array(data['PredictedProb'])*weight})
        else:
            result['PredictedProb'] += np.array(data['PredictedProb'])*weight
    result.to_csv('result.csv', index=False)

if __name__=='__main__':
    main(sys.argv)