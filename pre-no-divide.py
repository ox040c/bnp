import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


def notnull(data):
    return np.logical_not(pd.isnull(data))


def ln(indices, data, a, b):
    for i in indices:
        attr = featureNames[i]
        if attr not in drops:
            # transform
            data[attr] = np.log(np.maximum(1.0, a * data[attr] + b))
            # standardize
            flag = notnull(data[attr])
            data[attr][flag] = scaler.fit_transform(data[attr][flag].reshape(-1, 1))


def decode(x):
    ans = 0
    for i in x:
        ans = ans * 26 + (ord(i) - 64)
    return ans - 1

def cleanFeatures(data):
    # data = d.copy()
    total = len(data)
    for i in range(1, 132):
        attr = featureNames[i]
        if attr not in drops:
            flag = pd.isnull(data[attr])
            # if np.sum(flag) > total/4:
            #     data.drop(attr, axis=1, inplace=True)
            #     print '[WARNING] Drop %s ...' % attr
            # else:
            if i in cat_alpha or i in cat_num:
                data[attr][flag] = -1
            else:
                data[attr][flag] = -999 #np.mean(data[attr][np.logical_not(flag)])
    return data

def divideAndSave(data):
    cleanFeatures(data)
    train = data[:totalTrain]
    test = data[totalTrain:]
    test.index -= totalTrain
    train.to_csv('train.t1.csv')
    test.to_csv('test.t1.csv')

CONF_INTERVAL = 12*5

drops = ['v107', 'v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v11', 'v13', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20',
         'v25', 'v26', 'v27', 'v28', 'v29', 'v32', 'v33', 'v35', 'v36', 'v37', 'v39', 'v41', 'v42', 'v43', 'v44', 'v45',
         'v46', 'v48', 'v49', 'v51', 'v53', 'v54', 'v55', 'v57', 'v58', 'v59', 'v60', 'v61', 'v63', 'v64', 'v65', 'v67',
         'v68', 'v69', 'v70', 'v73', 'v76', 'v77', 'v78', 'v80', 'v81', 'v82', 'v83', 'v84', 'v85', 'v86', 'v87', 'v88',
         'v89', 'v90', 'v92', 'v93', 'v94', 'v95', 'v96', 'v97', 'v98', 'v99', 'v100', 'v101', 'v102', 'v103', 'v104',
         'v105', 'v106', 'v108', 'v109', 'v111', 'v115', 'v116', 'v117', 'v118', 'v119', 'v120', 'v121', 'v122', 'v123',
         'v124', 'v126', 'v127', 'v128', 'v130', 'v131']

ln_0 = [23, 1, 6, 7, 8, 10, 13, 15, 17, 18, 19, 25, 26, 27, 28, 32, 33, 37, 39, 43, 46, 50, 54, 55, 60, 63, 73, 76, 80,
        83, 84, 85, 86, 88, 92, 94, 95, 99, 101, 102, 103, 104, 108, 111, 116, 118, 119, 120, 121, 123, 124, 126, 127,
        128, 130, 131]
ln_1 = [20, 65, 68, 114]
ln_2 = [11, 36, 53, 61]
ln_3 = [12]
bias = [4, 14, 16, 21, 29, 34, 35, 40, 41, 42, 44, 45, 48, 49, 51, 57, 64, 67, 90, 93, 96]
ok = [2, 5, 9, 58, 59, 69, 70, 77, 78, 81, 82, 87, 89, 97, 98, 100, 105, 106, 109, 115, 117, 122]
cat_num = [38, 62, 72, 129]
cat_alpha = [3, 24, 30, 31, 47, 52, 66, 71, 74, 75, 79, 91, 107, 110, 112, 22, 56, 113, 125]

assert len(ln_0) + len(ln_1) + len(ln_2) + len(ln_3) + len(bias) + len(ok) + len(cat_num) + len(cat_alpha) == 131

# input
train = pd.read_csv("train-raw.csv")
train = train.drop(['ID'], axis=1)
target = train.target
featureNames = train.columns.values
totalTrain = len(train)

test = pd.read_csv("test-raw.csv")
test = test.drop(['ID'], axis=1)
test.index += totalTrain

data = pd.concat([train, test])

# remove features
for i in drops:
    data.drop(i, axis=1, inplace=True)

print "Drop: ",
print drops

scaler = preprocessing.MinMaxScaler((0, 20))

# ln_?
ln(ln_0, data, 1, 1)
ln(ln_1, data, -1, 21)
ln(ln_2, data, -1, 19)
ln(ln_3, data, 1, -4)

# cat_alpha
for i in cat_alpha:
    attr = featureNames[i]
    if attr not in drops:
        flag = notnull(data[attr])
        data[attr][flag] = data[attr][flag].map(decode)

# smooth minimum & maximum
for i in range(1, 132):
    print 'Transforming %d ...' % i
    if i not in cat_alpha and i not in cat_num:
        attr = featureNames[i]
        if attr not in drops:
            flag = notnull(data[attr])
            vals = np.array(data[attr][flag])
            vals.sort()
            m = vals[CONF_INTERVAL-1]
            data[attr][data[attr]<m] = m
            m = vals[-CONF_INTERVAL]
            data[attr][data[attr]>m] = m
            data[attr][flag] = scaler.fit_transform(data[attr][flag].reshape(-1, 1))

divideAndSave(data)
