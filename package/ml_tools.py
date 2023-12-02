# -*- coding: utf-8 -*-
"""
Created on    2023/11/16 10:36

@author: roger
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

feature_path = {
    "X_train" : r"./features/X_train.npy",
    "X_test" : r"./features/X_test.npy",
    "y_train" : r"./features/y_train.npy",
    "y_test" : r"./features/y_test.npy"
}

#input X:2D ndarray(n * width, 256);   output 
def data_process(X_train0, X_test0, X_train1, X_test1, X_train2, X_test2, nor_method, width):

    """X = X.reshape(y.shape[0], width, X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split, random_state = 42)

    X_train = X_train.reshape(y_train.shape[0] * width, X.shape[2])
    X_test = X_test.reshape(y_test.shape[0] * width, X.shape[2])

    shape0 = X_train.shape[0]
    shape1 = X_test.shape[0]
    shape2 = X_train.shape[1]"""

    X_train = np.concatenate((X_train0, X_train1, X_train2), axis = 0)
    X_test = np.concatenate((X_test0, X_test1, X_test2), axis = 0)

    shape0 = X_train.shape[0]
    shape1 = X_test.shape[0]
    shape2 = X_train.shape[1] #256

    X_train, X_test = normalize(X_train, X_test, nor_method)

    X_train = X_train.reshape((int(shape0 / width), width, shape2, 1))
    X_test = X_test.reshape((int(shape1 / width), width, shape2, 1))


    return X_train, X_test

def normalize(X_train, X_test, nor_method):
    if nor_method is 0:
        scaler = StandardScaler()
        print("Standard Scaler")
    elif nor_method is 1:
        scaler = MinMaxScaler()
        print("MinMax Scaler")
    else:
        print("No normalization\n")
        return X_train, X_test
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("--Normalization Done--\n")
    return X_train, X_test

def feature_save(X_train, X_test, y_train, y_test, feature_path):
    print("\n--feature saving start--\n")
    for key in feature_path:
        if os.path.exists(feature_path[key]):
            os.remove(feature_path[key])
            print(f"remove {feature_path[key]}")
        else:
            print(f"{feature_path[key]} is empty")
        
        np.save(feature_path[key], locals()[key])

    print("--feature saving finish--")
    #print("y_test:" np.load(feature_path["y_test"])) #debug
    return

def feature_load(feature_path):
    print("\n--feature loading start--\n")
    for key in feature_path:
        if os.path.exists(feature_path[key]) == False:
            print(f"Error! {feature_path[key]} is empty")
        
    X_train = np.load(feature_path["X_train"])
    X_test = np.load(feature_path["X_test"])
    y_train = np.load(feature_path["y_train"])
    y_test = np.load(feature_path["y_test"])
    print("--feature loading finish--")

    return X_train, X_test, y_train, y_test

