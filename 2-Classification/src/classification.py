# @toffanetto

import numpy as np
import pandas as pd

NUMBER_OF_CLASSES = 6
NUMBER_OF_ATRIBUTES = 561

def getData(train, raw):
    
    if(train == True and raw == False):
    
        df_X = pd.read_csv("../data/UCI HAR Dataset/train/X_train.txt", sep="\s+", header=None)
        df_y = pd.read_csv("../data/UCI HAR Dataset/train/y_train.txt", names=["Class"])
        
        X = df_X.to_numpy()
        y = df_y.to_numpy()
        return X, y
    
    if(train == False and raw == False):
        df_X = pd.read_csv("../data/UCI HAR Dataset/test/X_test.txt", sep="\s+", header=None)
        df_y = pd.read_csv("../data/UCI HAR Dataset/test/y_test.txt", names=["Class"])
        
        X = df_X.to_numpy()
        y = df_y.to_numpy()
        return X, y
    
def softmax(x,w):
    n = w.shape
    y = np.zeros([n[0]])
    num = np.zeros([n[0]])
    
    for k in range(w.shape[0]):
        num[k] = (x).dot(w[k])
        
    y = np.exp(num)/sum(np.exp(num))
    
    n_class = np.argmax(y)+1
    
    return y, n_class

def oneHotEncoding(y):
    y_ohe = np.zeros([len(y),NUMBER_OF_CLASSES])
    for i in range(len(y)):
        y_ohe[i,(y[i]-1)] = 1
    return y_ohe

def getJ_CE(y, y_hat):
    J_CE = 0
    
    for i in range(y.shape[1]):
        for k in range(y.shape[0]):
            J_CE += y[i][k]*np.log(y_hat[i][k])
    
    return -(J_CE)
            
def getdJ_CEdW(y, y_hat, x):
    
    return

    
    