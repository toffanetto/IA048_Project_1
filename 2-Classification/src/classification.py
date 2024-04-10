# @toffanetto

import numpy as np
import pandas as pd
import random

NUMBER_OF_CLASSES = 6
NUMBER_OF_ATRIBUTES = 561

STEP = 0.005

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
    y = np.zeros([x.shape[0],n[0]])
    n_class = np.zeros([x.shape[0]], dtype=np.uint8)
    num = np.zeros([n[0]])
    
    for j in range(x.shape[0]):
    
        for k in range(w.shape[0]):
            num[k] = np.sum((x[j]).dot(w[k].T))
            
        y[j] = np.exp(num)/np.sum(np.exp(num))
        
        n_class[j] = np.argmax(y[j])+1
    
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
    e = (y - y_hat)
    dJCE = (x.T*e).T
    
    return dJCE

def trainClassifier(X, y,batch):
    W = np.random.rand(NUMBER_OF_CLASSES, NUMBER_OF_ATRIBUTES)/100 # Rand values in the interval (0, 0,01)

    y_ohe = oneHotEncoding(y=y)
    
    hit = [0]
    not_hit = [0]
    RMSE_sample = [0]
    
    r = list(range(len(y)))
    
    random.shuffle(r)

    for i in r:

        Xi = X[i, np.newaxis] 
        
        y_hat, class_y_hat = softmax(Xi,W)
        
        if(y[i] == class_y_hat):
            hit.append(hit[-1]+1)
        else:
            not_hit.append(hit[-1]+1)
            
        RMSE_sample.append(np.average(np.sqrt(y[i]**2 - y_hat**2)))
        
        dJCE = getdJ_CEdW(y=y_ohe[i],y_hat=y_hat,x=Xi)
        
        W = W + STEP*dJCE
        
        
    #print(W)
    
    return W, RMSE_sample, hit, not_hit
    
def classify(x,W):
    y_hat, class_y_hat = softmax(x=x,w=W)
    return y_hat,class_y_hat

def rateModel(y,y_hat):
    hit_rate = 0
    not_hit_rate = 0
    confusion_matrix = np.zeros([NUMBER_OF_CLASSES,NUMBER_OF_CLASSES])
    
    for i in range(len(y)):
        confusion_matrix[y[i]-1, y_hat[i]-1] += 1
        if (y[i] == y_hat[i]):
            hit_rate += 1
        
    hit_rate /= len(y)
    not_hit_rate = 1 - hit_rate
        
    return confusion_matrix, hit_rate, not_hit_rate