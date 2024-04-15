# @toffanetto

import numpy as np
import pandas as pd
import random

NUMBER_OF_CLASSES = 6

STEP = 0.01

def getData(train, raw):
    
    if(train == True and raw == False):
    
        df_X = pd.read_csv("../data/UCI HAR Dataset/train/X_train.txt", sep="\s+", header=None)
        df_y = pd.read_csv("../data/UCI HAR Dataset/train/y_train.txt", names=["Class"])
        
        X = df_X.to_numpy()
        y = df_y.to_numpy()
        
        classes_rate = np.zeros(6)

        for i in range(len(y)):
            classes_rate[y[i]-1] += 1
            
        classes_rate /= len(y)

        return X, y, classes_rate
    
    if(train == False and raw == False):
        df_X = pd.read_csv("../data/UCI HAR Dataset/test/X_test.txt", sep="\s+", header=None)
        df_y = pd.read_csv("../data/UCI HAR Dataset/test/y_test.txt", names=["Class"])
        
        X = df_X.to_numpy()
        y = df_y.to_numpy()
        
        classes_rate = np.zeros(6)

        for i in range(len(y)):
            classes_rate[y[i]-1] += 1
            
        classes_rate /= len(y)

        return X, y, classes_rate
    
    if(train == True and raw == True):
        
        df_acc_x_train = pd.read_csv("../data/UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt", 
                                     sep="\s+", header=None)
        df_acc_y_train = pd.read_csv("../data/UCI HAR Dataset/train/Inertial Signals/body_acc_y_train.txt", 
                                     sep="\s+", header=None)
        df_acc_z_train = pd.read_csv("../data/UCI HAR Dataset/train/Inertial Signals/body_acc_z_train.txt", 
                                     sep="\s+", header=None)
        df_gyro_x_train = pd.read_csv("../data/UCI HAR Dataset/train/Inertial Signals/body_gyro_x_train.txt", 
                                      sep="\s+", header=None)
        df_gyro_y_train = pd.read_csv("../data/UCI HAR Dataset/train/Inertial Signals/body_gyro_y_train.txt", 
                                      sep="\s+", header=None)
        df_gyro_z_train = pd.read_csv("../data/UCI HAR Dataset/train/Inertial Signals/body_gyro_z_train.txt", 
                                      sep="\s+", header=None)
        
        df_X = pd.concat([df_acc_x_train,df_acc_y_train,df_acc_z_train,
                          df_gyro_x_train,df_gyro_y_train,df_gyro_z_train],axis=1)
        
        df_y = pd.read_csv("../data/UCI HAR Dataset/train/y_train.txt", names=["Class"])
        
        X = df_X.to_numpy()
        y = df_y.to_numpy()
        
        classes_rate = np.zeros(6)

        for i in range(len(y)):
            classes_rate[y[i]-1] += 1
            
        classes_rate /= len(y)

        return X, y, classes_rate
        
    
    if(train == False and raw == True):
        
        df_acc_x_test = pd.read_csv("../data/UCI HAR Dataset/test/Inertial Signals/body_acc_x_test.txt", 
                                     sep="\s+", header=None)
        df_acc_y_test = pd.read_csv("../data/UCI HAR Dataset/test/Inertial Signals/body_acc_y_test.txt", 
                                     sep="\s+", header=None)
        df_acc_z_test = pd.read_csv("../data/UCI HAR Dataset/test/Inertial Signals/body_acc_z_test.txt", 
                                     sep="\s+", header=None)
        df_gyro_x_test = pd.read_csv("../data/UCI HAR Dataset/test/Inertial Signals/body_gyro_x_test.txt", 
                                      sep="\s+", header=None)
        df_gyro_y_test = pd.read_csv("../data/UCI HAR Dataset/test/Inertial Signals/body_gyro_y_test.txt", 
                                      sep="\s+", header=None)
        df_gyro_z_test = pd.read_csv("../data/UCI HAR Dataset/test/Inertial Signals/body_gyro_z_test.txt", 
                                      sep="\s+", header=None)
        
        df_X = pd.concat([df_acc_x_test,df_acc_y_test,df_acc_z_test,
                          df_gyro_x_test,df_gyro_y_test,df_gyro_z_test],axis=1)
        
        df_y = pd.read_csv("../data/UCI HAR Dataset/test/y_test.txt", names=["Class"])
        
        X = df_X.to_numpy()
        y = df_y.to_numpy()
        
        classes_rate = np.zeros(6)

        for i in range(len(y)):
            classes_rate[y[i]-1] += 1
            
        classes_rate /= len(y)

        return X, y, classes_rate
        
    
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

def getJ_CE(y, X, W):
    y_hat, class_y_hat = softmax(X,W)
    J_CE = 0
    
    try:
        for i in range(y.shape[0]):
            for k in range(y.shape[1]):
                J_CE += y[i][k]*np.log(y_hat[i][k])
                
        J_CE /= (y.shape[0]*y.shape[1])
        
    except:
        y_hat = y_hat.T
        for k in range(y.shape[0]):
            J_CE += y[k]*np.log(y_hat[k])
            
        J_CE /= y.shape[0]
            
    return -(J_CE)
            
def getdJ_CEdW(y, y_hat, x):
    e = (y - y_hat)
    dJCE = (x.T*e).T
    
    return dJCE

def validateClassifier(y,X,W,classes_rate):
    y_hat, class_y_hat = softmax(X,W)
    
    hit = np.zeros(6)
    
    for i in range(len(y)):
         hit[y[i]-1] += 1 if y[i] == class_y_hat[i] else 0
    
    return np.average(hit/(classes_rate*len(y)))

def trainClassifier(X,y,epochs,batch,classes_rate):
    ba_train = [0]
    ba_val = [0]
    J_train = []
    J_val = []
    J = 0
    
    x_val = X[np.int16(len(y)*0.7):len(y)]
    y_val = y[np.int16(len(y)*0.7):len(y)]
    
    X = X[0:np.int16(len(y)*0.7)]
    y = y[0:np.int16(len(y)*0.7)]
    
    batch = len(y) if batch == 0 else batch
    
    W_len = epochs*(np.int16(np.ceil(len(y)/batch)))
    
    number_of_atributes = X.shape[1]
    
    W = np.zeros([W_len+1,NUMBER_OF_CLASSES,number_of_atributes])
    W[0] = np.random.rand(NUMBER_OF_CLASSES, number_of_atributes)/100 # Rand values in the interval (0, 0,01)
    
    y_ohe = oneHotEncoding(y=y)
    y_ohe_val = oneHotEncoding(y=y_val)
    
    r = list(range(len(y)))
    
    l = 0
    
    for k in range(epochs):
        
        dJCE = 0
        
        hit = np.zeros(NUMBER_OF_CLASSES)
        
        random.shuffle(r)
    
        j = 0

        for i in r:
            
            j += 1

            Xi = X[i, np.newaxis] 
            
            y_hat, class_y_hat = softmax(Xi,W[l])
            
            hit[y[i]-1] += 1 if y[i] == class_y_hat else 0
            
            dJCE += getdJ_CEdW(y=y_ohe[i],y_hat=y_hat,x=Xi)
            
            if(j == batch or i==r[-1]):
            
                W[l+1] = W[l] + STEP*dJCE/j
                    
                J_train.append(getJ_CE(y_ohe, X, W[l+1]))
                ba_train.append(np.average(hit/(classes_rate*j)))
                
                ba_val.append(validateClassifier(y_val,x_val,W[l],classes_rate))
                J_val.append(getJ_CE(y_ohe_val,x_val,W[l+1]))
        
                dJCE = 0
                hit = np.zeros(NUMBER_OF_CLASSES)
                j = 0   
                l += 1
        
    W = W[np.argmin(J_val)]
        
    #print(W)
    
    return W, ba_train, ba_val, J_train, J_val
    
def classify(x,W):
    y_hat, class_y_hat = softmax(x=x,w=W)
    return y_hat,class_y_hat

def rateModel(y,y_hat,classes_rate):
    
    hit = np.zeros(6)
    confusion_matrix = np.zeros([NUMBER_OF_CLASSES,NUMBER_OF_CLASSES])
    
    for i in range(len(y)):
        confusion_matrix[y[i]-1, y_hat[i]-1] += 1
        
        hit[y[i]-1] += 1 if y[i] == y_hat[i] else 0
            
    ba = np.average(hit/(classes_rate*len(y)))
        
    return confusion_matrix, ba

def confusionMatrixExtract(confusion_matrix):
    score = {1 : {'Precision': 0, 'Recall': 0}, 
             2 : {'Precision': 0, 'Recall': 0},
             3 : {'Precision': 0, 'Recall': 0},
             4 : {'Precision': 0, 'Recall': 0},
             5 : {'Precision': 0, 'Recall': 0},
             6 : {'Precision': 0, 'Recall': 0}}
    
    for i in range(NUMBER_OF_CLASSES):
        TP = FP = FN =0
        for j in range(NUMBER_OF_CLASSES):
            TP = confusion_matrix[i][j] if i == j else TP
            FP += confusion_matrix[i][j] if i != j else 0
            FN += confusion_matrix[j][i] if i != j else 0
            
        score[i+1]['Precision']= TP/(TP+FP)
        score[i+1]['Recall']= TP/(TP+FN)
    
    return score
    