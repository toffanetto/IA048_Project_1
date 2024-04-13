# @toffanetto

from os import X_OK
import numpy as np
import pandas as pd
import bisect

NUMBER_OF_CLASSES = 6
NUMBER_OF_ATRIBUTES = 561

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

    
def getDist(x,y):
    d = np.zeros(y.shape[0])
    
    for i in range(y.shape[0]):
        d[i] = np.sum(np.square(x - y[i]))
    
    return np.sqrt(d)

def getKNN(x, Y, K):
    kNN = []
    
    d = getDist(x,Y[0])
    
    for i in range(Y[1].shape[0]):
        
        bisect.insort(kNN, (d[i], Y[1][i]), key=lambda tup: tup[0])
        
        if(len(kNN)>K):
            kNN.pop()
            
    return kNN

def kNN_investigate(kNN):
    votes = np.zeros(NUMBER_OF_CLASSES)
    
    for i in range(len(kNN)):
        votes[kNN[i][1]-1] += 1
        
    return np.argmax(votes)+1
    
def classify(x, Y, y_label, k):
    
    y_hat = np.zeros(y_label.shape[0],dtype=np.int16)
    
    for i in range(x.shape[0]):
        kNN = getKNN(x[i], Y, K=k)
        
        class_kNN =  kNN_investigate(kNN)
        
        y_hat[i] = class_kNN
        
    return y_hat

def kFoldValidation(x,y,k,classes_rate):
    fold_len = np.int16(np.ceil(len(y)/k))
        
    x_fold = []
    y_fold = []
    
    k_voting = []
    ba_voting = []

    for i in range(k):
        try:
            x_fold.append(x[i*fold_len:(i+1)*fold_len])
            y_fold.append(y[i*fold_len:(i+1)*fold_len])
        except:
            x_fold.append(x[i*fold_len:len(y)])
            y_fold.append(y[i*fold_len:len(y)])
    
    for i in range(k):
        x_train = None
        y_train = None
        x_val = x_fold[i]
        y_val = y_fold[i]
        for j in range(k):
            if(j!=i):
                try:
                    x_train = np.concatenate((x_train,x_fold[j]))
                    y_train = np.concatenate((y_train,y_fold[j]))
                except:
                    x_train = x_fold[j]
                    y_train = y_fold[j]
                    
        k_i, ba_i = findBestK(x=x_val, Y=[x_train, y_train], y_label=y_val, classes_rate=classes_rate)
        k_voting.append(k_i)
        ba_voting.append(ba_i)
    
    return k_voting, ba_voting
    

def findBestK(x, Y, y_label, classes_rate):
    
    ba = np.zeros(30)
    
    for k in range(1,30):
    
        hit = np.zeros(NUMBER_OF_CLASSES)
        
        for i in range(x.shape[0]):
            kNN = getKNN(x[i], Y, K=k)
            
            class_kNN =  kNN_investigate(kNN)
            
            hit[class_kNN-1] += 1 if class_kNN == y_label[i] else 0

        ba[k] = np.average(hit/(classes_rate*x.shape[0]))
    
    k = np.argmax(ba)
    
    return k, ba

def rateModel(y,y_hat,classes_rate):
    
    hit = np.zeros(NUMBER_OF_CLASSES)
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
    