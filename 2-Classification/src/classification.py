# @toffanetto

import pandas as pd

def getData(train, raw):
    
    if(train == True and raw == True):
        df_X = pd.read_csv("./data/UCI\ HAR\ Dataset/train/X_train.txt", sep=" ")
        df_y = pd.read_csv("./data/UCI\ HAR\ Dataset/train/y_train.txt", sep=" ")
    
    if(train == False and raw):
        df_X = pd.read_csv("./data/UCI\ HAR\ Dataset/test/X_test.txt", sep=" ")
        df_y = pd.read_csv("./data/UCI\ HAR\ Dataset/test/y_test.txt", sep=" ")