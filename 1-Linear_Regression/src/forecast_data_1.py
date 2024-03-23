# @toffanetto

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def trainingModel(data, k):
    y = data[k+1:len(data)]
    x = np.ones([len(y), k+1])

    for i in range(1,k+1):
        x[:, i] = data[i:len(y)+i]

    w = np.linalg.pinv(x).dot(y)
    
    return w

def testModel(data, k, w):
    y = data[k+1:len(data)]

    x = np.ones([len(y), k+1])

    for i in range(1,k+1):
        x[:, i] = data[i:len(y)+i]

    y_hat = x.dot(w)

    rms_error = np.sqrt(np.average(np.square(np.subtract(y,y_hat))))

    map_error = np.average(np.abs(np.divide(np.subtract(y,y_hat),y)))

    return y, y_hat, rms_error, map_error

###############################################################################
# Read data

data_csv = pd.read_csv("./data/air_traffic.csv", thousands=',')

data_flights = pd.DataFrame(data_csv, columns=["Year", "Month", "Flt"])

print(data_flights)

flights = data_flights["Flt"].to_numpy()

n = np.linspace(0,len(flights),len(flights))

year = data_flights["Year"].to_numpy()

month = data_flights["Month"].to_numpy()

#------------------------------------------------------------------------------
# Dataset - Training and Validation

rate_train_validation = 0.8

t_max = np.int16(204*rate_train_validation)

ds_t_flights = flights[0:t_max]
ds_t_month = month[0:t_max]
ds_t_year = year[0:t_max]
ds_t_n = np.linspace(0, len(ds_t_flights)-1, len(ds_t_flights))

ds_v_flights = flights[t_max+1:204]
ds_v_month = month[t_max+1:204]
ds_v_year = year[t_max+1:204]
ds_v_n = np.linspace(144, 203, len(ds_v_flights))

print('\nTraining dataset length: '+str(len(ds_t_flights)))
print('\nValidation dataset length: '+str(len(ds_v_flights)))

#------------------------------------------------------------------------------
# Dataset - Test

ds_test_flights = flights[205:len(flights)]
ds_test_month = month[205:len(month)]
ds_test_year = year[205:len(year)]
ds_test_n = np.linspace(0, len(ds_test_flights), len(ds_test_flights))

ds_test2_flights = flights[len(flights)-21:len(flights)]
ds_test2_month = month[len(month)-21:len(month)]
ds_test2_year = year[len(year)-21:len(year)]

###############################################################################
# Linear Regression - Least Squares

print('\n-> Training...')

#------------------------------------------------------------------------------
# Searching the best model

rms_error_t = np.zeros(24)
rms_error_v = np.zeros(24)

for k in range(1,25):

    w = trainingModel(ds_t_flights,k)

    y_t, y_t_hat, rms_error_t[k-1] = testModel(ds_t_flights,k,w)[0:3]
    y_v, y_v_hat, rms_error_v[k-1] = testModel(ds_v_flights,k,w)[0:3]

    # plt.figure(figsize = (10,8))
    # plt.plot(y_v, 'b')
    # plt.plot(y_v_hat, 'r')
    # plt.xlabel('x')
    # plt.ylabel('y')

k_n = np.linspace(1, len(rms_error_v), len(rms_error_v), dtype=np.uint8)

k = np.argmin(rms_error_v) + 1

plt.figure(figsize = (8,6))
plt.plot(k_n,rms_error_v, '.-',label='RMSE Validation')
plt.plot(k_n,rms_error_t, '-.', label='RMSE Training')
plt.plot(k,rms_error_v[k-1], 'r.', label='min(RMSE)')
plt.legend(loc='upper right')
plt.xticks(k_n,k_n)
plt.grid()
plt.title('Root Mean Square Error - RMSE')
plt.xlim([1,24])
plt.xlabel('Number of predictor inputs (K)')
plt.ylabel('Nº of flights')

plt.savefig("./plot/RMSE_by_K.pdf", format="pdf", bbox_inches="tight")

#------------------------------------------------------------------------------
# Calculating and simulation of best model

w = trainingModel(ds_t_flights,k)

y_v, y_v_hat, rms_error, map_error = testModel(ds_v_flights,k,w)

plt.figure(figsize = (8,6))
plt.plot(y_v, 'b', label=r'$y(n)$')
plt.plot(y_v_hat, 'r', label=r'$\hat{y}(n)$')
plt.legend(loc='upper right')
plt.xlabel('Samples')
plt.ylabel('Nº of flights')
plt.suptitle('Validation of the model for K = '+str(k))
plt.title('RMSE = '+str("{:.3f}".format(rms_error))+' | MAPE = '+str("{:.3f}".format(map_error*100))+' %', fontsize = 10)
plt.xlim([0,len(y_v)-1])
plt.grid()

plt.savefig("./plot/validation_best_K.pdf", format="pdf", bbox_inches="tight")

print('\nK = '+str(k))

print('\nRMSE of validation dataset = '+str("{:.3f}".format(rms_error))+'\n')

print('MAPE of validation dataset = '+str("{:.3f}".format(map_error*100))+' %\n')

#------------------------------------------------------------------------------
# Testing the model with Test Dataset

y_test, y_test_hat, rms_error, map_error = testModel(ds_test_flights,k,w)

plt.figure(figsize = (8,6))
plt.plot(y_test, 'b', label=r'$y(n)$')
plt.plot(y_test_hat, 'r', label=r'$\hat{y}(n)$')
plt.legend(loc='upper right')
plt.xlabel('Samples')
plt.ylabel('Nº of flights')
plt.suptitle('Test 2020~2023 of the model for K = '+str(k))
plt.title('RMSE = '+str("{:.3f}".format(rms_error))+' | MAPE = '+str("{:.3f}".format(map_error*100))+' %', fontsize = 10)
plt.xlim([0,len(y_test)-1])
plt.grid()

plt.savefig("./plot/test_best_K.pdf", format="pdf", bbox_inches="tight")

print('\nK = '+str(k))

print('\nRMSE of test dataset = '+str("{:.3f}".format(rms_error))+'\n')

print('MAPE of test dataset = '+str("{:.3f}".format(map_error*100))+' %\n')

#------------------------------------------------------------------------------
# Testing the model with Test 2 Dataset

y_test2, y_test2_hat, rms_error, map_error = testModel(ds_test2_flights,k,w)

plt.figure(figsize = (8,6))
plt.plot(y_test2, 'b', label=r'$y(n)$')
plt.plot(y_test2_hat, 'r', label=r'$\hat{y}(n)$')
plt.legend(loc='upper right')
plt.xlabel('Samples')
plt.ylabel('Nº of flights')
plt.suptitle('Test 2022~2023 of the model for K = '+str(k))
plt.title('RMSE = '+str("{:.3f}".format(rms_error))+' | MAPE = '+str("{:.3f}".format(map_error*100))+' %', fontsize = 10)
plt.xlim([0,len(y_test2)-1])
plt.grid()

plt.savefig("./plot/test2_best_K.pdf", format="pdf", bbox_inches="tight")

print('\nK = '+str(k))

print('\nRMSE of test 2 dataset = '+str("{:.3f}".format(rms_error))+'\n')

print('MAPE of test 2 dataset = '+str("{:.3f}".format(map_error*100))+' %\n')

plt.show()

    

