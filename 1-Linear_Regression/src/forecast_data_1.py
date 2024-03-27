# @toffanetto

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RATE_TRAINING_VALIDATION = 0.8

def trainingModel(data, k):
    y = data[k:len(data)]
    x = np.ones([len(y), k+1])

    for i in range(1,k+1):
        x[:, i] = data[i-1:len(y)+i-1]

    w = np.linalg.pinv(x).dot(y)
    
    return w

def testModel(data, k, w):
    y = data[k:len(data)]
    x = np.ones([len(y), k+1])

    for i in range(1,k+1):
        x[:, i] = data[i-1:len(y)+i-1]

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

###############################################################################
# Linear Regression - Least Squares

print('\n-> Training...')

#------------------------------------------------------------------------------
# Searching the best model

rms_error_t = np.zeros(24)
rms_error_v = np.zeros(24)

for k in range(1,25):

    #------------------------------------------------------------------------------
    # Dataset - Training and Validation

    tv_max = (2019-2003)*12 + 12

    t_max = np.int16(tv_max*RATE_TRAINING_VALIDATION)

    ds_t_flights = flights[0:t_max]
    ds_t_month = month[0:t_max]
    ds_t_year = year[0:t_max]

    ds_v_flights = flights[t_max-k:tv_max]
    ds_v_month = month[t_max-k:tv_max]
    ds_v_year = year[t_max-k:tv_max]

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

plt.savefig("./plot/b/RMSE_by_K.pdf", format="pdf", bbox_inches="tight")

#------------------------------------------------------------------------------
# Calculating and simulation of best model

#------------------------------------------------------------------------------
# Dataset - Training and Validation

tv_max = (2019-2003)*12 + 12

t_max = np.int16(tv_max*RATE_TRAINING_VALIDATION)

ds_t_flights = flights[0:t_max]
ds_t_month = month[0:t_max]
ds_t_year = year[0:t_max]

ds_v_flights = flights[t_max-k:tv_max]
ds_v_month = month[t_max-k:tv_max]
ds_v_year = year[t_max-k:tv_max]

ds_v_date = []

for i in range(k,len(ds_v_year)):
    if((i+1)%3 == 0):
        ds_v_date.append(str(ds_v_month[i])+'/'+str(ds_v_year[i])[2:4])
    else:
        ds_v_date.append('')

n = np.linspace(0,len(ds_v_date)-1,len(ds_v_date))

print('\nTraining dataset length: '+str(len(ds_t_flights)))
print('\nValidation dataset length: '+str(len(ds_v_flights)))

w = trainingModel(ds_t_flights,k)

y_v, y_v_hat, rms_error, map_error = testModel(ds_v_flights,k,w)


fig = plt.figure(figsize = (8,6))

mid = (fig.subplotpars.right + fig.subplotpars.left)/2
plt.plot(y_v, '.-b', label=r'$y(n)$')
plt.plot(y_v_hat, '.-r', label=r'$\hat{y}(n)$')
plt.xticks(n, ds_v_date, rotation='vertical')
plt.legend(loc='lower right')
plt.xlabel('Samples')
plt.ylabel('Nº of flights')
plt.suptitle('Validation of the model for K = '+str(k), x=mid)
plt.title('RMSE = '+str("{:.3f}".format(rms_error))+' | MAPE = '+str("{:.3f}".format(map_error*100))+' %', fontsize = 10)
plt.xlim([0,len(y_v)-1])
plt.grid()
plt.tight_layout()

plt.savefig("./plot/b/validation_best_K.pdf", format="pdf", bbox_inches="tight")

print('\nK = '+str(k))

print('\nRMSE of validation dataset = '+str("{:.3f}".format(rms_error))+'\n')

print('MAPE of validation dataset = '+str("{:.3f}".format(map_error*100))+' %\n')

#------------------------------------------------------------------------------
# Testing the model with Test Dataset

# Dataset - Test 1

test_min1 = (2020-2003)*12 

ds_test_flights = flights[test_min1-k:len(flights)]
ds_test_month = month[test_min1-k:len(month)]
ds_test_year = year[test_min1-k:len(year)]

ds_test_date = []

for i in range(k,len(ds_test_year)):
    if((i+1)%3 == 0):
        ds_test_date.append(str(ds_test_month[i])+'/'+str(ds_test_year[i])[2:4])
    else:
        ds_test_date.append('')

n = np.linspace(0,len(ds_test_date)-1,len(ds_test_date))

y_test, y_test_hat, rms_error, map_error = testModel(ds_test_flights,k,w)

fig = plt.figure(figsize = (8,6))

mid = (fig.subplotpars.right + fig.subplotpars.left)/2
plt.plot(y_test, '.-b', label=r'$y(n)$')
plt.plot(y_test_hat, '.-r', label=r'$\hat{y}(n)$')
plt.legend(loc='lower right')
plt.xticks(n, ds_test_date, rotation='vertical')
plt.xlabel('Samples')
plt.ylabel('Nº of flights')
plt.suptitle('Test 2020~2023 of the model for K = '+str(k), x=mid)
plt.title('RMSE = '+str("{:.3f}".format(rms_error))+' | MAPE = '+str("{:.3f}".format(map_error*100))+' %', fontsize = 10)
plt.xlim([0,len(y_test)-1])
plt.grid()
plt.tight_layout()

plt.savefig("./plot/b/test_best_K.pdf", format="pdf", bbox_inches="tight")

print('\nK = '+str(k))

print('\nRMSE of test dataset = '+str("{:.3f}".format(rms_error))+'\n')

print('MAPE of test dataset = '+str("{:.3f}".format(map_error*100))+' %\n')

#------------------------------------------------------------------------------
# Testing the model with Test 2 Dataset

# Dataset - Test 2

test_min2 = (2022-2003)*12
ds_test2_flights = flights[test_min2-k:len(flights)]
ds_test2_month = month[test_min2-k:len(month)]
ds_test2_year = year[test_min2-k:len(year)]

ds_test2_date = []

for i in range(k,len(ds_test2_year)):
    if((i+1)%3 == 0):
        ds_test2_date.append(str(ds_test2_month[i])+'/'+str(ds_test2_year[i])[2:4])
    else:
        ds_test2_date.append('')

n = np.linspace(0,len(ds_test2_date)-1,len(ds_test2_date))

y_test2, y_test2_hat, rms_error, map_error = testModel(ds_test2_flights,k,w)

fig = plt.figure(figsize = (8,6))

mid = (fig.subplotpars.right + fig.subplotpars.left)/2
plt.plot(y_test2, '.-b', label=r'$y(n)$')
plt.plot(y_test2_hat, 'r', label=r'$\hat{y}(n)$')
plt.legend(loc='lower right')
plt.xticks(n, ds_test2_date, rotation='vertical')
plt.xlabel('Samples')
plt.ylabel('Nº of flights') 
plt.suptitle('Test 2022~2023 of the model for K = '+str(k), x=mid)
plt.title('RMSE = '+str("{:.3f}".format(rms_error))+' | MAPE = '+str("{:.3f}".format(map_error*100))+' %', fontsize = 10)
plt.xlim([0,len(y_test2)-1])
plt.grid()

plt.savefig("./plot/b/test2_best_K.pdf", format="pdf", bbox_inches="tight")

print('\nK = '+str(k))

print('\nRMSE of test 2 dataset = '+str("{:.3f}".format(rms_error))+'\n')

print('MAPE of test 2 dataset = '+str("{:.3f}".format(map_error*100))+' %\n')


plt.tight_layout()

plt.show()

    

