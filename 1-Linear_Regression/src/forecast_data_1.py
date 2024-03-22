# @toffanetto

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_csv = pd.read_csv("./data/air_traffic.csv", thousands=',')

data_flights = pd.DataFrame(data_csv, columns=["Year", "Month", "Flt"])

print(data_flights)

flights = data_flights["Flt"].to_numpy()

n = np.linspace(0,len(flights),len(flights))

year = data_flights["Year"].to_numpy()

month = data_flights["Month"].to_numpy()

# Dataset - Training and Validation

rate_train_validation = 0.7

t_max = np.int16(204*rate_train_validation)

ds_t_flights = flights[0:t_max]
ds_t_month = month[0:t_max]
ds_t_year = year[0:t_max]
ds_t_n = np.linspace(0, len(ds_t_flights)-1, len(ds_t_flights))

ds_v_flights = flights[t_max+1:204]
ds_v_month = month[t_max+1:204]
ds_v_year = year[t_max+1:204]
ds_v_n = np.linspace(144, 203, len(ds_v_flights))

# Dataset - Test

ds_test_flights = flights[205:len(flights)]
ds_test_month = month[205:len(month)]
ds_test_year = year[205:len(year)]
ds_test_n = np.linspace(0, len(ds_test_flights), len(ds_test_flights))

# Linear Regression - Least Squares


rms_error = np.zeros(24)

for k in range(1,25):
    y_t = ds_t_flights[k+1:len(ds_t_flights)]
    y_v = ds_v_flights[k+1:len(ds_v_flights)]

    x_t = np.ones([len(y_t), k+1])
    x_v = np.ones([len(y_v), k+1])

    for i in range(1,k+1):
        x_t[:, i] = ds_t_flights[i:len(y_t)+i]
        x_v[:, i] = ds_v_flights[i:len(y_v)+i]

    w = np.linalg.pinv(x_t).dot(y_t)

    #print(w)

    y_v_hat = x_v.dot(w)

    rms_error[i-1] = np.sqrt(np.average(np.square(np.subtract(y_v,y_v_hat))))

    # plt.figure(figsize = (10,8))
    # plt.plot(y_v, 'b')
    # plt.plot(y_v_hat, 'r')
    # plt.xlabel('x')
    # plt.ylabel('y')

k_n = np.linspace(1, len(rms_error), len(rms_error), dtype=np.uint8)

k = np.argmin(rms_error) + 1

plt.figure(figsize = (8,6))
plt.plot(k_n,rms_error)
plt.plot(k,rms_error[k-1], 'r.')
plt.xticks(k_n,k_n)
plt.grid()
plt.title('Root Mean Square Error - RMSE')
plt.xlim([1,24])
plt.xlabel('Number of past samples (K)')
plt.ylabel('Nº of flights')

plt.savefig("./plot/RMSE_by_K.pdf", format="pdf", bbox_inches="tight")

y_t = ds_t_flights[k+1:len(ds_t_flights)]
y_v = ds_v_flights[k+1:len(ds_v_flights)]

x_t = np.ones([len(y_t), k+1])
x_v = np.ones([len(y_v), k+1])

for i in range(1,k+1):
    x_t[:, i] = ds_t_flights[i:len(y_t)+i]
    x_v[:, i] = ds_v_flights[i:len(y_v)+i]

w = np.linalg.pinv(x_t).dot(y_t)

#print(w)

y_v_hat = x_v.dot(w)

rms_error = np.sqrt(np.average(np.square(np.subtract(y_v,y_v_hat))))

map_error = np.average(np.abs(np.divide(np.subtract(y_v,y_v_hat),y_v)))

plt.figure(figsize = (8,6))
plt.plot(y_v, 'b', label='Real output')
plt.plot(y_v_hat, 'r', label='Predicted output')
plt.legend(loc='upper right')
plt.xlabel('Samples')
plt.ylabel('Nº of flights')
plt.title('Validation of the model for K = '+str(k))
plt.xlim([1,24])
plt.grid()

plt.savefig("./plot/validation_best_K.pdf", format="pdf", bbox_inches="tight")

print('\n\nK = '+str(k))

print('\nRMSE = '+str(rms_error)+'\n')

print('MAPE = '+str(map_error*100)+' %\n')

plt.show()

    

