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

ds_tv_flights = flights[0:204]
ds_tv_month = month[0:204]
ds_tv_year = year[0:204]
ds_tv_n = np.linspace(0, 203, len(ds_tv_flights))

# Dataset - Test

ds_test_flights = flights[205:len(flights)]
ds_test_month = month[205:len(month)]
ds_test_year = year[205:len(month)]

# Linear Regression - Least Squares

##

