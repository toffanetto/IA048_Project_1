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

years = []
date = []

for i in range(len(year)):
    date.append(str(month[i])+'/'+str(year[i])[2:4])
    if month[i] == 1:
        years.append(str(year[i]))
    else:
        years.append('')

y = np.linspace(0, 252, int(252/12)+1)

print(y)

# Plot entire time series
plt.figure(figsize = (10,6))
plt.xticks(n, years, rotation='vertical')
plt.xticks(n, years, rotation='vertical', minor=True)
plt.plot(n, flights)
plt.xlabel('Time')
plt.ylabel('Number of flights')
plt.xlim([0, len(flights)-1])
plt.ylim([0.2e6, 1e6])
plt.grid(axis='y')
plt.vlines(y, 0, 1e6, colors='tab:gray', linewidth=0.5)
plt.title('Total number of flights in US')

plt.savefig("./plot/time_series.pdf", format="pdf", bbox_inches="tight")

# Plot entire with divisions time series
plt.figure(figsize = (10,6))
plt.xticks(n, years, rotation='vertical')
plt.plot(n, flights)
plt.xlabel('Time')
plt.ylabel('Number of flights')
plt.xlim([0, len(flights)-1])
plt.ylim([0.2e6, 1e6])
plt.grid(axis='y')
plt.vlines(y, 0, 1e6, colors='tab:gray', linewidth=0.5)
plt.vlines(((2008-2003)*12+8), 0, 1e6, colors='tab:red', linewidth=2)
plt.vlines(((2019-2003)*12+11), 0, 1e6, colors='tab:red', linewidth=2)
plt.title('Total number of flights in US')

plt.savefig("./plot/time_series_divided.pdf", format="pdf", bbox_inches="tight")

plt.tight_layout()

plt.show()
