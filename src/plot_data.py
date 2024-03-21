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
plt.plot(flights)
plt.xlabel('Time')
plt.ylabel('Number of flights')
plt.xlim([0, len(flights)-1])
plt.ylim([0.2e6, 1e6])
plt.grid(axis='y')
plt.vlines(y, 0, 1e6, colors='tab:gray', linewidth=0.5)
plt.title('Total number of flights in US')

plt.savefig("./plot/time_series.pdf", format="pdf", bbox_inches="tight")

# Comparison of many years
fig, ax = plt.subplots(nrows=3, ncols=1)
plt.sca(ax[0])
plt.xticks(n,date)
plt.grid(axis='x')
plt.sca(ax[1])
plt.xticks(n,date)
plt.grid(axis='x')
plt.sca(ax[2])
plt.xticks(n,date)
plt.grid(axis='x')

ax[0].plot(flights)
ax[0].set_xlim([0, 11])
ax[0].set_ylim([np.min(flights[0:11])*0.9, np.max(flights[0:11])*1.1])

ax[1].plot(flights)
ax[1].set_xlim([12, 23])
ax[1].set_ylim([np.min(flights[12:23])*0.9, np.max(flights[12:23])*1.1])

ax[2].plot(flights)
ax[2].set_xlim([24, 35])
ax[2].set_ylim([np.min(flights[24:35])*0.9, np.max(flights[24:35])*1.1])

for a in ax.flat:
    a.set(xlabel='Time', ylabel='Nº of flights')

fig.suptitle('Comparison of number of flights for different years')

plt.tight_layout()

plt.savefig("./plot/comparison.pdf", format="pdf", bbox_inches="tight")

plt.show()
