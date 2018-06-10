import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

fuel_econ = pd.read_csv('./data/fuel_econ.csv')
# print(fuel_econ.head())

plt.scatter(data=fuel_econ, x='city', y='highway', alpha=0.2)
# plt.show()
plt.clf()

# print(fuel_econ['displ'].describe())
# print(fuel_econ['co2'].describe())
x_bins = np.arange(0.6, fuel_econ['displ'].max()+0.4, 0.4)
y_bins = np.arange(0, fuel_econ['co2'].max()+50, 50)

plt.hist2d(data=fuel_econ, x='displ', y='co2', 
           bins=[x_bins, y_bins],
           cmap='viridis_r',  # change colors from light to dark
           cmin=0.5)  # don't color values below this
plt.colorbar()  # side bar
plt.xlabel('Displacement (l)')
plt.ylabel('CO2 (g/mi)')
plt.show()
# plt.clf()
