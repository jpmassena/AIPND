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

sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars',
                 'Midsize Cars', 'Large Cars']
v_classes = pd.api.types.CategoricalDtype(ordered=True,
                                          categories=sedan_classes)
fuel_econ['VClass'] = fuel_econ['VClass'].astype(v_classes)
fuel_econ_sub = fuel_econ.loc[fuel_econ['fuelType'].isin(['Premium Gasoline',
                                                          'Regular Gasoline'])]

sb.countplot(data=fuel_econ_sub, x='VClass', hue='fuelType')
plt.xticks(rotation=10)
plt.show()
