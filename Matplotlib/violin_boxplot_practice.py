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

# we need to order the sedan classes by categorical size, so that the
# visualization makes more sense
sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars',
                 'Midsize Cars', 'Large Cars']
v_classes = pd.api.types.CategoricalDtype(ordered=True,
                                          categories=sedan_classes)
fuel_econ['VClass'] = fuel_econ['VClass'].astype(v_classes)

base_color = sb.color_palette()[0]
sb.violinplot(data=fuel_econ, x='VClass', y='displ', color=base_color)
plt.xticks(rotation=10)
plt.xlabel('Vehicle Class')
plt.ylabel('Displacement')
plt.show()
