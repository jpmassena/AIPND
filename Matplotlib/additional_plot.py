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

most_makes = fuel_econ['make'].value_counts().index[:18]
fuel_econ_sub = fuel_econ.loc[fuel_econ['make'].isin(most_makes)]

make_means = fuel_econ_sub.groupby('make').mean()
comb_order = make_means.sort_values('comb', ascending=False).index
g = sb.FacetGrid(data=fuel_econ_sub, col='make', col_wrap=6, size=2,
                 col_order=comb_order)
# try sb.distplot instead of plt.hist to see the plot in terms of density!
g.map(plt.hist, 'comb', bins=np.arange(12, fuel_econ_sub['comb'].max()+2, 2))
g.set_titles('{col_name}')
# plt.show()
plt.clf()

base_color = sb.color_palette()[0]
sb.barplot(data=fuel_econ_sub, x='comb', y='make',
           color=base_color, order=comb_order, ci='sd')
plt.show()
