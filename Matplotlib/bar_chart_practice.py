import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

pokemon = pd.read_csv('./data/pokemon.csv')
# print(pokemon.head())

base_color = sb.color_palette()[0]
sb.countplot(data=pokemon, x='generation_id', color=base_color)

# plt.show()
plt.clf()

# creates a new dataframe that puts all of the type counts in a single column
pkmn_types = pokemon.melt(id_vars=['id', 'species'],
                          value_vars=['type_1', 'type_2'],
                          var_name='type_level', value_name='type').dropna()
# print(pkmn_types.head())

# get order of bars by frequency
type_counts = pkmn_types['type'].value_counts()
type_order = type_counts.index

# calculate proportion of the most common type
n_points = pkmn_types['species'].unique().shape[0]
n_max = type_counts[0]
max_rel = n_max / n_points

# define the label locations and names
tick_props = np.arange(0, max_rel, 0.02)
tick_names = ['{:0.2f}'.format(v) for v in tick_props]

sb.countplot(data=pkmn_types, y='type', color=base_color, order=type_order)
plt.xticks(tick_props * n_points, tick_names)
plt.xlabel('proportion')

# plt.show()
