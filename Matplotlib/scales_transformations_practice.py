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

# print(pokemon['height'].describe())
bins = np.arange(0, pokemon['height'].max()+0.2, 0.2)
plt.hist(data=pokemon, x='height', bins=bins)
plt.xlim((0, 4))
# plt.show()
plt.clf()

# np.log10(pokemon['weight'].describe())
bins = 10 ** np.arange(-1, 3+0.1, 0.1)
ticks = [.1, .3, 1, 3, 10, 30, 100, 300, 1000]
labels = ['{}'.format(v) for v in ticks]
plt.hist(data=pokemon, x='weight', bins=bins)
plt.xscale('log')
plt.xticks(ticks, labels)
plt.show()
