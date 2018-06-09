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

bins = np.arange(0, pokemon['special-defense'].max()+5, 5)
plt.hist(x=pokemon['special-defense'], bins=bins)

plt.show()
