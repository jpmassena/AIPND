import numpy as np
import pandas as pd
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# array of money
money = np.array([70, 100, 20, 80, 40, 70, 60, 100])
currency_label = ["USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "HKD"]

# Creates pandas dataframe with column labels(currency_label) from the numpy
# vector for printing.
money_df = pd.DataFrame(data=money, index=currency_label, columns=["Amounts"])

print(money_df.T)

# Imports conversion rates(weights) matrix as a pandas dataframe.
conversion_rates_df = pd.read_csv(
    './currencyConversionMatrix.csv', header=0, index_col=0)

# Creates numpy matrix from a pandas dataframe to create the conversion
# rates(weights) matrix.
conversion_rates = conversion_rates_df.values

print(conversion_rates_df)

# money_totals = np.matmul(money.T, conversion_rates)
money_totals = money.T.dot(conversion_rates)

# Converts the resulting money totals vector into a dataframe for printing.
money_totals_df = pd.DataFrame(data=money_totals, index=currency_label,
                               columns=["Money Totals"])
print(money_totals_df.T)
