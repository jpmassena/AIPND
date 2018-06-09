import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# load Google stock
google_stock = pd.read_csv(
    './GOOG.csv',
    index_col=['Date'],  # column that will be used as index
    parse_dates=True,  # parse dates from strings
    usecols=['Date', 'Adj Close'])  # columns to load

# load Apple stock
apple_stock = pd.read_csv(
    './AAPL.csv',
    index_col=['Date'],
    parse_dates=True,
    usecols=['Date', 'Adj Close'])

# load Amazon stock
amazon_stock = pd.read_csv(
    './AMZN.csv',
    index_col=['Date'],
    parse_dates=True,
    usecols=['Date', 'Adj Close'])

print(google_stock.head())

# create calendar dates between '2000-01-01' and  '2016-12-31'
dates = pd.date_range('2000-01-01', '2016-12-31')

# create and empty DataFrame that uses the above dates as indices
all_stocks = pd.DataFrame(index=dates)

print(all_stocks.head())

# change the Adj Close column label to Google
google_stock = google_stock.rename(columns={"Adj Close": "Google"})

# change the Adj Close column label to Apple
apple_stock = apple_stock.rename(columns={"Adj Close": "Apple"})

# change the Adj Close column label to Amazon
amazon_stock = amazon_stock.rename(columns={"Adj Close": "Amazon"})

print(google_stock.head())

# join the Google stock to all_stocks
all_stocks = all_stocks.join(google_stock)

# join the Google stock to all_stocks
all_stocks = all_stocks.join(apple_stock)

# join the Google stock to all_stocks
all_stocks = all_stocks.join(amazon_stock)

print(all_stocks.head())

# check if there are any NaN values in the all_stocks dataframe
nans = all_stocks.isnull().sum().sum()
print('\nNumber of NaN values: ', nans)

# remove any rows that contain NaN values
if nans > 0:
    all_stocks.dropna(axis=0, inplace=True)

# Print the average stock price for each stock
print("\nMean of stocks: \n", all_stocks.mean())

# Print the median stock price for each stock
print("\nMedian of stocks: \n", all_stocks.median())

# Print the standard deviation of the stock price for each stock
print("\nStandard Deviation of stocks: \n", all_stocks.std())

# Print the correlation between stocks
print("\nCorrelation between stocks: \n", all_stocks.corr())

# compute the rolling mean using a 150-Day window for Google stock
rollingMean = all_stocks['Google'].rolling(150).mean()

print("\nGoogle's stock rolling mean for 150 days: \n", rollingMean)

# plot the Google stock data
plt.plot(all_stocks['Google'])

# plot the rolling mean ontop of our Google stock data
plt.plot(rollingMean)
plt.legend(['Google Stock Price', 'Rolling Mean'])
plt.show()