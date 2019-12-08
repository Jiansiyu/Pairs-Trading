
import sys
from os.path import dirname
sys.path.append(dirname('/home/newdriver/Storage/HomeDir/Learning/Python/Pairs-Trading/Data-Analysis/stocker'))
sys.path.append(dirname('/home/newdriver/Learning/Python/FinalProject'))

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import statsmodels
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

matplotlib.use('Agg')
# Turn interactive plotting off
plt.ioff()

np.random.seed(107) # So that you can get the same random numbers as me

"""Let's keep this simple and create two fake securities with returns drawn from a normal distribution and created with a random walk"""

# Generate daily returns

Xreturns = np.random.normal(0, 1, 100)

# sum up and shift the prices up

X = pd.Series(np.cumsum(
    Xreturns), name='X') + 50
X.plot(figsize=(15,7))
plt.show()

"""For the sake of the illustration and intuition, we will generate Y to have a clear link with X, so the price of Y should very in a similar way to X. What we can do is just take X and shift it up slightly and add some noise from a normal distribution."""

noise = np.random.normal(0, 1, 100)
Y = X + 5 + noise
Y.name = 'Y'

pd.concat([X, Y], axis=1).plot(figsize=(15, 7))

plt.show()

(Y/X).plot(figsize=(15,7))

plt.axhline((Y/X).mean(), color='red', linestyle='--')

plt.xlabel('Time')
plt.legend(['Price Ratio', 'Mean'])
plt.show()
score, pvalue, _ = coint(X,Y)
print(pvalue)

ret1 = np.random.normal(1, 1, 100)
ret2 = np.random.normal(2, 1, 100)

s1 = pd.Series(np.cumsum(ret1), name='X_divering')
s2 = pd.Series(np.cumsum(ret2), name='Y_diverging')

pd.concat([s1, s2], axis=1).plot(figsize=(15, 7))
plt.show()

print('Correlation: ' + str(s1.corr(s2)))
score, pvalue, _ = coint(s1, s2)
print('Cointegration test p-value: ' + str(pvalue))
Y2 = pd.Series(np.random.normal(0, 1, 800), name='Y2') + 20
Y3 = Y2.copy()

Y3[0:100] = 30
Y3[100:200] = 10
Y3[200:300] = 30
Y3[300:400] = 10
Y3[400:500] = 30
Y3[500:600] = 10
Y3[600:700] = 30
Y3[700:800] = 10

Y2.plot(figsize=(15,7))
Y3.plot()
plt.ylim([0,40])
plt.show()
# very low correlation
print('Correlation: ' + str(Y2.corr(Y3)))
score, pvalue, _ = coint(Y2, Y3)
print('Cointegration test p-value: ' + str(pvalue))

"""## How to Actually make a Pairs Trade

Now that we've clearly explained the essence of pair trading and the concept of cointegration, it's time to get to the nitty-gritty.

We know that if two time series are cointegrated, they will drift towards and apart from each other around the mean. We can be confident that if the two series start to diverge, they will eventually converge later. 

When the series diverge from one another, we say that the *spread* is high. When they drift back towards each other, we say that the *spread* is low. We need to buy one security and short the other. But which ones? 

Remember the equation we had? 

*Y = αX + e* 

As the ratio (Y/X) moves around the mean α, we watch for when X and Y are far apart, which is when α is either too high or too low. Then, when the ratio of the series moves back toward each other, we make money.

In general, we **long the security that is underperforming** and **short the security that is overperforming.**

In terms of the equation, when α is smaller than usual, that means that Y is underperforming and X is overperforming, so we buy Y and sell X.

When α is larger than usual, we sell Y and buy X.

## Testing on Historical Data

Now let's find some actual securities that are cointegrated based on historical data.
"""

def find_cointegrated_pairs(data):
  n = data.shape[1]
  score_matrix = np.zeros((n, n))
  pvalue_matrix = np.ones((n, n))
  keys = data.keys()
  pairs = [] # We store the stock pairs that are likely to be cointegrated
  for i in range(n):
    for j in range(i+1, n):
      S1 = data[keys[i]]
      S2 = data[keys[j]]
      result = coint(S1, S2)
      score = result[0] # t-score
      pvalue = result[1]
      score_matrix[i,j] = score
      pvalue_matrix[i, j] = pvalue
      if pvalue < 0.02:
        pairs.append((keys[i], keys[j]))
  return score_matrix, pvalue_matrix, pairs

'''
Read the realdata into the datafame

'''

import stocker
from stocker.stocker import Stocker

data = pd.DataFrame()

#stocks = ['AAPL', 'ADBE', 'SYMC', 'EBAY', 'MSFT', 'QCOM', 'HPQ', 'JNPR', 'AMD', 'IBM']
'''
changed by siyu
Get the nasdaq NDX 100 top30 stocks
'''
from FinalProject.DataAcquisition import StockDataAcquire
stocks = StockDataAcquire().GetNasdaq100IndexNDX()[0:10]



apple = Stocker('AAPL')
df = apple.make_df('1990-12-12', '2016-12-12')
df = df.set_index(['Date'])
apple_closes = df['Adj. Close']

df.head()
apple_closes.head()

for ticker in stocks:
  name = str(ticker)
  print(name)
  s = Stocker(name)
  df = s.make_df('2000-12-12', '2016-12-12')
  df = df.set_index(['Date'])
  data[name] = df['Adj. Close']

data.head(50)


from pandas_datareader import data as pdr
import datetime
#import fix_yahoo_finance as yf
import yfinance as yf

start_sp = datetime.datetime(2000, 12, 12)
end_sp = datetime.datetime(2016, 12, 12)
yf.pdr_override()
sp500 = pdr.get_data_yahoo('^GSPC',
                           start_sp, end_sp)

prices = pd.DataFrame()
prices['SP500'] = sp500['Adj Close']

prices.head()

all_prices = pd.merge(prices, data, left_index=True, right_index=True)

all_prices.head()
stocks = stocks + ['SP500']
"""Now that we've got our data, let's try to find some cointegrated pairs."""

# Creating a heatmap to show the p-values of the cointegration test

scores, pvalues, pairs = find_cointegrated_pairs(all_prices)
import seaborn
m = [0, 0.2, 0.4, 0.6, 0.8, 1]
seaborn.heatmap(pvalues, xticklabels=stocks,
               yticklabels=stocks, cmap='RdYlGn_r',
               mask = (pvalues >= 0.98))
plt.show()
print(pairs)

"""According to this heatmap which plots the various p-values for all of the pairs, we've got 4 pairs that appear to be cointegrated. Let's plot their ratios on a graph to see what's going on."""

# trade using a simpe strategy
'''
trade function, in this function need to return the time serial data that contains the earning in each step
'''
def trade1(S1, S2, window1, window2):
    
    # If window length is 0, algorithm doesn't make sense, so exit
    if (window1 == 0) or (window2 == 0):
        return 0
    
    # Compute rolling mean and rolling standard deviation
    ratios = S1/S2
    ma1 = ratios.rolling(window=window1,
                               center=False).mean()
    ma2 = ratios.rolling(window=window2,
                               center=False).mean()
    std = ratios.rolling(window=window2,
                        center=False).std()
    zscore = (ma1 - ma2)/std
    # Simulate trading
    # Start with no money and no positions
    money = 0
    countS1 = 0
    countS2 = 0

    returnSerial=zscore.copy()
    for i in range(len(ratios)):
        # Sell short if the z-score is > 1
        if zscore[i] < -1:
            money += S1[i] - S2[i] * ratios[i]
            countS1 -= 1
            countS2 += ratios[i]
            #print('Selling Ratio %s %s %s %s'%(money, ratios[i], countS1,countS2))
        # Buy long if the z-score is < -1
        elif zscore[i] > 1:
            money -= S1[i] - S2[i] * ratios[i]
            countS1 += 1
            countS2 -= ratios[i]
            #print('Buying Ratio %s %s %s %s'%(money,ratios[i], countS1,countS2))
        # Clear positions if the z-score between -.5 and .5
        elif abs(zscore[i]) < 0.75:
            money += S1[i] * countS1 + S2[i] * countS2
            countS1 = 0
            countS2 = 0
            #print('Exit pos %s %s %s %s'%(money,ratios[i], countS1,countS2))
        returnSerial[i]=money
    return money, returnSerial

# Trade using a simple strategy
def trade(S1, S2, window1, window2):
    
    # If window length is 0, algorithm doesn't make sense, so exit
    if (window1 == 0) or (window2 == 0):
        return 0
    
    # Compute rolling mean and rolling standard deviation
    ratios = S1/S2
    ma1 = ratios.rolling(window=window1,
                               center=False).mean()
    ma2 = ratios.rolling(window=window2,
                               center=False).mean()
    std = ratios.rolling(window=window2,
                        center=False).std()
    zscore = (ma1 - ma2)/std
    
    # Simulate trading
    # Start with no money and no positions
    money = 0
    countS1 = 0
    countS2 = 0
    print(ratios)
    returnSerial=zscore.copy()
    for i in range(len(ratios)):
        # Sell short if the z-score is > 1
        if zscore[i] < -1:
            money += S1[i] - S2[i] * ratios[i]
            countS1 -= 1
            countS2 += ratios[i]
            #print('Selling Ratio %s %s %s %s'%(money, ratios[i], countS1,countS2))
        # Buy long if the z-score is < -1
        elif zscore[i] > 1:
            money -= S1[i] - S2[i] * ratios[i]
            countS1 += 1
            countS2 -= ratios[i]
            #print('Buying Ratio %s %s %s %s'%(money,ratios[i], countS1,countS2))
        # Clear positions if the z-score between -.5 and .5
        elif abs(zscore[i]) < 0.75:
            money += S1[i] * countS1 + S2[i] * countS2
            countS1 = 0
            countS2 = 0
            #print('Exit pos %s %s %s %s'%(money,ratios[i], countS1,countS2))
        print("beforechange:{}  vs ratio {}".format(returnSerial[i], ratios[i]))
        returnSerial[i]=money
        print("afterchange:{}".format(returnSerial[i]))
    return money

def zscore(series):
  return (series - series.mean()) / np.std(series)

import os
for stockPair in pairs:
  print("******** {} vs {} ********".format(stockPair[0],stockPair[1]))
  # create folder and save plot in folder
  workfolder='result/{}.{}'.format(stockPair[0],stockPair[1])
  if not os.path.isdir(workfolder):
      try:
          os.mkdir(workfolder)
      except OSError:
          print('creat folder faild {}'.format(workfolder))

  stock1=all_prices[stockPair[0]]
  stock2=all_prices[stockPair[1]]
  score, pvalue, _ = coint(stock1, stock2)
  print(" score :{}".format(score))
  print(" pvalue:{}".format(pvalue))

  price_ratios= stock1/stock2
  price_ratios.plot()
  plt.axhline(price_ratios.mean())
  plt.title('{} vs {}'.format(stockPair[0],stockPair[1]))
  plt.savefig('{}/priceratio.png'.format(workfolder))
  plt.show()

  
  zscore(price_ratios).plot()
  plt.axhline(zscore(price_ratios).mean())
  plt.axhline(1.0, color='red')
  plt.axhline(-1.0, color='green')
  plt.savefig('{}/zscore.png'.format(workfolder))
  plt.show()

  ratios=all_prices[stockPair[0]] / all_prices[stockPair[1]]
  #print(len(ratios))

  train=ratios[:2017]
  test=ratios[2017:]

  ratios_mavg5 = train.rolling(window=5, center=False).mean()
  ratios_mavg60 = train.rolling(window=60, center=False).mean()
  std_60 = train.rolling(window=60, center=False).std()

  zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60
  plt.figure(figsize=(15, 7))
  plt.plot(train.index, train.values)
  plt.plot(ratios_mavg5.index, ratios_mavg5.values)
  plt.plot(ratios_mavg60.index, ratios_mavg60.values)

  plt.legend(['Ratio', '5d Ratio MA', '60d Ratio MA'])

  plt.ylabel('Ratio')
  plt.savefig('{}/mavg.png'.format(workfolder))
  plt.show()
  
  plt.figure(figsize=(15,7))
  zscore_60_5.plot()
  plt.axhline(0, color='black')
  plt.axhline(1.0, color='red', linestyle='--')
  plt.axhline(-1.0, color='green', linestyle='--')
  plt.legend(['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
  plt.savefig('{}/zscore_60_5.png'.format(workfolder))
  plt.show()

  plt.figure(figsize=(18,7))

  train[160:].plot()
  buy = train.copy()
  sell = train.copy()
  buy[zscore_60_5>-1] = 0
  sell[zscore_60_5<1] = 0
  buy[160:].plot(color='g', linestyle='None', marker='^')
  sell[160:].plot(color='r', linestyle='None', marker='^')
  x1, x2, y1, y2 = plt.axis()
  plt.axis((x1, x2, ratios.min(), ratios.max()))
  plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
  plt.savefig('{}/train.png'.format(workfolder))
  plt.show()

  plt.figure(figsize=(18,9))
  S1 = all_prices[stockPair[0]].iloc[:2017]
  S2 = all_prices[stockPair[1]].iloc[:2017]

  S1[60:].plot(color='b')
  S2[60:].plot(color='c')
  buyR = 0*S1.copy()
  sellR = 0*S1.copy()

  # When you buy the ratio, you buy stock S1 and sell S2
  buyR[buy!=0] = S1[buy!=0]
  sellR[buy!=0] = S2[buy!=0]

  # When you sell the ratio, you sell stock S1 and buy S2
  buyR[sell!=0] = S2[sell!=0]
  sellR[sell!=0] = S1[sell!=0]

  buyR[60:].plot(color='g', linestyle='None', marker='^')
  sellR[60:].plot(color='r', linestyle='None', marker='^')
  x1, x2, y1, y2 = plt.axis()
  plt.axis((x1, x2, min(S1.min(), S2.min()), max(S1.max(), S2.max())))

  plt.legend([stockPair[0],stockPair[1], 'Buy Signal', 'Sell Signal'])
  plt.savefig('{}/stock.png'.format(workfolder))
  plt.show()
  #trade(all_prices[stockPair[0]].iloc[:2017],all_prices[stockPair[1]].iloc[:2017],60,5).plot()
  earning, tradeEarningCurve=trade1(all_prices[stockPair[0]].iloc[:2017],all_prices[stockPair[1]].iloc[:2017],60,5)
  print("  Earning:{}".format(earning))
  tradeEarningCurve.plot()
  plt.show()
  plt.savefig('{}/earning.png'.format(workfolder))



"""**BOOM! How 'bout dat?** That is beautiful. Now we can clearly see when we should buy or sell on the respective stocks.

Let's see how much money we can make off of this strategy, shall we?
"""
#trade(all_prices['MSFT'].iloc[:2017], all_prices['ADBE'].iloc[:2017], 60, 5)

"""### Backtest on Test Data

Let's test our function on the test data (2010-2016)
"""

#trade(all_prices['MSFT'].iloc[2018:], all_prices['ADBE'].iloc[2018:], 60, 5)

"""Looks like our strategy is profitable! Given that this data is occuring smack in the middle of the Great Recession, I'd say that's not bad!

## Areas of Improvement and Further Steps

By no means is this a perfect strategy and by no means was the implementation depicted in this article the best. There are several things that can be improved. Feel free to play around with the notebook or python files!

### 1. Using more securities and more varied time ranges

For the pairs trading strategy cointegration test, I only used a handful of stocks. Feel free to test this out on many more, as there are a lot of stocks in the stock market! Also, I only used the time range from 2000 to 2016, which by no means is representative of the average of the stock market in terms of returns or volatility.

### 2. Dealing with overfitting

Anything related to data analysis and training models has much to do with the problem of overfitting, which is simply when a model is trained a bit too closely to the data that it fails to perform when given actual, real data to predict on. There are many different ways to deal with overfitting like validation, Kalman filters, and other statistical methods. 

### 3. Adjusting the trading signals

One thing I noticed about my trading signal algorithm is that it doesn't account for when the stock prices actually overlap and cross each other. Because the code only calls for a buy or sell given their ratio, it doesn't take into account which stock is actually higher or lower. Feel free to improve on this for your own trading strategy!

### 4. More advanced methods

This is just the bare bones of what you can do with algorithmic pair trading. It's super simple because it only deals with moving averages and ratios. If you want to use more complicated statistics, feel free to do so. Some examples of more complex stuff are: the Hurst exponent, half-life of mean reversion, and Kalman filters.
"""
