# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 09:31:52 2018

@author: Benjamin Umeh
"""

"""
Mini Project 2: Optimizing Trend Trading using Mean Reversion Inputs
"""
# Import Relevant Libraries
import pandas as pd
from pandas_datareader import data as web
import datetime
import numpy as np
import matplotlib.pyplot as plt
from Compute_KPI_OutputClass import Output

# Initialize the Start and end date
start = datetime.datetime(2008,1,1)
end = datetime.date.today() 
#Input a valid ticker
msft = web.DataReader(input("Please Input the name of the Ticker:\n"), "yahoo", start, end)
# Assign the short and long windows
short_window = 40
long_window = 100
# Assign the signal DataFrame with the `signal` column
signals = pd.DataFrame(index=msft.index)
# Set default signal = 0
signals['signal'] = 0.0
# Compute short simple moving average over the short window
signals['short_mavg'] = msft['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
# Compute long simple moving average over the long window
signals['long_mavg'] = msft['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
# Compute signals
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)   
# Raise orders
signals['positions'] = signals['signal'].diff()
# Print total number of orders
print "signals['positions'].count()"

# Plot Positions
signals['positions'].plot()

# Print signals
print(signals)
# assign the plot figure
fig = plt.figure()
# Add a subplot and label for y-axis
ax1 = fig.add_subplot(111,  ylabel='Price ($)')
# Plot the closing price
msft['Close'].plot(ax=ax1, color='r', lw=2.)
# Plot the short and long moving averages
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)
# Plot the buy signals
ax1.plot(signals.loc[signals.positions == 1.0].index, signals.short_mavg[signals.positions == 1.0],'^', markersize=10, color='m')
# Plot the sell signals
ax1.plot(signals.loc[signals.positions == -1.0].index, signals.short_mavg[signals.positions == -1.0],'v', markersize=10, color='k')
# Show the plot
plt.show()
# Backtest Strategy
# Assign the initial capital
initial_capital= float(100000.0)
# Create a DataFrame of positions
positions = pd.DataFrame(index=signals.index).fillna(0.0)
# Buy 100 shares of MSFT per trade
positions['MSFT'] = 100*signals['signal']   
# Initialize the portfolio with value owned   
portfolio = positions.multiply(msft['Adj Close'], axis=0)
# Save the difference in shares owned 
pos_diff = positions.diff()
# Add holdings to portfolio
portfolio['holdings'] = (positions.multiply(msft['Adj Close'], axis=0)).sum(axis=1)
# Add cash to portfolio
portfolio['cash'] = initial_capital - (pos_diff.multiply(msft['Adj Close'], axis=0)).sum(axis=1).cumsum()   
# Add total to portfolio
portfolio['total'] = portfolio['cash'] + portfolio['holdings']
# Add `returns` to portfolio
portfolio['returns'] = portfolio['total'].pct_change()
# Print the first lines of `portfolio`
print(positions)
# Visualize the Equity Curve / Strategy Result
fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')
# Plot the equity curve in dollars
portfolio['total'].plot(ax=ax1, lw=2.)
ax1.plot(portfolio.loc[signals.positions == 1.0].index, portfolio.total[signals.positions == 1.0],'^', markersize=10, color='m')
ax1.plot(portfolio.loc[signals.positions == -1.0].index, portfolio.total[signals.positions == -1.0],'v', markersize=10, color='k')
# Show the plot
plt.title('Equity Curve of SMA Crossover Strategy using MSFT Stock')
plt.show()



# Compute the Relevant KPIs
#create a Output kpi object
x = Output(portfolio['returns'])
res = x.generate_output()

CAGR = portfolio['returns'][-1]/(portfolio['returns'][0]).pow(1./(len(portfolio['returns'])-1 )).sub(1)
print "CAGR:",CAGR