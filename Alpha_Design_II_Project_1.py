# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 09:55:59 2018

@author: win8
"""
"""
Import Libraries & Packages
"""
import pandas as pd
import numpy as np
from yahoofinancials import YahooFinancials
#1. Select a universe of 100 stocks spread across different industry verticals – information
#technology, utilities, banking and financial services, midcaps, large caps etc.
"""
symbols	= ["AAPL","ABBV",	"ABT","ACN","AGN","AIG","ALL","AMGN","AMZN","AXP","BA","BAC",
           "IB","BK","BKNG","BLK",	"BMY","BRK.B","C","CAT","CELG","CHTR","CL","CMCSA",
           "COF",	"COP","COST","CSCO","CVS","CVX","DHR","DIS","DUK","DWDP","EMR","EXC",
           "FB",	"FDX",	"FOX",	"GD",	"GE",	"GILD",	"GM",	"GOOG",	"GS",	"HAL",	
           "HD",	"HON",	"IBM",	"INTC",	"JNJ",	"JPM",	"KHC",	"KMI",	"KO",	"LLY",	"LMT",
           "W",	"MA",	"MCD",	"MDLZ",	"MDT",	"MET",	"MMM",	"MO",	"MRK",	"MS",	"MSFT",	
           "NEE",	"NFLX","NKE",	"ORCL","OXY",	"PEP",	"PFE",	"PG","PM",	"PYPL","QCOM",
           "RTN",	"SBUX","SLB","SO","SPG","T","TGT",	"TWX","TXN","UNH","UNP","UPS","USB",	
           "UTX",	"V",	"VZ",	"WBA",	"WFC",	"WMT",	"XOM",]
"""
#2. Download previous year’s balance sheets for all and calculate the following metrics:
#• Earnings Yield
#• EBITA
#• Free cash flow yield
#• Return on Capital Employed
#• Book to Market

#I created a function todownloads the required financials and calculates the required 
#metrics as follows:

def fin_metrics(symbols):
    """symbols = tickers of the universe of required number of stocks"""
    #sort the list of the symbols and assign it to tickers
    tickers = sorted(symbols)
    #create the YahooFinancials object as records
    records = YahooFinancials(tickers)
    #get the financial statements
    statements = records.get_financial_stmts('annual', ['income', 'cash', 'balance'])
    #get the ebit for each ticker from the financials 
    ebit = records.get_ebit()
    #get the previous stock prices for each stock
    share_pr = records.get_prev_close_price()
    #create an empty dataframe to hold the final result
    results_df = pd.DataFrame()
    #create empty lists to hold the metrics
    e_yield = []
    ebita = []
    fcfy = []
    roce = []
    b2m = []
    #get the firt ticker in the list of tickers
    first_sym = tickers[0]
    #create the indexes for the dataframe of the various financial statments
    bs_index = statements['balanceSheetHistory'][first_sym][0].values()[0].keys()
    is_index = statements['incomeStatementHistory'][first_sym][0].values()[0].keys()
    cf_index = statements['cashflowStatementHistory'][first_sym][0].values()[0].keys()
    #create the empty dataframes to hold the various financial statements
    bs_df = pd.DataFrame(index=bs_index)
    is_df = pd.DataFrame(index=is_index)
    cf_df = pd.DataFrame(index=cf_index)
    for s in symbols:
        #get the components of the various financial statements
        bs_result = pd.DataFrame(statements['balanceSheetHistory'][s][0].values()).T
        is_result = pd.DataFrame(statements['incomeStatementHistory'][s][0].values()).T
        cf_result = pd.DataFrame(statements['cashflowStatementHistory'][s][0].values()).T
        #put the components of the various financial statement in the empty dataframes created for them
        is_df = pd.concat([is_df, is_result], axis=1, join='outer')
        bs_df = pd.concat([bs_df, bs_result], axis=1, join='outer')
        cf_df = pd.concat([cf_df, cf_result], axis=1, join='outer')
        #create a list of the dataframes of the various financial statemenmts
        frames = [bs_df,is_df,cf_df]
        #concatenate the dataframe of the various financial statements into a single dataframe
        all_fins = pd.concat(frames).fillna(0)
    #Assign the list of tickers as the label of the single dataframe columns
    all_fins.columns = tickers
    for t in all_fins:
        #compute the required metrics and assign them to the dataframe created for them
        ev = all_fins.loc['commonStock'][t]*share_pr[t] + all_fins.loc['shortLongTermDebt'][t]-all_fins.loc['cash'][t]
        y = ebit[t]/float(ev)
        fcff = all_fins.loc['totalCashFromOperatingActivities'][t] - all_fins.loc['capitalExpenditures'][t]
        fcfs = float(fcff)/all_fins.loc['commonStock'][t]
        fcf_yield = float(fcfs)/share_pr[t]
        ce = all_fins.loc['totalAssets'][t] - all_fins.loc['totalCurrentLiabilities'][t]
        roc_employed =float(ebit[t])/ce
        cse = all_fins.loc['totalStockholderEquity'][t] - all_fins.loc['otherStockholderEquity'][t]
        b2_market = float(cse)/all_fins.loc['commonStock'][t]*share_pr[t]
        fcfy.append(fcf_yield)
        e_yield.append(y)
        ebita.append(ebit[t])
        roce.append(roc_employed)
        b2m.append(b2_market)
    results_df['EBITA'] = ebita
    results_df['Earnings Yield'] = e_yield
    results_df['Free Cash Flow Yield'] = fcfy
    results_df['Return on Capital Employed'] = roce
    results_df['Book to Market'] = b2m
    results_df.index = tickers
    return results_df

#assign the dataframe of the metrics to metrics
sym = ['ABBV','AAPL','MSFT', 'WMT']
metrics = fin_metrics(sym)
#3. Arrange stocks in deciles in according to the value of these metrics:
#• the top decile being Glamour Stocks bottom Decile being Value Stocks.
metrics['Average Metrics'] = metrics.mean(axis = 1)
metrics['decile'] = pd.qcut(metrics['Average Metrics'], 10, labels=False)
metrics = metrics.sort_values(by=['decile'], ascending=[True])
value_stocks = []
glamour_stocks = []
for t in list(metrics.index.values):
    if metrics.loc[t]['decile']>=9:
        value_stocks.append(t)
    elif metrics.loc[t]['decile']<=2:
        glamour_stocks.append(t)
        
print "The glamour stocks are:",glamour_stocks
print "The value stocks are:",value_stocks

#4. Choose only the Value decile stocks to proceed (you should roughly have 
#20 stocks in the value decile)

#stocks in the value decile)

value_metrics = metrics.loc[value_stocks]
value_metrics = value_metrics.T
value_metrics
#5. For the value decile stocks, calculate the following (from the time of publishing of last Annual
#report to date) risk and return metrics
"""
• CAGR
• Standard Deviation of Returns
• Downside Deviation of Returns
• Sharpe Ratio
• Sortino Ratio (with MAR Set at 5%)
• Worst Drawdown
• Worst Moth Return
• Best Month Return
• Profitable Months
"""

#create and empty dataframe for the return metrics
returns_metrics = pd.DataFrame()
#compute the return metrics
#set Risk Free Rate for Sharpe Ratio
rf = 0.001 
# set MAR for Sortino Ratio
tr = 0.05 
returns_metrics['CAGR'] = value_metrics.iloc[:, -1].div(value_metrics.iloc[:, 0]).pow(1./(len(value_metrics.columns)-1 )).sub(1)
returns_metrics['Std Dev'] = value_metrics.std(axis = 1)
returns_metrics['Sharpe Ratio'] = (value_metrics.T.pct_change().mean()-rf)/returns_metrics['Std Dev']
returns_metrics['Worse Month Return'] = value_metrics.min(axis=1)
returns_metrics['Best Month Return'] = value_metrics.max(axis=1)

# Calculate Sortino Ratio
# compute the difference between returns and the target return
diff = tr-value_metrics.pct_change().mean(axis=1) 
# clip the minimum of each at 0
diff = diff.clip(0)  
# Compute the lower partial moment (lpm) of the returns
lpm = (diff**2).sum/len(value_metrics) .T
sortino_ratio = (value_metrics.mean(axis=1)-tr) / np.sqrt(lpm)
returns_metrics['sortino_ratio'] = sortino_ratio

#6. Tabulate the results and identify which value metrics provide the best risk adjusted measure of return.
print returns_metrics