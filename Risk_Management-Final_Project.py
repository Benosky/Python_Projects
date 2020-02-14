# -*- coding: utf-8 -*-
"""
Created on Mon Apr 02 13:22:25 2018

@author: win8
"""

#Risk Assessment of Portfolios: Part 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
from pandas import DataFrame as df
from Compute_KPI_OutputClass import Output
import datetime as dt
from datetime import timedelta
import calendar
import mibian   # To use this library, either execute "pip install mibian" at the Anaconda Prompt or download from http://code.mibian.net/

#I. Download data for last 10 years for all the 30 stocks of the Dow Jones Industrial 
#Average (ignore survivorship bias, and unless you have access to a point-in-time 
#database, simply download the data for the current set of DJIA index constituents.)


yf.pdr_override()
stocks =["MMM","AXP","AAPL","BA","CAT","CVX","CSCO","KO","DD","XOM","V","GE","GS",	
         "HD","INTC","IBM","JNJ","JPM","MCD","MRK","MSFT","NKE","PFE","PG","TRV",
         "UTX",	"UNH",	"VZ",	"WMT",	"DIS"]
stocks.sort() #sort symbols in alphabetical order
stk_prs =pdr.get_data_yahoo(stocks, start="2008-05-20",end="2018-05-20")["Close"]

#download data into DataFrame
data = df(stk_prs)

#II. In addition, download data for Futures contract of the DJIA for the same 
#period of time (consider the nearest month contract always)

#download the ^DJI Index value as a proxy/synthetic futures contract 
dat = pdr.get_data_yahoo("^DJI", start="2008-05-20",end="2018-05-20")
djia_dat = dat["Close"]


#III. For each of the cases outlined below, (unless specifically mentioned) randomly
#select a basket of 10 stocks from the list and create an equal weighted portfolio
#with a total value of 10,000$ invested all together at the beginning of the period
#under questions. Risk-return profile calculations should include calculations for
#the following KPIs –
investment = 10000 #Initial total investment amount
w =0.1 #assets equal weights

#selecting a random sample of n stock to create the portfolio

#portfolio 1 prices
longOnlyport = data.sample(n=10,replace=False,axis=1) #portfolio stock prices

#daily returns of stocks in long only portfolio
longOnlyport_dly_ret = longOnlyport.pct_change()[1:]

#computing the portfolio daily returns
longOnlyport_dly_ret["LongOnly_Return"] = longOnlyport_dly_ret.mean(axis=1) 

#ii. Risk Management by Stop Loss:
#Estimate the risk-return profile of such a long-only portfolio over the period
#of last 10 years.

print "---------------ii.Risk Management by Stop Loss----------------------"
#resampling the portfolio daily returns to monthly returns
longOnlyport_mly_ret = longOnlyport_dly_ret.resample('M').mean() 

#resampling the portfolio daily returns to yearly returns
longOnlyport_yrly_ret = longOnlyport_dly_ret.resample('A').mean()

#Create max monthly drawdown function
def max_monthly_ddown(x):
        """x = the series of portfolios monthlhy returns"""
        mdd =[]
        for i in range(len(list(x.columns.values))):
            largest_drop = min(x.iloc[:,i])#largest drop of portfolio i
            y_indx = np.argmin(x.iloc[:,i]) #the index of the largest drop
            rets_b4_drop = x.iloc[:,i][:y_indx] #series of prices before largest drop
            peak_b4_drop = max(rets_b4_drop) #peak before largest drop
            dd = (peak_b4_drop-largest_drop)/float(peak_b4_drop)
            mdd.append(dd)
        return mdd  

#Create generate kpi function
def generate_port_kpi(port_dly_ret):
    """Generates the KPI's for the simulated results.
    port_dly_ret: daily returns for each portfolio"""
    #Create the dataframe of the average daily returns for each portfolio as an
    #Output object.
    x = Output(port_dly_ret)
    res = x.generate_output()  
    mtly_ret = port_dly_ret.resample('M').mean()
    yrly_ret = port_dly_ret.resample('A').mean()
    kpi = pd.DataFrame()
    kpi["Average Monthly Returns"] = mtly_ret.mean() 
    kpi["Average Yearly Returns"] = yrly_ret.mean()
    kpi["Annualized Std Dev"] = res.loc["Annualized Std Dev"]
    kpi["Maximum Drawdowns"] = res.loc['Max Drawdown']
    kpi["Gain to Pain Ratio"] = res.loc['Gain to Pain Ratio']
    kpi["Lake Ratio"] = res.loc["Lake Ratio"]
    kpi["Sharpe Ratio"] = res.loc["Sharpe Ratio"]
    kpi["Maximum Monthly Drawdowns"] = max_monthly_ddown(port_dly_ret)
    kpi["Percentage of Positive Months"] = mtly_ret.apply(lambda x: (float(len(x[x > 0])) / float(len(mtly_ret)))*100)
    return kpi.T    

#ii. Now consider a modified portfolio, where position in any stock is fully 
#liquidated if the price drop 20% from its nearest 6-month high and its position is 
#filled by another randomly selected stock from the rest of the group.


#Create a function to modify the portfolio
def modify_port(d):
    """ d is the dataframe of all the stock prices"""
    #selecting the stocks for the initial portfolio
    port_p = d.sample(n=10,replace=False,axis=1) 
    #get a list of the column titles of the portfolio dataframe
    headers = list(port_p.columns.values)
    # drop the stocks in the portfolio from the rest of the group to avoid duplication of stocks in the portfolio
    mod_data = d.drop(headers, axis=1)
    #select a random stock from the cleaned dataframe of all stocks
    por = mod_data.sample(n=1,replace=False,axis=1)
    for x in range(120,len(port_p)):
        for i in headers:
            #full liquidation condition for the stocks
            if port_p[i][x]<(max(port_p[i][x-120:x])*(0.8)):
                #liquidate any stock that fails to meet the condition
                new_port = port_p.drop(i,axis=1)
                #fill the position of the liquidated stock the new randomly selected stock from the rest of the group.
                new_port.join(por)
    return port_p  


#iii. Estimate the risk-return profile of such a modified portfolio over the same 
#period. Compare and contrast the results with the simple long-only portfolio

modified_port = modify_port(data)#modified portfolio

#daily returns of stocks in modified portfolio
port_2 = modified_port.pct_change()[1:]

#computing the portfolio daily returns
port_2["StopLossPort_Return"] = port_2.mean(axis=1) 

#resampling the daily returns to monthly returns
port_2_Mly = port_2.resample('M').mean() 

#create a list of the portfolio returns to be compared
ports_ret_list1 = [longOnlyport_dly_ret["LongOnly_Return"],port_2["StopLossPort_Return"]]
#concatenate the list of the portfolio returns to be compared into a single dataframe
ports_ret_df1 = pd.concat(ports_ret_list1, axis=1)

print "==Compare and contrast the stop loss portfolio results with the simple long-only portfolio======="
#Generate the kpis
compare_1 = generate_port_kpi(ports_ret_df1)
print "\n",compare_1

print "\nThe Long Only portfolio outperformed the Stop Loss Portfolio across all major \
KPI's including in terms of the risk adjusted returns represented by the Sharpe \
Ratio of 0.676."

print "---------------iii. Risk Management by Hedging----------------------"

#i. Now consider a modified portfolio, where the overall long position of the 
#portfolio is hedged with a net short position in the DJIA index amounting to about 
#10% of the overall long portfolio size. As a rough estimate you can construct the 
#long-only portfolio on 9,500$ and use 500$ as the margin money to short the DJIA 
#for a overall position of 950$ short.

#download data into DataFrame
djia = df(djia_dat)
#DJIA overall position of 950$ short.
djia_short_positn = 950 
#Total position of the hedging portfolio, including the short position on DJIA
inv3_total_positn = 10450 
# it is an equal weighted portfolio
w2 = 0.09090909090909090909090909090909 

port3_pr =df()  
#portfolio 3 long assets
port3_pr =port3_pr.append(longOnlyport)
#include the short djia position in portfolio 3 
port3_pr['DJIA_short'] = djia 
#calculate the long assets returns
port3_dly_rtns = longOnlyport.pct_change()[1:]
#calculate the long position returns
port3_dly_rtns['Long Asset Returns'] = port3_dly_rtns.mean(axis=1)
#calculate the percentage change in the DJIA index and multiply it by -ve 1 to reflect that it is a return on a short position 
port3_dly_rtns['Short DJIA Daily Returns'] = (djia.pct_change()[1:])*(-1)
#calculate the hedged portfolio returns
port3_dly_rtns["HedgedPort_Return"] = (port3_dly_rtns['Long Asset Returns']+port3_dly_rtns['Short DJIA Daily Returns'])/2

#resampling the daily returns to monthly returns
port_3_Mly = port3_dly_rtns.resample('M').mean() 
#create a list of the portfolio returns to be compared
ports_ret_list2 = [longOnlyport_dly_ret["LongOnly_Return"],port3_dly_rtns["HedgedPort_Return"]]
#concatenate the list of the portfolio returns to be compared into a single dataframe
ports_ret_df2 = pd.concat(ports_ret_list2, axis=1)

print "==Compare and contrast the hedged portfolio results with the simple long-only portfolio======="
#Generate the kpis
compare_2 = generate_port_kpi(ports_ret_df2)
print "\n",compare_2


print "\nThe Long Only portfolio outperformed the Hedged portfolio in terms of the \
Average Monthly and Yearly Returns, and Percentage of Positive Months; but, of \
course, with higher risk profiles as represented by higher drawdowns and lake \
ratio, lower Gain to Pain Ratio and Standard Dev. But overall, in terms of risk \
adjusted return, the Long Only portfolio, with a Sharpe Ratio of 0.676 \
outperformed the Hedged portfolio"

print "====Risk Assessment of Portfolios: Part 2===="

print "i. Risk Mitigation by Diversification & Non-Correlation"


#ii. Now consider 4 more modified portfolios consisting of a randomly
#selected basket of 15, 20, 25 & 30 (all the stocks of the DJIA) stocks.

#create a list of the portfolio sizes
ports_sizes = [15,20,25,30]

#Create a list of the portfolio names
name_list = ["ModPort15","ModPort20","ModPort25","ModPort30"]

#Create function to generate random portfolios of the desired sizes from the dataframe 
#of the group of stocks and returns the average daily returns of the various portfolios

def ran_ports_avg_dly_ret(df,size_list,ports_names):
    """generates random portfolios with number of stock as listed in the size_list
    from the dataframe of a group of stocks, df, and returns the average daily returns 
    of the various portfolios"""
    df_r = df.pct_change()[1:]
    ports = []
    port_rets = pd.DataFrame()
    for s in size_list:
        port_pr = df_r.sample(n=s,replace=False,axis=1)
        ports.append(port_pr)
    for t in range(len(ports)):
        port_rets[t] = ports[t].mean(axis = 1)
    port_rets.columns = ports_names
    return port_rets

#Create a function to return the list of stocks in each random portfolio
def stocks_in_ran_ports(df,size_list):
    """Returns the list of stocks in each random portfolio 
    in ascending order of their sizes"""
    ports = []
    ports_stk_list = []
    for s in size_list:
        port_pr = df.sample(n=s,replace=False,axis=1)
        ports.append(port_pr)
    for t in ports:
        port_stocks = list(t.columns.values)
        ports_stk_list.append(port_stocks)
    for v in ports_stk_list:
        print v

#print out the list of stock in each modified random portfolio
print "===Stocks in each modified random portfolio in ascending order of their sizes are:"
stocks_in_ran_ports(data,ports_sizes)
#Create the dataframe of the various portfolios average daily returns 
modPorts_df = ran_ports_avg_dly_ret(data,ports_sizes,name_list)

#iii. Estimate the risk-return profile of such a modified portfolio over the
#same period. Compare and contrast the results with the simple longonly
#portfolio.

#create a list of the portfolio returns to be compared
ports_ret_list3 = [longOnlyport_dly_ret["LongOnly_Return"],modPorts_df]
#concatenate the list of the portfolio returns to be compared into a single dataframe
ports_ret_df3 = pd.concat(ports_ret_list3, axis=1)

print "==Compare and contrast the more diversified portfolios results with the simple long-only portfolio======="
#generate the kpis
compare_3 = generate_port_kpi(ports_ret_df3)
print "\n",compare_3


print "\nSurprisingly, the simple 10 stock long only portfolio outperformed all the other more \
diversified portfolios across all of the key performance indicators except for the \
Maximum Monthly Drawdowns and the percentage of Positive Months. This goes to \
show that diversification did not work beyond 10 stocks"


#v. Now redo steps (ii) to (iii), but this time do not randomly select stocks.
#Based on the 10 stocks initially in the portfolio, isolate the remaining
#20, calculate their historical return correlation (for time before the 10
#years in question – download data for at least 3 more back years for
#this) and rank them according to highest non-correlation to the overall
#portfolio under question. Now add stocks in their order of non-correlation to 
#the initial 10 stock portfolio to create 4 new portfolios of
#size 15,20,25 & 30 respectively

#download DJIA data for 4 years back
dow_stk =["MMM","AXP","AAPL","BA","CAT","CVX","CSCO","KO","DD","XOM","V","GE","GS",	
          "HD","INTC","IBM","JNJ","JPM","MCD","MRK","MSFT","NKE","PFE","PG","TRV",
          "UTX",	"UNH",	"VZ",	"WMT",	"DIS"]
dow_stk.sort() #sort symbols in alphabetical order
dow_4yr =pdr.get_data_yahoo(dow_stk, start="2005-04-03",end="2008-04-03")["Close"]

#download data into DataFrame
data_4yr_back = df(dow_4yr)
#tickers of stocks in the initial long only portfolio
d_tickers = list(longOnlyport.columns.values)
#isolate the remaining 20 stocks to form a new dataframe and calculate their daily returns
new_df = data_4yr_back.drop(d_tickers, axis=1).pct_change()[1:]
#include the average daily return of the initial long only portfolio to the new dataframe
new_df["LongOnly_Return"] =  data_4yr_back[d_tickers].pct_change()[1:].mean(axis=1) 
#calculate the correlation of the stocks with the initial portfolio and sort them 
# in descending order of their non-correlation to the initial portfolio
new_df = df(new_df.corr(method='pearson', min_periods=1)["LongOnly_Return"])
new_df_sorted = new_df.sort_values("LongOnly_Return") #,axis=1, ascending=False, inplace=True)
#select the index of the sorted correlation dataframe and drop the index of the initial portfolio return
sorted_cor = new_df_sorted.drop("LongOnly_Return", axis=0)
title = list(sorted_cor.index.values)

#Create a dataframe of the daily returns of the 4 new non-correlation stocks portfolios 
names = ["cor_port15","cor_port20","cor_port25","cor_port30"]
names[0]
#additional stocks
c = 5
cor_ports_df =df()
for n in range(len(names)):
    cor_ports_df[names[n]] = pd.concat([longOnlyport,data[title[:(n+1)*c]]],axis=1).pct_change().mean(axis=1)[1:]

#create a list of the portfolio returns to be compared
ports_ret_list4 = [longOnlyport_dly_ret["LongOnly_Return"],cor_ports_df]
#concatenate the list of the portfolio returns to be compared into a single dataframe
ports_ret_df4 = pd.concat(ports_ret_list4, axis=1)

#generate the kpis
compare_4 = generate_port_kpi(ports_ret_df4)
print "\n",compare_4

#vi. Analyze what percentage of the initial market-risk in the 10-stock portfolio was
#successfully “diversified off” by specifically adding non-correlated assets to the 
#portfolio. Based on the results, comment on whether diversification through addition
#of non-correlated assets work better than simple random diversification as observed 
#in step (iv)
print "\n==Compare and contrast the more diversified non correlated portfolios results\
          with the simple long-only portfolio======="


print "\nIncreasing the portfolio size from 10 to 15 successfully diversified \
off about 14% of the initial market-risk in the 10-stock portfolio. Thereafter, \
increasing the portfolio size from 10 to 20, 25, or 30, increased the market-risk \
in the 10-stock portfolio by 0.61%, 0.59% and 7.88%, respectively. In all, only the \
portfolio with 15 stocks (i.e., with the 5 most non-correlated stocks) with the \
highest Sharpe Ratio of 0.780, can be said to somewhat outperformed the long only \
portfolio. The rest of the portfolios underperformed the long only portfolio" 


print "\n=======ii. Risk Analysis of Options Portfolios – The Theta Decay Benefit======="

# plot DJIA  
plt.plot(dat['Adj Close'], label = 'DJIA')
plt.legend()
plt.show()

#II. Also download relevant Options data for DJIA â€“ consider the earliest expiry Put Option contract that is just at-the-money (ATM)
# Again, Yahoo may not not support downloading Options data for DJIA and no free source may be available
# If so, then compute options data yourself - below is one approach to do so for Put Options

# using code from starter pack we amend it and  built upon in the first half of the project

# Take Out-of-Money strick price (X) based on stepsize of N points
N_step = 250               # Assume stepsize of N points
N_trading_per_year = 252   # Number of trading days in one year excluding holidays and weekends
N_per_year = 365           # Number of days in a calendar year
N_per_month = 30           # Nubmer of days in a calendar month
N_per_week = 7             # Number of days in a calendar week
N_per_two_weeks = 14       # Number of days in calendar two weeks
N_Friday = 4               # Number for Friday

Strike  = "Strike"         # Add new column for strike price
Sigma   = "Volatility"     # Add new column for 30-day volatility
Expire  = "Expires"        # Add new column for option expiration date
Open = 'Open'
Close   = "Adj Close"      # Adjusted closing price used to compute strike price and volatility


# Compute and store stike price and volatility and save as new column

dat[Strike] = ((dat[Close]-N_step/2)/N_step).round()*N_step
#Adjust the Strike prices to reflect the 4th strike out of the money. Since most
#strike prices are in increments of $2.50 and $5 or even $10(for very higher priced stocks), we add
# 4 * $10= 40 to the Strike price to make it the 4th strike out of the money
dat[Strike] = dat[Strike] + 40
dat[Sigma] = dat[Close].pct_change().rolling(N_per_month).std()*np.sqrt(N_trading_per_year)
dat[Sigma]
dat.dropna(inplace = True)

plt.plot(dat[Sigma])
plt.title('Index volatility')
plt.show()

# copied from started code
# Function: get_expire_dates - Compute days to expiration
#
# Input:  dates - list of dates
# Output: expire_date_list - list of expiration dates

def get_expire_days(dates):
    expire_date_list = [] # Start with empty list
    
    # For each row (day), determine option expiration date  (do two weeks from 1st to 3rd friday of the month -- of course, this can be adjusted)  
    for name in dates:
        now = name                                                      # Current date
        if now.month < 12:
            next_month = now.month + 1
            year = now.year
        else:
            next_month = 1
            year = now.year + 1
        first_day_of_month = dt.datetime(year, next_month, 1)  # Save 1st day of month
        first_friday = first_day_of_month + timedelta(days=((N_Friday-calendar.monthrange(year,next_month)[0]) + N_per_week) % N_per_week) # 1st Friday
        third_friday = first_friday + timedelta(days=N_per_two_weeks)   # 3rd Friday
        if third_friday >= now: 
            expire_date_list.append(third_friday)                           # Add 2-week expiration date to table
        else:
            expire_date_list.append(np.nan)
    return expire_date_list

# Add expiration date of option (default is 2 weeks)
expire_dates_list = get_expire_days(dat.index)
expire_dates_df = pd.DataFrame(data = expire_dates_list, index = dat.index)
expire_dates_df = expire_dates_df.fillna(method = 'bfill')
expire_dates_df.columns = ['Expires']

dat = dat.merge(expire_dates_df, left_index = True, right_index = True)

strike_price = dat.groupby('Expires')['Strike'].mean()

# fill out each date with strike price for current option
for cnt,idx in enumerate(dat.index):
    idx_locator = dat.loc[idx]['Expires']
    dat.loc[idx,'Strike'] = strike_price[idx_locator] # write out located strike price against current date

# Get ready to add call options columns to table (price, mibian price, Theta, and hedge)
Call_M_open = "Call_price_mibian_open"      # Mibian Call price field name
Call_T_open = "Call_Theta_open"             # Call Theta field name
Call_H_open = "Call_hedge_open"             # Call hedge field name

Call_M_close = "Call_price_mibian_close"      # Mibian Call price field name
Call_T_close = "Call_Theta_close"             # Call Theta field name
Call_H_close = "Call_hedge_close"             # Call hedge field name

# Calculate Call option price (can do the same for put option price)
for row in dat.index:
    # calculate opening put prices, deltas and put_hedge
    S_open = dat.loc[row, Open]    # Open Price
    S_close = dat.loc[row, Close]    # Close Price
    K = dat.loc[row, Strike]   # Strike Price
    r = 0.01                  # Interest Free Rate
    v = dat.loc[row, Sigma]    # Computed Volatility (30 days)
    T = (dat.loc[row, Expire] - row).days # Time to expiration in days
    
    # If non-zero volatility, compute the Call option prices and Theta
    if v > 0:
        if T > 0:
            p_open = mibian.BS([S_open, K, r * 100, T], volatility=v * 100)
            dat.loc[row, Call_M_open] = p_open.callPrice
            dat.loc[row, Call_T_open] = p_open.callTheta
            
            p_close = mibian.BS([S_close, K, r*100, T], volatility=v * 100)
            dat.loc[row, Call_M_close] = p_close.callPrice
            dat.loc[row, Call_T_close] = p_close.callTheta
            
        else:
            dat.loc[row, Call_M_open] = 0
            dat.loc[row, Call_T_open] = 0
    
            dat.loc[row, Call_M_close] = 0
            dat.loc[row, Call_T_close] = 0

# III. Consider any particular trading month during the past 2 years (choose a month with few holidays).  Start trading on 1st day of the trading month.
dat[Call_H_open] = np.round(dat[Call_T_open]*100,1)  # Hedge the Theta
dat[Call_H_close] = np.round(dat[Call_T_close]*100,1)  # Hedge the Theta

#iii. Now consider a modified portfolio that tries to benefit from the Theta
#decay of options and generate a safe income on its long-holdings ( a
#slightly altered covered call scenario). With a long position in all the 30
#stocks for DJIA, sell an equivalent amount of Out of the Money Call
#Options to generate an X% cover. Vary X between 50%, 75% and 100%
#- thus considering 3 scenarios overall for the covered portfolio.

#Create the function for the modified theta portfolio
def modThetaPort(data_df,invest,opt_df):
    """Generates the a dataframe of the daily returns for the 3 scenarios of the 
    modified theta portfolio
    data_df= the dataframe of the original stocks prices over the period under review
    investment = initial investment amount
    opt_df = dataframe of options data
    
    """
    #Compute the long position
    
    #drop stocks with missing values
    clean_data = data_df.dropna(axis=1)
    #Equal weight for the stocks in the long portfolio
    w4 = 1.0/len(list(clean_data.columns.values))
    
    #compute initial equity holding for each stock
    stk_amt = invest*w4/data_df.iloc[0]
    
    #compute total daily portfolio position overtime and resample it to monthly averages
    y =(data_df*stk_amt).sum(axis=1)
    #create an empty dataframe for the result of the function
    res_df =pd.DataFrame()
    #compute the series of average premium
    res_df[" Average_Premium"] = (opt_df["Call_price_mibian_close"] + opt_df["Call_price_mibian_close"])/float(2)
    #Create a column for long_portfolio position in the dataframe
    res_df["long_position"] = y
    
    #To have the value of dj["long_position"] as the 50%, 70%, and 100% covers for a short call we 
    #must sell the following numbers of the call options:  
    
    res_df["calls_50pct_covered"] = (res_df["long_position"]/float(0.5))/opt_df['Adj Close']
    res_df["calls_75pct_covered"] = (res_df["long_position"]/float(0.75))/opt_df['Adj Close']
    res_df["calls_100pct_covered"] = res_df["long_position"]/opt_df['Adj Close']
    
    #Compute the premiums derived from the above three scenarios using the average of 
    #Call_price_mibian_close and Call_price_mibian_open as the option's average premium
    res_df["prem_50pct_cover"] = (res_df["calls_50pct_covered"] * res_df[" Average_Premium"]).dropna()
    res_df["prem_75pct_cover"] = (res_df["calls_75pct_covered"] * res_df[" Average_Premium"]).dropna()
    res_df["prem_100pct_cover"] = (res_df["calls_100pct_covered"] * res_df[" Average_Premium"]).dropna()
    
    #To compute the modified Options portfolio returns across the 3 scenarios, add the 
    #premiums in each scenario to the returns of the 30 stock long-only portfolio 
    #discussed in step (ii) of “Risk Mitigation by Diversification & Non-Correlation”
    series_50 = []
    series_75 = []
    series_100 = []
    for i in range(len(res_df["prem_50pct_cover"])):
        if opt_df["Strike"][i]>=opt_df['Adj Close'][i]:
            res_50 = res_df["long_position"][i] + res_df["prem_50pct_cover"][i]
            res_75 = res_df["long_position"][i] + res_df["prem_75pct_cover"][i]
            res_100 = res_df["long_position"][i] + res_df["prem_100pct_cover"][i]
        else:
            #Capture the occasional losses due to exercise
            res_50 = res_df["long_position"][i] + res_df["prem_50pct_cover"][i]-(opt_df['Adj Close'][i]-opt_df["Strike"][i])
            res_75 = res_df["long_position"][i] + res_df["prem_75pct_cover"][i]-(opt_df['Adj Close'][i]-opt_df["Strike"][i])
            res_100 = res_df["long_position"][i] + res_df["prem_100pct_cover"][i]-(opt_df['Adj Close'][i]-opt_df["Strike"][i])
        series_50.append(res_50)
        series_75.append(res_75)
        series_100.append(res_100)
        
    res_df["ret_50pct_cover"] = series_50
    res_df["ret_75pct_cover"] = series_75
    res_df["ret_100pct_cover"] = series_100
    result = pd.concat([res_df["ret_50pct_cover"],res_df["ret_75pct_cover"],res_df["ret_100pct_cover"]],axis=1).dropna().pct_change()[1:]
    return result


dat[" Average_Premium"] = (dat["Call_price_mibian_close"] + dat["Call_price_mibian_close"])/float(2)

mod_theta_port = modThetaPort(data,investment,dat)

#create a list of the portfolio returns to be compared
ports_ret_list5 = [cor_ports_df["cor_port30"],mod_theta_port]
#concatenate the list of the portfolio returns to be compared into a single dataframe
ports_ret_df5 = pd.concat(ports_ret_list5, axis=1).dropna()

#generate the kpis
compare_5 = generate_port_kpi(ports_ret_df5)
print "\n",compare_5

#v. Based on the results, comment on whether adding a pinch of Short
#Options positions to a overall diversified long portfolio can produce
#significantly better risk-return profiles than standard diversified
#portfolios.
print "\n==Compare and contrast the risk-return profile of the modified Options portfolio with the simple longonly\
portfolio of 30 stocks======="
 
print "\nAdding a pinch of Short Options positions to the overall diversified portfolio \
increased the Average Monthly and Yearly Returns in the three scenarios of the modified \
Options portfolio. However, in terms of the risk adjusted returns, the simple longonly \
portfolio of 30 stocks, with a Sharpe ratio of 0.66, outperformed the modified Options \
portfolio. Therefore, adding a pinch of Short Options positions to an overall diversified \
long portfolio cannot produce significantly better risk-return profiles than standard \
diversified portfolios. "

