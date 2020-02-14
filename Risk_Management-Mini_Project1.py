# -*- coding: utf-8 -*-
"""
Created on Wed May 02 11:20:17 2018

@author: win8
"""

if __name__ == '__main__':
    #import the required modules
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pandas_datareader import data as pdr
    import fix_yahoo_finance as yf
    from Compute_KPI_OutputClass import Output

    yf.pdr_override()
    stocks =["MMM","AXP","AAPL","BA","CAT","CVX","CSCO","KO","DD","XOM","V","GE",
         "GS","HD","INTC","IBM","JNJ","JPM","MCD","MRK","MSFT","NKE","PFE",
         "PG","TRV","UTX",	"UNH",	"VZ",	"WMT",	"DIS"]
    stocks.sort() #sort symbols in alphabetical order
    # get the data and calculate the KPIs
    stk_prices =pdr.get_data_yahoo(stocks, start="2013-04-19",end="2018-04-19")["Close"]
    # Resample the daily prices to monthly using the last price of the month
    monthly_stk_prices =  stk_prices.resample('M').last()
    #calculate the monthly returns
    monthly_returns = monthly_stk_prices.pct_change()[1:] #drop the first NaN
    #Average Monthly Return on a Positive Month
    positive_return_months = monthly_returns.apply(lambda x: x[x > 0].mean())
     #Average Monthly Return on a negative Month
    negative_return_months = monthly_returns.apply(lambda x: x[x < 0].mean())
    # Probability of a Positive Month
    prob_pos_mont = monthly_returns.apply(lambda x: float(len(x[x > 0])) / float(len(monthly_returns)))
        
    # use kernel density estimation to estimate the monthly return distribution for each stock
    #params = {'bandwidth': np.logspace(-1, 1, 20)}
    #grid = GridSearchCV(KernelDensity(), params)
    #models_s = monthly_returns.apply(lambda x: grid.fit(x.values.reshape(-1, 1)).best_estimator_)
    #I was not able to use kernel density estimation because models_s returned the following error (which was taking too much time to resolve)
    #: ValueError: ("Input contains NaN, infinity or a value too large for dtype('float64').", u'occurred at index DD')
    
    #Create function to generate random portfolios of desired sizes from the dataframe 
    #of selected stocks and returns the monthly returns of the stocks in the various 
    #portfolios
    def ran_ports_hst_mret(df,size_list):
        """generates random portfolios with number of stock as listed in the size_list
        from the dataframe of selected stocks closed prices,df, and returns the monthly returns 
        of the stocks in the various portfolios"""
        df_r = df.resample('M').last().pct_change()[1:]
        ports = []
        for s in size_list:
            port_pr = df_r.sample(n=s,replace=False,axis=1)
            ports.append(port_pr)
        return ports
    
    #Create a function to return the list of stocks in each random portfolio
    def stocks_in_ran_ports(ran_ports):
        """Returns the list of stocks in each random portfolio 
        in ascending order of their sizes"""
        ports_stk_list = []
        for s in ran_ports:
            port_stocks = list(s.columns.values)
            ports_stk_list.append(port_stocks)
        for s in ports_stk_list:
            print s
    
    #Create a function generate the historical average monthly returns of the randomly 
    #selected portfolios
    def ports_avg_hst_m_rets(df,col_names,size_list):
        """Returns the randomly generated portfolios' average historical monthly 
        returns 
        size_list = list of desired portfolios sizes arranged in ascending order
        df = the original dataframe of selected stocks closed prices 
        col_names= the names we choose to assign to the columns of the returned dataframe
        containing each portfolio's average monthly returns"""
        lst = ran_ports_hst_mret(df,size_list)
        port_rets = pd.DataFrame()
        for s in range(len(lst)):
            port_rets[s] = lst[s].mean(axis = 1)
        port_rets.columns = col_names
        return port_rets
    
    #Create the monte carlo simulation function
    def port_monte_carlo_sim(df,inv,num_simulations,port_names,predicted_months,size_list):
        """ Generates a list containing the simulation of portfolio values for the various portfolios. 
        df = the population of stock (e.g.DJIA 30) to create the random portfolios from
        inv = Initial investment amount
        num_simulations = number of simulations
        port_names = a list of the names of the portfolios in ascending order of their sizes
        predicted_months = number of predictions to make
        size_list = a list of the sizes of the portfolios in ascending order"""
        #Create the various portfolios and calculate their monthly returns
        pmr = ports_avg_hst_m_rets(df,port_names,size_list)
        sim_result = []
        for t in range(len(list(pmr.columns.values))):
            #Calculate each portfolio maximum and minimum monthly returns
            mu = pmr.iloc[:,t].mean()
            vol = pmr.iloc[:,t].std()
            #create an empty dataframe for the simulation results
            simul_df = pd.DataFrame()
            #Create Each Simulation as a Column in dataframe simul_df
            for x in range(num_simulations):
                #set an empty list to hold our simulated monthly portfolio values
                series = [] 
                #generate the predicted monthly portfolio values
                for i in range(predicted_months):
                    #generate a list of monthly returns using random normal distribution and
                    mntly_ret = inv * (1 + np.random.normal(mu,vol))
                    #append the ending value of each simulated run to the empty list we created at the beginning
                    series.append(mntly_ret)
                    #replace the initial investment amount with the new portfolio value
                    inv = mntly_ret
                    #Assign the simulation results to the columns of the empty dataframe created above    
                simul_df[x] = series
            sim_result.append(simul_df)
        return sim_result
    
    #Create the shocked monte carlo simulation function
    def shocked_port_monte_carlo_sim(df,inv,num_simulations,port_names,predicted_months,size_list):
        """Generates a list containing the simulation of portfolio values for the 
        various portfolios with one two-day consecutive downwards price shock of 10% total 
        at steps 40 and 41"""
        #Create the various portfolios and calculate their monthly returns
        pmr = ports_avg_hst_m_rets(df,port_names,size_list)
        sim_result = []
        for t in range(len(list(pmr.columns.values))):
            #Calculate each portfolio maximum and minimum monthly returns
            mu = pmr.iloc[:,t].mean()
            vol = pmr.iloc[:,t].std()
            #create an empty dataframe for the simulation results
            simul_df = pd.DataFrame()
            #Create Each Simulation as a Column in dataframe simul_df
            for x in range(num_simulations):
                #set an empty list to hold our simulated monthly portfolio values
                series = [] 
                #generate the predicted monthly portfolio values
                for i in range(predicted_months):
                    #generate a list of monthly returns using random normal distribution
                    mntly_ret = inv * (1 + np.random.normal(mu,vol))
                    #append the ending value of each simulated run to the empty list we created at the beginning
                    series.append(mntly_ret)
                    #replace the initial investment amount with the new portfolio value
                    inv =  mntly_ret
                #add one two-day consecutive downwards price shock of 10% total at step 40 and 41
                series[39] = series[39]*(0.94)
                series[40] = series[40]*(0.96)
                #Assign the simulation results to the columns of the empty dataframe created above    
                simul_df[x] = series
            sim_result.append(simul_df)
        return sim_result    
    
    #Create a function generate the simulated average monthly returns of the randomly 
    #selected portfolios
    def ports_avg_sim_m_rets(lst,col_names):
        """Returns a dataframe containing the average simulated monthly 
        returns of the various portfolios
        lst= the list containing the simulation of portfolio values for the various portfolios 
        whose mean simulated return will be determined
        col_names= the names we choose to assign to the columns of the output dataframe
        containing each portfolio's mean simulated returns"""
        port_rets = pd.DataFrame()
        for s in range(len(lst)):
            rets = lst[s].pct_change().mean(axis = 1)
            port_rets[s] = rets.fillna(rets.mean())
        port_rets.columns = col_names
        return port_rets
        
    #Create a function to generate a single dataframe of all the simulations of all the portfolios
    def sim_res_df(list_sims):
        """Generates a single dataframe of all the simulations of all the portfolios
        list_sims = the list of all the simulations from running the Monte Carlo simulation"""
        #concatenate the results of all the simulations in sim_list above and assign it to sim_results
        sim_results_df = pd.concat(sim_list, axis=1)
        return sim_results_df
             
    #Create max drawdown function
    def max_ddown(avg_sim_ret):
        """sim_returns is the dataframe of simulated average portfolio returns for each portfolio"""
        mdd =[]
        for i in range(len(list(avg_sim_ret.columns.values))):
            largest_drop = min(avg_sim_ret.iloc[:,i])#largest drop of portfolio i
            y_indx = np.argmin(avg_sim_ret.iloc[:,i]) #the index of the largest drop
            rets_b4_drop = avg_sim_ret.iloc[:,i][:y_indx] #series of prices before largest drop
            peak_b4_drop = max(rets_b4_drop) #peak before largest drop
            dd = (peak_b4_drop-largest_drop)/float(peak_b4_drop)
            mdd.append(dd)
        return mdd  
    
    #Create generate kpi function
    def generate_kpi(savr):
        """Generates the KPI's for the simulated results.
        savr: simulated average monthly returns for each portfolio"""
        #Create the dataframe of simulated average monthly returns for each portfolio as an
        #Output object.
        x = Output(savr)
        res = x.generate_output()  
        kpi = pd.DataFrame()
        kpi["Mean Portfolio Returns"] = savr.mean(axis=0)
        kpi["Variance of Portfolio Returns"] = savr.std(axis=0)**0.5
        kpi["Maximum Drawdowns"] = max_ddown(savr)
        kpi["Loss Rate"] = res.loc['Loss Rate']
        kpi["Win Rate"] = res.loc["Win Rate"]
        kpi["Risk of Ruin"] = ((1-(kpi["Win Rate"]-kpi["Loss Rate"]))/(1+(kpi["Win Rate"]-kpi["Loss Rate"])))**(float(inv)/kpi["Maximum Drawdowns"])
        return kpi.T    
        
            
    #IV. Graphically represent the risk and return profiles of each of 600 simulations in two
    #separate plots – one of Risk and another for the cumulative portfolio growth
    
    #Initialize the parameters are as follows:
    #list of the portfolio sizes
    p_sizes = [15,16,17,18,19,20] 
    #list of the portfolio names
    name_list = ["Sim_15","Sim_16","Sim_17","Sim_18","Sim_19","Sim_20"]
    #number of simulations
    num_sims = 100
    #number of predicted months
    predtns = 60
    #Initial Investment
    inv = 1000000
    
    #Generate the random portfolios
    r_ports = ran_ports_hst_mret(stk_prices,p_sizes)
    
    #run simulations and return a list of all the simulations of for all the portfolios
    sim_list = port_monte_carlo_sim(stk_prices,inv,num_sims,name_list,predtns,p_sizes)
    
    #Generate the dataframe containing the average simulated monthly returns of each portfolios
    sims_avg_rets = ports_avg_sim_m_rets(sim_list,name_list)
    
    #generate a range of dates for the months predicted by the simulation
    dates = pd.date_range(start='05-01-2018', periods=60, freq='M')
    
    #make the range of dates the index of the simulated/predicted portfolio average monthly returns dataframe
    sims_avg_rets.index = dates
    
    #Generate a dataframe of all the simulations of all the portfolios
    df_sim_res = sim_res_df(sim_list)
    
    #plot a line graph of the simulations
    plt.plot(df_sim_res)
    
    #plot the histogram of the cumulative growth of all the simulations
    
    #Calculate the returns on all the simulations
    all_sims_returns = df_sim_res.pct_change().replace([np.inf, -np.inf], np.nan)
    
    #create the cumulative growth of all the simulations as the last 
    #row of the sim_res dataframe 
    all_sims_returns.loc["Cumulative"] = all_sims_returns.sum(axis=0)
    cum_growth = all_sims_returns.loc["Cumulative"]
    
    #plot the histogram of the cumulative growth of all the simulations
    fig1 = plt.figure()
    plt.hist(list(cum_growth),bins=50)
    plt.title("The Cumulative Growth of Portfolios From Simple Simulation")
    plt.xlabel("Cumulative Portfolio Growth")
    plt.ylabel("Frequency")
    plt.show 


    #create the standard deviation of all the simulations as the last 
    #row of the all_sims_returns dataframe 
    all_sims_returns.loc["Risk"] = all_sims_returns.std(axis=0)
    risk = all_sims_returns.loc["Risk"]
    #plot the histogram of the risk of all the simulations
    fig2 = plt.figure()
    plt.hist(list(risk),bins=50)
    plt.title("The Risks of Portfolios From Simple Simulation")
    plt.xlabel("Portfolio Risk")
    plt.ylabel("Frequency")
    plt.show
    
    
    #V. Based on the results of the simulation, calculate measures of the mean portfolio
    #returns, the overall variance of portfolio returns, maximum drawdown and the risk of
    #ruin
    
    print "The List of Stocks In Each Portfolio In Ascending Order Of Portfolio Sizes are:"
    ports_stocks = stocks_in_ran_ports(r_ports)
    ports_stocks
    
    #Run the required_kpi function to generate the required KPIs and print out the results
    required_kpi = generate_kpi(sims_avg_rets)
    print '===========KPIs For The Simple Monte Carlo Simulation========='
    print required_kpi


    #Rerun the same simulations again, this time deliberately adding one two-day
    #consecutive downwards price shock of 10% total anytime between the 10-50th step.
    #Redo steps IV & V for the new set of simulation results
    
    #run simulations and return a list of all the simulations of for all the portfolios
    sim_list2 = shocked_port_monte_carlo_sim(stk_prices,inv,num_sims,name_list,predtns,p_sizes)
    #Generate the dataframe containing the average simulated monthly returns of each portfolios
    sims_avg_rets2 = ports_avg_sim_m_rets(sim_list2,name_list)
    #make the range of dates the index of the simulated/predicted portfolio average monthly returns dataframe
    sims_avg_rets2.index = dates
    #Generates dataframe of all the 600 simulations of all the portfolios
    df_sim_res2 = sim_res_df(sim_list2)
    
    #plot the histogram of the cumulative growth of all the simulations
    
    #Calculate the returns on all the simulations
    all_sims_returns2 = df_sim_res2.pct_change().replace([np.inf, -np.inf], np.nan)
    #create the cumulative growth of all the simulations as the last 
    #row of the sim_res dataframe 
    all_sims_returns2.loc["Cumulative"] = all_sims_returns2.sum(axis=0)
    cum_growth2 = all_sims_returns2.loc["Cumulative"]
    #plot the histogram of the cumulative growth of all the simulations
    fig3 = plt.figure()
    plt.hist(list(cum_growth2),bins=50)
    plt.title("The Cumulative Growth of Portfolios From Shocked Simulation")
    plt.xlabel("Cumulative Portfolio Growth")
    plt.ylabel("Frequency")
    plt.show 

    #plot the histogram of the risks of all the simulations from the shocked monte carlo function

    #create the standard deviation of all the simulations as the last 
    #row of the all_sims_returns dataframe 
    all_sims_returns2.loc["Risk"] = all_sims_returns2.std(axis=0)
    risk = all_sims_returns2.loc["Risk"]
    #plot the histogram of the risks of all the simulations
    fig4 = plt.figure()
    plt.hist(list(risk),bins=50)
    plt.title("The Risks of Portfolios From Shocked Simulation")
    plt.xlabel("Portfolio Risk")
    plt.ylabel("Frequency")
    plt.show
    
    required_kpi2 = generate_kpi(sims_avg_rets2)
    print '===========KPIs For Shocked Monte Carlo Simulation========='
    print required_kpi2
    
    
    #Analysis:
    #how likely is the trader to reach her goal of 25% CAGR from a portfolio based 
    #on solely DJIA stocks?
    
    prob_cagr = len(cum_growth[cum_growth>0.25])/len(cum_growth)
    print "\nThe likelihood of the trader reaching her goal of 25% is",prob_cagr,"%"
    
    #What the risks she is exposing herself to by following her investment strategy?
    print '\nThe risks that the trader is exposing herself to by following her \
    investment Strategy are:\n\
         (1) Equity Risk\n\
         (2) Strategic Risk'
    
    #Does such an investment strategy auger a favorable risk-return profile? 
    
    #Generate the main kpi output and extract the average Sharpe Ratio across all portfolios
    kpi_res = Output(sims_avg_rets)    
    x = kpi_res.generate_output()
    sharpe = x.loc["Sharpe Ratio"].mean()
    print "\nWith an average Sharpe Ratio of",sharpe,"across all the portfolios the \
    investment strategy augurs a favorable, in fact an excellent, risk-return profile \
    if indeed the simulations returned the accurate numbers"
    
    #How does the strategy stand up against downwards price shocks?
    
    #Extract the Sharpe Ratio from the shocked Monte Carlo simulation and compare it
    #with the Sharpe Ratio from the simple Monte Carlo simulation.
    kpi_res2 = Output(sims_avg_rets2)    
    y = kpi_res2.generate_output()
    sharpe2 = y.loc["Sharpe Ratio"].mean()
    sharpe2
    print "\nThe average Sharpe Ratio of",sharpe,"from the simple Monte Carlo simulation\
    quite exceeds the average Sharpe Ratio of",sharpe2,"from the shocked Monte Carlo\
    simulation. This indicates that the strategy does not stand up well against price shocks"
    
    #Considering all 1200 simulations, how bad would the trader be doing under the worst
    #case scenario?
    
    #Determine the minimum cumulative return wich represents the worst case/return
    worst_case = round(all_sims_returns.loc["Cumulative"].min()*100,2)
    print "\nIn the worst case the trader will have a loss of",worst_case,"%"
    
    #In what Percentage of cases does she face risk of total ruin?
    print "\nThe trader does not face risk of total ruin, as the risk\
    of ruin is zero across all portfolios"