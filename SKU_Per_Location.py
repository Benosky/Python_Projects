#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 22:01:24 2018

@author: benjaminumeh
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:25:40 2018

@author: benjaminumeh
"""


import pandas as pd
from pandas import ExcelWriter
#import itertools
import time
#import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook


#def sku_per_locn
def locn_bal(md30,int_rate):
    """This function computes the recent "current opening" balance of the various SKUs 
    for a particular location.
    md30: is the recent excel copy of the database (Master Data 3.0)
    sku: is the SKU of the product we want to query"""
    #locn = "All Purpose Pharmacy"
    #Read the excel sheet and assign it to data 
    data = pd.read_excel("md_30.xlsx", sheet_name='Sheet1')
    #Fill NaN with zero to prevent groupby from excluding some columns
    md = data.fillna(0)
    #Delete rows where prev.opening and closing are both zeroes
    #md = md[(md[["prev. opening","closing"]] != 0).all(axis=1)]
    #Create a 'month_year' column
    md['month_year'] = md.date.dt.to_period('M')
    #Convert the 'Price' column to numeric to aid their multiplication
    md["Price"] = pd.to_numeric(md["Price"],errors='coerce')
    md["Average Inventory"] = (md["prev. opening"] + md["closing"])/2.0
    md["Inventory Amount"] = md["Average Inventory"] * md["Price"]
    md["Inventory Finance Cost"] = (md["Inventory Amount"]*((30/float(365))*0.32))#md["Inventory Amount"]
    #Set the "SKU" column as the index
    mdp = md.set_index('location')
    #Load the data into a dataframe
    mdf = pd.DataFrame(mdp)
    #Sort the dataframe according to the Location
    mdf = mdf.sort_values('location',inplace=False)
    #Extract the list of unique index values and convert it from list of unicodes to list of strings
    ind_lc = map(str,list(mdf.index.unique()))
    #Get the list of the unique values in the month_year column
    
    #report = report.set_index('month_year')
    #Create empty lists to hold the dataframes of each location metrics
    sku_count = [] 
    avg_inv_amt = []
    inv_fin_cost = []
    loc_sku = []
    cost_per_sku = []
    for l in ind_lc:
        #Filter the dataframe by the location
        dt = mdf.loc[l]
        #Sort the dataframe by "SKU" and "date"
        dt = dt.sort_values(by=['date','SKU'])
        #Reset the index
        dt = dt.reset_index()
        #Set the year-month as the index
        dt = dt.set_index('month_year')
        yr_months = map(str,list(dt.index.unique()))#report.month_year.unique()
        #Create empty DataFrame to hold location data
        loc_df = pd.DataFrame()
        for y in yr_months:
            dt_m = dt.loc[y]
            #Select the required columns from the dataframe
            dt_m = dt_m[["locationsku","SKU","Inventory Amount","Inventory Finance Cost",'date']]
            dt_m.columns = ['loc_sku('+l+')','sku_cnt('+l+')','Inv_amt('+l+')','inv_cst('+l+')','date']
            #group the dataframe by "locationsku" and compute the mean Inventory Amount and Inventory Finance Cost
            #dt = dt.groupby('SKU',as_index=False)[["Inventory Amount","Inventory Finance Cost"]].mean()
            dt_m = (dt_m.groupby('loc_sku('+l+')')
                    .agg({'sku_cnt('+l+')':'last', 'Inv_amt('+l+')': 'mean','inv_cst('+l+')': 'mean','date':'last'}))
                    #.rename(columns={'SKU':'SKU Count'}))       
            dt_m['sku_cnt('+l+')'] = 1
            #dt_ym = dt_ym.reset_index('date', drop =True)
            dt_m.index = dt_m.date.dt.to_period('M') #dt_m['date']
            dt_ym = dt_m.drop(['date'], axis = 1)
            dt_ym = dt_ym.sum()
            dt_ym = pd.DataFrame(dt_ym)
            dt_ym = dt_ym.T 
            dt_ym.index = [dt_m.index[0]]
            dt_ym['Cst/SKU('+l+')'] = dt_ym['inv_cst('+l+')']/dt_ym['sku_cnt('+l+')']
            dt_ym = dt_ym.round(2)
            loc_df = loc_df.append(dt_ym)
        loc_sku.append(loc_df)
        sku_count.append(loc_df[['sku_cnt('+l+')']])
        avg_inv_amt.append(loc_df[['Inv_amt('+l+')']])
        inv_fin_cost.append(loc_df[['inv_cst('+l+')']])
        cost_per_sku.append(loc_df[['Cst/SKU('+l+')']])
    hist_loc_sku = pd.concat(loc_sku, axis=1, sort=False)
    
    curr_loc_df = pd.DataFrame()
    for m,n in enumerate(ind_lc):#range(1,len(result_cln),5):
        #kpi_df.loc[j, 'SKU_Product'] = ind[j]
        curr_loc_df.loc[m, 'Location'] = n
        curr_loc_df.loc[m, 'Current_SKU_Count'] = hist_loc_sku['sku_cnt('+n+')'][-1]#round(k.iloc[:,2][-1])#Round it up to be a whole number#result[result_cln[29]][-1]#result[result_cln[k]].tail(1)#result[result_cln[k]].iat[-1]#result.iloc[:,k][-1]
        curr_loc_df.loc[m, 'Current Month Avg. Inventory'] = round(hist_loc_sku['Inv_amt('+n+')'][-1],2)#round(k.iloc[:,3][-1])
        curr_loc_df.loc[m, 'Current Month Inventory Cost'] = round(hist_loc_sku['inv_cst('+n+')'][-1],2)#round(k.iloc[:,1][-1],2)
        curr_loc_df.loc[m, 'Current Invt Cost Per SKU'] = round(hist_loc_sku['Cst/SKU('+n+')'][-1],2)#round(k.iloc[:,1][-1],2)
        #lst_dii = result[result_cln[k]]
    curr_loc_df.index += 1
    
    
    hist_cost_per_sku = pd.concat(cost_per_sku, axis=1, sort=False)
    #Trim the column names
    hcpsk = list(hist_cost_per_sku.columns.values)
    cps_loc = [x[8:-1] for x in hcpsk]
    hist_cost_per_sku.columns = cps_loc
    
    
    hist_sku_ploc = pd.concat(sku_count, axis=1, sort=False)
    #Trim the column names
    hsploc = list(hist_sku_ploc.columns.values)
    skcnt_loc = [x[8:-1] for x in hsploc]
    hist_sku_ploc.columns = skcnt_loc
    
    hist_invamt_ploc = pd.concat(avg_inv_amt, axis=1, sort=False)
    #Trim the column names
    hiaploc = list(hist_invamt_ploc.columns.values)
    hiamt = [x[8:-1] for x in hiaploc]
    hist_invamt_ploc.columns = hiamt
    
    hist_invcst_ploc = pd.concat(inv_fin_cost, axis=1, sort=False)
    #Trim the column names
    hinvcl_df = list(hist_invcst_ploc.columns.values)
    invcst_loc = [x[8:-1] for x in hinvcl_df]
    hist_invcst_ploc.columns = invcst_loc
    
    """
    hist_sku_ploc.index = hist_sku_ploc.index.date
    hist_invamt_ploc.index = hist_invamt_ploc.index.date
    hist_invcst_ploc.index = hist_invcst_ploc.index.date
    """
    hist_cost_per_sku = hist_cost_per_sku.T
    hist_cost_per_sku.index.name = "Location"
    hist_cost_per_sku = hist_cost_per_sku.reset_index(inplace=False)
    hist_cost_per_sku.index.name = "Index"

    
    hist_sku_ploc = hist_sku_ploc.T
    hist_sku_ploc.index.name = "Location"
    hist_sku_ploc = hist_sku_ploc.reset_index(inplace=False)
    hist_sku_ploc.index.name = "Index"

    
    hist_invamt_ploc = hist_invamt_ploc.T
    hist_invamt_ploc.index.name = "Location"
    hist_invamt_ploc = hist_invamt_ploc.reset_index(inplace=False)
    hist_invamt_ploc.index.name = "Index"

    
    hist_invcst_ploc = hist_invcst_ploc.T
    hist_invcst_ploc.index.name = "Location"
    hist_invcst_ploc = hist_invcst_ploc.reset_index(inplace=False)
    hist_invcst_ploc.index.name = "Index"

    
    """
    #Interpolate to fill missing numbers
    result_qsld.iloc[:,2:] = result_qsld.iloc[:,2:].interpolate(axis=1)
    result_qsld = result_qsld.round(0)
    
    
    #Interpolate to fill missing numbers
    result_sales.iloc[:,2:] = result_sales.iloc[:,2:].interpolate(axis=1)
    result_sales = result_sales.round(0)
    """
    #Start the Index from 1
    hist_cost_per_sku.index += 1
    hist_sku_ploc.index += 1
    hist_invamt_ploc.index += 1
    hist_invcst_ploc.index += 1
    #hist_sku_ploc.sort_index(axis=1)
    #hist_invamt_ploc.sort_index(axis=1)
    #hist_invcst_ploc.sort_index(axis=1)
    #hist_invcst_ploc
    
    book = load_workbook('KPI_locationsku.xlsx')
    writer = pd.ExcelWriter('KPI_locationsku.xlsx', engine='openpyxl') 
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    #data_filtered.to_excel(writer, "Main", cols=['Diff1', 'Diff2'])
    curr_loc_df.to_excel(writer,sheet_name='Current SKU & Inventory Per Loc',index=True)
    hist_cost_per_sku.to_excel(writer,sheet_name='Hist_Invt_Cost_Per_SKU',index=True)
    hist_sku_ploc.to_excel(writer,sheet_name='Hist_SKU_Per_Location',index=True)
    hist_invamt_ploc.to_excel(writer,sheet_name='Hist_Inventory_Amount_Locn',index=True)
    hist_invcst_ploc.to_excel(writer,sheet_name='Hist_Inventory_Fin_Cost_Per_Loc',index=True)
    
    writer.save()
    skuloc_result = writer.save()#_kpi.save()
    skuloc_result
    return skuloc_result

    
loc_bal = locn_bal("md_30.xlsx",0.32)
loc_bal
    
    