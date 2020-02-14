
import pandas as pd
from pandas import ExcelWriter
#import itertools
import time
import matplotlib.pyplot as plt
import numpy as np
"""
#def ocb_signal(md30,loc):
data = pd.read_excel("md_30.xlsx", sheet_name='Sheet1')
#fill all NaN with zeroes
md = data.fillna(0)
#Delete rows where sold and prev.opening are both zeroes
md = md[(md[["sold","prev. opening"]] != 0).all(axis=1)]
#Concatenate the SKU and product name
md['SKU_Prod']=md['SKU'].astype(str)+'-'+md['product']#+'_'+master_data['SKU'].astype(str)

#Compute the Inventory Turnover Ratio: The number of times inventory is sold, or the portion of
#inventory sold in the time period
md["Inv_Turn_ratio"] = (md["sold"]/md["prev. opening"])
#compute the Days in Inventory: The Average number of days to sell the inventory
days = 7
md["DII"] = days/md["Inv_Turn_ratio"]
#Compute rolling(monthly average) DII
md["Rolling_DII"] = md["DII"].rolling(8, min_periods=1).mean()

#compute the rolling minimum Inventory Turnover Ratio
rollmean = md["Inv_Turn_ratio"].rolling(6, min_periods=1).mean()

#Compute the rate of dispersion of the Inv_Turn_ratio
ITR_std = md["Inv_Turn_ratio"].std()

#Compute the redline
md["Red_Line"] = rollmean-(rollmean*(ITR_std*1.5))# - (1-rollmin)*0.025

#Compute the signal
md["Signal"] = (md["Red_Line"] - md["Inv_Turn_ratio"]).clip(0)# - (1-rollmin)*0.025
#Set the 'location' column as the index of dataframe md and assign it to mdp
mdp = md.set_index('SKU_Prod')
#mdp = md.set_index('SKU')
mdf = pd.DataFrame(mdp)
mdf.sort_values('SKU',inplace=True)
#Extract the list of unique index values and convert it from list of unicodes to list of strings
ind = map(str,list(mdf.index.unique()))

"""
"""
    
#Create an empty list to hold the dataframes of each SKU_Product
sku_prod = []
#Filter the rows for each SKU_product
for i in ind:
    #Filter rows by sku
    dt = (mdf.filter(like=i, axis=0)).set_index('date')[["Inv_Turn_ratio","DII","Rolling_DII","Red_Line","Signal"]].clip(0)
    #dt = mdp.loc[i]#.set_index('date')[["Inv_Turn_ratio","DII","Red_Line"]]
    #dt = dt.set_index('date')[["Inv_Turn_ratio","DII","Red_Line"]]
    #dt.columns = ['ITR('+i[:12]+')','DII('+i[:12]+')','RollDII('+i[:12]+')','RL('+i[:12]+')','Signal('+i[:12]+')']
    dt.columns = ['ITR('+i+')','DII('+i+')','RollDII('+i+')','RL('+i+')','Signal('+i+')']
    dt = dt.groupby('date').mean() # Produces Pandas DataFrame
    sku_prod.append(dt)
sku_prod


#for k in range(len(df_lev)):
kpi_df = pd.DataFrame()
#result_cln = list(result.columns.values)
#result[result_cln[1]].tail(1)#[-1]
#for j in range(len(ind)):
    #for k in range(1,len(result_cln),4):
for j,k in enumerate(sku_prod):#range(1,len(result_cln),5):
    kpi_df.loc[j, 'SKU_Product'] = ind[j]
    kpi_df.loc[j, 'DII'] = round(k.iloc[:,1][-1])#result[result_cln[29]][-1]#result[result_cln[k]].tail(1)#result[result_cln[k]].iat[-1]#result.iloc[:,k][-1]
    kpi_df.loc[j, 'OCB_Signal'] = "RED" if k.iloc[:,0][-1]< k.iloc[:,3][-1] else "OK"
    #lst_dii = result[result_cln[k]]
kpi_df    
"""

import pandas as pd
from pandas import ExcelWriter
#import itertools
import time
import matplotlib.pyplot as plt
import numpy as np

#def ocb_signal(md30,loc):
data = pd.read_excel("md_30.xlsx", sheet_name='Sheet1')
#fill all NaN with zeroes
md = data.fillna(0)
#Delete rows where sold and prev.opening are both zeroes
md = md[(md[["sold","prev. opening"]] != 0).all(axis=1)]
#Concatenate the SKU and product name
md['SKU_Prod']=md['SKU'].astype(str)+'-'+md['product']#+'_'+master_data['SKU'].astype(str)
#Concatenate locationsku and product name
md['locationsku_Prod']=md['locationsku'].astype(str)+'-'+md['product']#+'_'+master_data['SKU'].astype(str)
#Compute the Inventory Turnover Ratio: The number of times inventory is sold, or the portion of
#inventory sold in the time period
md["Inv_Turn_ratio"] = (md["sold"]/md["prev. opening"])
#compute the Days in Inventory: The Average number of days to sell the inventory
days = 7
md["DII"] = days/md["Inv_Turn_ratio"]
#Set the 'SKU_Prod' column as the index of dataframe md and assign it to mdp
mdp = md.set_index('SKU_Prod')
#mdp = md.set_index('SKU')
mdf = pd.DataFrame(mdp)
#Sort the dataframe according to the SKU
mdf.sort_values('SKU',inplace=True)
#Extract the list of unique index values and convert it from list of unicodes to list of strings
ind = map(str,list(mdf.index.unique()))

#Create an empty list to hold the dataframes of each SKU_Product
sku_prod = []
#Filter the rows for each SKU_product
for i in ind:
    #Filter rows by sku
    dt = (mdf.filter(like=i, axis=0)).set_index('date').clip(0)
    #dt = mdp.loc[i]#.set_index('date')[["Inv_Turn_ratio","DII","Red_Line"]]
    #dt = dt.set_index('date')[["Inv_Turn_ratio","DII","Red_Line"]]
    #dt.columns = ['ITR('+i[:12]+')','DII('+i[:12]+')','RollDII('+i[:12]+')','RL('+i[:12]+')','Signal('+i[:12]+')']
    #compute the Days in Inventory: The Average number of days to sell the inventory
    #Compute rolling(monthly average) DII
    dt["Rolling_DII"] = dt["DII"].rolling(8, min_periods=1).mean()
    
    #Compute the rolling Inventory Turnover Ratio
    dt["Rolling_ITR"] = dt["Inv_Turn_ratio"].rolling(8, min_periods=1).mean()
    #compute the mean rolling Inventory Turnover Ratio
    #rollmean = dt["Inv_Turn_ratio"].rolling(8, min_periods=1).mean()
    
    #Compute the rate of dispersion of the Inv_Turn_ratio
    ITR_std = dt["Inv_Turn_ratio"].std()
    
    #Compute the redline
    dt["Red_Line"] = dt["Rolling_ITR"]-(dt["Rolling_ITR"]*(ITR_std*1.5))# - (1-rollmin)*0.025
    
    #Compute the signal
    dt["Signal"] = (dt["Red_Line"] - dt["Inv_Turn_ratio"]).clip(0)# - (1-rollmin)*0.025
    dt = dt[["Inv_Turn_ratio","DII","Rolling_DII","Red_Line","Signal","Rolling_ITR"]]
    dt.columns = ['ITR('+i+')','DII('+i+')','RollDII('+i+')','RL('+i+')','Signal('+i+')','RollingITR('+i+')'] 
    #Fill NaN with zero to prevent groupby from excluding some columns
    dt = dt.fillna(0)
    dt = dt.groupby('date').mean() # Produces Pandas DataFrame
    #dt = dt.resample('D').mean() # Produces Pandas DataFrame
    sku_prod.append(dt)
sku_prod
#result = pd.concat([df1, df4], axis=1, join_axes=[df1.index])

#for k in range(len(df_lev)):
kpi_df = pd.DataFrame()
#result_cln = list(result.columns.values)
#result[result_cln[1]].tail(1)#[-1]
#for j in range(len(ind)):
    #for k in range(1,len(result_cln),4):
for j,k in enumerate(sku_prod):#range(1,len(result_cln),5):
    kpi_df.loc[j, 'SKU_Product'] = ind[j]
    kpi_df.loc[j, 'DII'] = round(k.iloc[:,1][-1])#Round it up to be a whole number#result[result_cln[29]][-1]#result[result_cln[k]].tail(1)#result[result_cln[k]].iat[-1]#result.iloc[:,k][-1]
    kpi_df.loc[j, 'RollingDII'] = round(k.iloc[:,2][-1])
    kpi_df.loc[j, 'ITR'] = round(k.iloc[:,0][-1],2)
    kpi_df.loc[j, 'RollingITR'] = round(k.iloc[:,-1][-1],2)
    kpi_df.loc[j, 'OCB_Signal'] = "RED" if k.iloc[:,0][-1]< k.iloc[:,3][-1] else "OK"
    #lst_dii = result[result_cln[k]]
kpi_df


#Sort dataframe md by loaction and SKU, and assign it to mdl
mdl = pd.DataFrame(md.sort_values(by=['location', 'SKU']))

#Set the 'locationsku_Prod' column as the index of dataframe mdl and assign it to mdlf
mdlf = mdl.set_index('locationsku_Prod')
#mdp = md.set_index('SKU')

#Extract the list of unique index values and convert it from list of unicodes to list of strings
ind_ls = map(str,list(mdlf.index.unique()))

#Extract the list of unique SKU_product names. and convert it from list of unicodes to list of strings
#prod_ls = map(str,list(mdlf.SKU_Prod.unique()))
#prod_ls

#Create an empty list to hold the dataframes of each SKU_Product
loctn_sku = []
#Filter the rows for each locationsku
for l in ind_ls:
    #Filter rows by sku
    dt_ls = (mdlf.filter(like=l, axis=0)).set_index('date').clip(0)
    #Compute rolling(monthly average) DII
    dt_ls["Rolling_DII"] = dt_ls["DII"].rolling(8, min_periods=1).mean()
    #Compute the rolling Inventory Turnover Ratio
    dt_ls["Rolling_ITR"] = dt_ls["Inv_Turn_ratio"].rolling(8, min_periods=1).mean()
    #compute the mean rolling Inventory Turnover Ratio
    rollmean_ls = dt_ls["Inv_Turn_ratio"].rolling(8, min_periods=1).mean()
    #Compute the rate of dispersion of the Inv_Turn_ratio
    ITR_std_ls = dt_ls["Inv_Turn_ratio"].std()
    
    #Compute the redline
    dt_ls["Red_Line"] = rollmean_ls-(rollmean_ls*(ITR_std_ls*1.5))# - (1-rollmin)*0.025
    
    #Compute the signal
    dt_ls["Signal"] = (dt_ls["Red_Line"] - dt_ls["Inv_Turn_ratio"]).clip(0)# - (1-rollmin)*0.025
    dt_ls = dt_ls[["Inv_Turn_ratio","DII","Rolling_DII","Red_Line","Signal","Rolling_ITR"]]
    dt_ls.columns = ['ITR('+l+')','DII('+l+')','RollDII('+l+')','RL('+l+')','Signal('+l+')','RollingITR('+l+')'] 
    #Fill NaN with zero to prevent groupby from excluding some columns
    dt_ls = dt_ls.fillna(0)
    dt_ls = dt_ls.groupby('date').mean() # Produces Pandas DataFrame
    #dt = dt.resample('D').mean() # Produces Pandas DataFrame
    loctn_sku.append(dt_ls)
loctn_sku

kpi_ls = pd.DataFrame()
#result_cln = list(result.columns.values)
#result[result_cln[1]].tail(1)#[-1]
#for j in range(len(ind)):
    #for k in range(1,len(result_cln),4):
for m,n in enumerate(loctn_sku):#range(1,len(result_cln),5):
    kpi_ls.loc[m, 'SKU_Product'] = ind_ls[m]
    kpi_ls.loc[m, 'DII'] = round(n.iloc[:,1][-1])#Round it up to be a whole number#result[result_cln[29]][-1]#result[result_cln[k]].tail(1)#result[result_cln[k]].iat[-1]#result.iloc[:,k][-1]
    kpi_ls.loc[m, 'RollingDII'] = round(n.iloc[:,2][-1])
    kpi_ls.loc[m, 'ITR'] = round(n.iloc[:,0][-1],2)
    kpi_ls.loc[m, 'RollingITR'] = round(n.iloc[:,-1][-1],2)
    kpi_ls.loc[m, 'OCB_Signal'] = "RED" if n.iloc[:,0][-1]< n.iloc[:,3][-1] else "OK"
    #lst_dii = result[result_cln[k]]
kpi_ls

writer_kpi = ExcelWriter('KPI_locationsku.xlsx', engine='xlsxwriter')
#write_lsku = ExcelWriter('KPI_locationsku.xlsx')
#write the single dataframe to excel
kpi_df.to_excel(writer_kpi,sheet_name='Sheet1',index=False)
kpi_ls.to_excel(writer_kpi,sheet_name='Sheet2',index=False)
# Get the xlsxwriter workbook and worksheet objects.
workbook  = writer_kpi.book
worksheet1 = writer_kpi.sheets['Sheet1']
worksheet2 = writer_kpi.sheets['Sheet2']

format1 = workbook.add_format({'bg_color': '#FFC7CE',
                               'font_color': '#9C0006'})
format2 = workbook.add_format({'num_format': '0%'})

# Apply a conditional format to the cell range.

worksheet1.set_column('A:A', 45)
worksheet1.set_column('B:B', 6)
worksheet1.set_column('C:C', 15)
worksheet1.set_column('D:D',7, format2)
worksheet1.set_column('E:E',7, format2)

worksheet2.set_column('A:A', 65)
worksheet2.set_column('B:B', 6)
worksheet2.set_column('C:C', 15)
worksheet2.set_column('D:D',7, format2)
worksheet2.set_column('E:E',7, format2)

worksheet1.conditional_format('F2:F{}'.format(len(kpi_df)+1), {'type':'cell', 'criteria': '=','value':'RED', 'format': format1})
worksheet2.conditional_format('F2:F{}'.format(len(kpi_ls)+1), {'type':'cell', 'criteria': '=','value':'RED', 'format': format1})
#save the excel in directory folder
kpi_lsku_result = writer_kpi.save()
kpi_lsku_result



