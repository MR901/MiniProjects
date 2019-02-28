
# coding: utf-8

# In[ ]:


Medium range forecast


# ## Answer the following
# 
# 1. Aggregate the Sales_Qty for each Store-SKU at a month level; detect any Outliers in the Sales_Qty for each Store-SKU combination and apply an outlier treatment on the same.
# Specify the outlier treatment technique.  
# 
# 
# 2. Estimate the level of promotions (Discount%) for each Category-Store level at a month level - remove any outliers / inconsistencies from this, and specify the technique used; the level of promotions is defined as $$Discount\% = (1 - sum of SP / sum of MRP)$$  
# 
# 
# 3. Estimate the inventory levels at a weekly level for each Store-SKU by interpolating missing values from data on secondary and primary sales; the following equation holds true in general: (you can do this for a shorter period of Jan 2017 to Mar 2017) 
# $$Closing\ inventory\ on\ day\ [t] = Closing\ inventory\ on\ day\ [t-1] $$
# $$                                - Secondary(sales - returns)\ on\ day\ [t] $$
# $$                                + Primary (sales - returns)\ on\ day\ [t]$$
#     NOTE:
#         a. Secondary sales is the file named “WC_DS_Ex1_Sec_Sales.csv” - and it refers to sales from stores to customers (and returns by customers)
#         b. Primary sales is the file name “WC_DS_Ex1_Pri_Sales.csv” - and it refers to stock movements from retailer WH to stores (and returns back to WH)
#         c. Returns in both datasets are indicated by negative values in ‘Sales_Qty’ and ‘Qty’ fields respectively  
# 
# 
# 4. The inventory estimations in Question 3 will have data inconsistencies - take any assumption to resolve them and explain that assumption  
# 
# 
# 5. Using the Secondary sales data and inventory series from Question 3, determine average out-of-stock percentage (OOS%) for each Category-Store combination at a monthly level; the OOS % is defined as: 
# $$ OOS\% = 1 - \{Average\ of\ no.\ of\ unique\ SKUs\ in\ stock\ each\ day / No.\ of\ unique\ SKUs\ in\ stock\ over\ the\ entire\ month\}$$ (for each Category-Store combination each month)
# ( Again - do this for a short period of Jan 2017 - Mar 2017; for forecasting you can
# assume that the retailer will experience similar OOS levels in Jan-Mar 2018 )   
# 
# 
# 6. Using the historical secondary sales, inventory series, OOS% levels and promotion levels, determine the demand for each Store-SKU combination at a monthly level for the forecast period; use any forecasting technique that you’re comfortable with (you may use multiple techniques)  
# 
# 
# 7. Explain the approach for Question 6 clearly e.g. dividing data into train, validation and test sets, choice of technique used, metric(s) used to evaluate the results   
# 
# 
# 8. If any of the above steps is becoming computationally too expensive or taking too long; you are free to either simplify them or reduce the complexity (e.g. impute weekly inventory positions instead of daily)   

# In[5]:


import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
# import statsmodels.api as sm

from lib import datasetPrimAnalysis, namestr, splitTimeSeriesData
# warnings.filterwarnings("ignore")
# plt.style.use('fivethirtyeight')
# matplotlib.rcParams['axes.labelsize'] = 14
# matplotlib.rcParams['xtick.labelsize'] = 12
# matplotlib.rcParams['ytick.labelsize'] = 12
# matplotlib.rcParams['text.color'] = 'k'


# In[2]:


## Reading Data
# df = pd.read_excel("Superstore.xls")

primDf = pd.read_csv('data/WC_DS_Ex1_Pri_Sales.csv')
secDf = pd.read_csv('data/WC_DS_Ex1_Sec_Sales.csv')
invDf = pd.read_csv('data/WC_DS_Ex1_Inv.csv')

df_li = {'Secondary': secDf, 
         'Primary':primDf, 
         'Inventory': invDf
        }


# In[8]:


## transfering the feature "SKU_Code" to object and then viewing the result
for df_name in df_li:
    print('"{}" dataframe shape:  {}'.format(df_name, df_li[df_name].shape))
    
    ## Changing feature data type to object
    df_li[df_name][['SKU_Code']] = df_li[df_name][['SKU_Code']].astype(str)
    
    ## Changing feature data type to datetime
    # df_li[df_name]['Date'] = [ datetime.strptime(ele, '%Y-%m-%d') for ele in df_li[df_name]['Date'] ] # %H:%M:%S
    df_li[df_name]['Date'] = pd.to_datetime(df_li[df_name]['Date'],format='%Y-%m-%d') 
    
    ## sorting df based on Date
    df_li[df_name].sort_values(by=['Date'], inplace=True)
    df_li[df_name].reset_index(drop=True, inplace=True)
    
    display(df_li[df_name].head())
    _ = datasetPrimAnalysis(df_li[df_name])
    print('****'*25,'\n\n')


# In[6]:


## splitting the dataset

split_date = pd.datetime(2017, 6, 1)
for df_name in df_li:
    '''    x='qwe'; x1= 'asd';  exec("%s = %d; %s = %d" % (x,2,x1,3)); print(qwe, asd)
    '''
    print(df_name)
    varName = namestr(df_li[df_name], globals())[0]
    exec("{0}, {1} = splitTimeSeriesData(df_li[df_name], split_date)".format(varName+'_train', varName+'_test'))

print('\nOriginal "Primary" DataFrame Shape: {} \n\tTrain Shape: {}\tTest Shape:{}'.format(
    primDf.shape, primDf_train.shape, primDf_test.shape))


# In[9]:


## Analyzing Primary DataSet: Stores Sales

'''
Aggregate the Sales_Qty for each Store-SKU at a month level; detect any Outliers in the Sales_Qty 
for each Store-SKU combination and apply an outlier treatment on the same. Specify the outlier 
treatment technique.
'''

tempDF = primDf.copy()

# primDf.agg({'A' : ['sum', 'min'], 'B' : ['min', 'max']})


# In[57]:


a= tempDF['Date'][1]
a.strftime(format= '%Y-%m')

tempDF['YYMM'] = tempDF['Date'].apply(lambda x: x.strftime(format= '%Y-%m'))



# tempDF.agg({'YYMM' : ['sum', 'min'], 'Qty' : ['min', 'max', 'sum']})

# tempDF.apply(np.min, axis=0)

gk = tempDF.groupby(by = ['YYMM', 'Store_Code', 'SKU_Code'], axis=0)
gk.first()
gk.Qty.agg(np.sum)

gk = tempDF.groupby(by = ['YYMM', 'Store_Code'], axis=0)
gk.first()
gk.Qty.agg(np.sum)


# In[30]:


tempDF.iloc[0,:]['Date']


# In[ ]:


a=secDf[secDf["SKU_Code"].str.contains("603132")] # -1 signifies stock empty


# In[ ]:


a.loc[a['Sales_Qty'] == -1,: ]


# In[ ]:


primDf_train.set_index('Date', inplace=True)
display(primDf_train.head())


# In[ ]:


ts = primDf_train['Qty']


# In[ ]:


## Dickey Fuller Test: This is one of the statistical tests for checking stationarity.

from statsmodels.tsa.stattools import adfuller
ts = primDf_train['Qty']

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    # rolmean = pd.rolling_mean(timeseries, window=12)
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    #Plot rolling statistics:
    plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
test_stationarity(ts)

