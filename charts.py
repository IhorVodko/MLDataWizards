#!/usr/bin/env python
# coding: utf-8

# In[140]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
import re
import seaborn as sns
from scipy import stats

import matplotlib.pyplot as plt
plt.style.use('dark_background')


# In[2]:


data = pd.read_csv('test_all.csv')


# In[3]:


data['gross_margin'] = data['gross_profit']/data['revenue']
data['operating_margin'] = data['operating_income_(loss)']/data['revenue']
data['P_E_ttm'] = (data['pub_price']*data['shares_(diluted)'])/data['net_income']
data['EV_ebitda_ttm'] = (data['pub_price']*data['shares_(diluted)']+data['long_term_debt']-data['cash,_cash_equivalents_&_short_term_investments'])/data['net_income']
data['int_coverage'] = data['operating_income_(loss)']/-data['interest_expense,_net']
data['FCF_yield'] = (data['net_cash_from_operating_activities']+data['net_cash_from_investing_activities'])/data['operating_income_(loss)']
data['ROA'] = data['net_income']/data['total_assets']
data['ROE'] = data['net_income']/data['total_equity']
data['Div_yield'] = -data['dividends_paid']/data['total_equity']
data['Market Cap']=data['pub_price']*data['shares_(diluted)']
data['Enterprise Value']=data['pub_price']*data['shares_(diluted)']+data['long_term_debt']-data['cash,_cash_equivalents_&_short_term_investments']


# In[62]:


data.columns


# In[4]:


#drop non-sensical data
L= ['EMC','NST','GOOG_old']
data = data[~data.ticker.isin(L)]


# In[5]:




#drop small market capitalization
df = data[data['Market Cap'] > 7000000000]
df = df[df['ROA'] > 0]
df = df[df['EV_ebitda_ttm'] > 0]
df = df[df['EV_ebitda_ttm'] < 50]


# In[14]:


plt.scatter(df['EV_ebitda_ttm'], df['ROA'])
plt.xlabel('EV_ebitda_ttm')
plt.ylabel('ROA')


# In[15]:


#drop small market capitalization and other data that doesn't make sense
df = data[data['Market Cap'] > 7000000000]
df = df[df['ROE'] > 0]
df = df[df['ROE'] < 1]
df = df[df['P_E_ttm'] > 0]
df = df[df['P_E_ttm'] < 30]


# In[16]:


plt.scatter(df['P_E_ttm'], df['ROE'])
plt.xlabel('P_E_ttm')
plt.ylabel('ROE')


# In[10]:


data.shape


# In[17]:


sns.kdeplot(data['Market Cap'], shade=True, label='PDF of Market Cap')


# In[116]:


df2 = data[data['Market Cap'] > 100000]  #> $0B
df2 = df2[df2['Market Cap'] < 10000000000]  #< $10B


sns.kdeplot(df2['Market Cap'], shade=True, label='PDF of Market Cap')


# In[19]:


df3 = data[data['Market Cap'] > 10000000000]  #> $10B
df3 = df3[data['Market Cap'] < 100000000000]  #< $100B


sns.kdeplot(df3['Market Cap'], shade=True, label='PDF of Market Cap')


# In[20]:


df4 = data[data['Market Cap'] > 10000000000]  #> $100B
df4 = df4[data['Market Cap'] < 1000000000000]  #< $1T


sns.kdeplot(df4['Market Cap'], shade=True, label='PDF of Market Cap')


# In[21]:


df2 = data[data['Market Cap'] > 0]
df2 = df2[df2['Market Cap'] < 1000000000000]
plt.hist(df2['Market Cap'], bins=7, color="#5ee3ff")
plt.xlabel('Market Capitalization')
plt.ylabel('Count')


# In[145]:


df2 = data[data['Market Cap'] > 10000000000]
df2 = df2[df2['Market Cap'] < 100000000000000]
plt.hist(df2['Market Cap'], bins=5, color="#5ee3ff")
plt.xlabel('Market Capitalization')
plt.ylabel('Count')


# In[23]:



df2 = data[data['Market Cap'] > 100000000000]
df2 = df2[df2['Market Cap'] < 1000000000000]
plt.hist(df2['Market Cap'], bins=8, color="#5ee3ff")
plt.xlabel('Market Capitalization')
plt.ylabel('Count')


# In[ ]:





# In[24]:


data = data[data['Market Cap'] > 0]
data = data[data['Market Cap'] < 1000000000000]

data['MktCapBin'] = pd.cut(data['Market Cap'], 5, labels=False)
X = data.groupby('MktCapBin')[['ticker', 'Market Cap']].agg({'Market Cap':'mean', 'ticker':'count'})
X.columns = ['Market Cap', 'count']
X


# In[ ]:





# In[25]:


#Merge industry and company information
ind = pd.read_csv('industries.csv',sep=';')
comp = pd.read_csv('us-companies.csv',sep=';')


#pd.merge(df3, df4, how='inner', on ='col2')


# In[26]:


comp
comp.columns = ['ticker', 'SimFinID', 'Company Name','IndustryId']
data = pd.merge(data, comp, how='inner', on='ticker')


# In[27]:


data = pd.merge(data, ind, how='inner', on='IndustryId')


# In[28]:


data.head(10)


# In[ ]:


df.groupby('Sector')['P_E_ttm'].median().sort_values(ascending=False,).plot.bar(color='#5ee3ff')
plt.ylabel('Median P/E Ratio_ttm')


# In[141]:



data.groupby('Sector')['ticker'].count().sort_values(ascending=False,).plot.bar(color='#5ee3ff')
plt.ylabel('Companies per Sector')


# In[146]:


df = data[data['P_E_ttm'] > 0]
df.groupby('Sector')['P_E_ttm'].median().sort_values(ascending=False,).plot.bar(color='#5ee3ff')
plt.ylabel('Median P/E Ratio_ttm')


# In[36]:


df = data[data['EV_ebitda_ttm'] > 0]
df = df[df['EV_ebitda_ttm'] < 25]
df.groupby('Sector')['EV_ebitda_ttm'].median().sort_values(ascending=False,).plot.bar(color='#5ee3ff')
plt.ylabel('Median EV_ebitda_ttm')


# In[41]:


df = data[data['ROA'] > 0]
df.groupby('Sector')['ROA'].mean().sort_values(ascending=False,).plot.bar(color='#5ee3ff')
plt.ylabel('Return on Assets')


# In[ ]:





# In[147]:


df = data[data['ROE'] > 0]
df.groupby('Sector')['ROE'].median().sort_values(ascending=False,).plot.bar(color='#5ee3ff')
plt.ylabel('Return on Equity')


# In[43]:


df = data[data['gross_margin'] > 0]
df.groupby('Sector')['gross_margin'].median().sort_values(ascending=False,).plot.bar(color='#5ee3ff')
plt.ylabel('Gross Margin')


# In[108]:


df = data[data['operating_margin'] > 0]
df.groupby('Sector')['operating_margin'].median().sort_values(ascending=False,).plot.bar(color='#5ee3ff')
plt.ylabel('operating_margin')


# In[47]:


df = data[data['Div_yield'] > 0]
df = df[df['Div_yield'] < .3]
df.groupby('Sector')['Div_yield'].median().sort_values(ascending=False,).plot.bar(color='#5ee3ff')
plt.ylabel('Div_yield')


# In[151]:


df = data[data['cash_from_(repurchase_of)_equity'] < 0]

df.groupby('Sector')['cash_from_(repurchase_of)_equity'].mean().sort_values(ascending=False,).plot.bar(color='#5ee3ff')
plt.ylabel('Stock Repurchase')


# In[49]:


IS = pd.read_csv('us-income-annual.csv',sep=';')
IS


# In[55]:


#df['C'] = df.B.pct_change().mul(100).round(2)

IS=IS.sort_values(['Ticker','Fiscal Year'])[['Ticker','Fiscal Year','Revenue','Net Income']]


# In[56]:


df = IS.groupby(['Ticker']).agg({"Revenue":["first","last"],"Net Income":["first","last"],"Ticker":["count"]})
df['Rev_growth'] = (df['Revenue']['last'] / df['Revenue']['first']-1)/df['Ticker']['count']
df['Inc_growth'] = (df['Net Income']['last'] / df['Net Income']['first']-1)/df['Ticker']['count']
#df['count']
#df.drop('Revenue', inplace=True)
#growth=df['Rev_growth','Inc_growth']
df


# In[57]:


#manually eliminate levels
df.to_csv('df.csv')


# In[50]:


#read data file back into data frame to fix mult levels
growth = pd.read_csv('df.csv')


# In[51]:


growth = growth.rename(columns={'Ticker':'ticker'})
growth


# In[52]:


data = pd.merge(data, growth, how='inner', on='ticker')
data


# In[148]:


#df = data[data['Div_yield'] > 0]
#df = df[df['Div_yield'] < .3]

data.groupby('Sector')['Rev_growth'].median().sort_values(ascending=False,).plot.bar(color='#5ee3ff')
plt.ylabel('Median Revenue Growth')


# In[149]:


df = data[data['Inc_growth'] > -.25]
df = df[df['Inc_growth'] < 5]
df.groupby('Sector')['Inc_growth'].median().sort_values(ascending=False,).plot.bar(color='#5ee3ff')
plt.ylabel('Median Income Growth')


# In[69]:


df = data[data['Market Cap'] > 5000000000]
df = df[df['Inc_growth'] < 1]
df = df[df['Inc_growth'] > -1]
df = df[df['EV_ebitda_ttm'] < 30]
df = df[df['EV_ebitda_ttm'] > 0]


plt.scatter(df['EV_ebitda_ttm'], df['Inc_growth'])
plt.xlabel('EV_ebitda_ttm')
plt.ylabel('Inc_growth')


# In[67]:


df = data[data['Market Cap'] > 1000000000]
df = df[df['Inc_growth'] > -1]
#df = df[df['cash_from_(repurchase_of)_equity'] < -1]
#df = df[df['cash_from_(repurchase_of)_equity'] > 0]


plt.scatter(df['cash_from_(repurchase_of)_equity'], df['Inc_growth'])
plt.xlabel('cash_from_(repurchase_of)_equity')
plt.ylabel('Inc_growth')


# In[124]:


df = data[data['Rev_growth'] < .6]

df = df[df['Market Cap'] > 1000000000]
df = df[df['Market Cap'] < 10000000000]
sns.lmplot("Market Cap", "Rev_growth", df)


# In[150]:


df = data[data['Market Cap'] > 0]
df = df[df['Market Cap'] < 100000000000000]
df = df[df['Rev_growth'] < .75]
plt.hist(df['Rev_growth'], bins=8, color="#5ee3ff")
plt.xlabel('Revenue Growth')
plt.ylabel('Count')


# In[132]:


data.columns


# In[137]:


cluster = data[['Market Cap','Rev_growth','Inc_growth','ROE','Div_yield','gross_margin','operating_margin','ROA','int_coverage','P_E_ttm','EV_ebitda_ttm']]

cluster = cluster[cluster['Market Cap'] > 1000000000]

cluster.to_csv('cluster.csv')


# In[ ]:




