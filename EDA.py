#!/usr/bin/env python
# coding: utf-8

# In[51]:


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
#plt.style.available


# In[56]:


data = pd.read_csv('test.csv')
data1=data


# In[84]:


data.columns


# In[86]:


#columns = ['Unnamed: 0', 'pub_price', 'f1_price','f1_year_date','key_py_fy','ticker', 'publish_date', 'shares_(basic)', 'shares_(diluted)']

columns = ['ticker']


data.drop(columns, inplace=True, axis=1)


# In[203]:


#create ratios for clustering

data['gross_margin'] = data['gross_profit']/data['revenue']
data['operating_margin'] = data['operating_income_(loss)']/data['revenue']
data['P_E_ttm'] = (data['pub_price']*data['shares_(diluted)'])/data['net_income']
data['EV_ebitda_ttm'] = (data['pub_price']*data['shares_(diluted)']+data['long_term_debt'])/data['net_income']
data['int_coverage'] = data['operating_income_(loss)']/-data['interest_expense,_net']
data['FCF_yield'] = (data['net_cash_from_operating_activities']+data['net_cash_from_investing_activities'])/data['operating_income_(loss)']
data['ROA'] = data['net_income']/data['total_assets']
data['ROE'] = data['net_income']/data['total_equity']
data['Div_yield'] = data['dividends_paid']/data['total_equity']


# In[87]:


data.head()


# In[88]:


#create new data frame of ratios
#ratios = data[['gross_margin', 'operating_margin','P_E_ttm','EV_ebitda_ttm','int_coverage','FCF_yield','ROA','ROE','Div_yield']].copy()

#ratios = data[['EV_ebitda_ttm','ROA']].copy()

ratios = data


# In[34]:


missing = ratios.isna().sum()
missing = missing[missing>0]
missing_perc = missing/ratios.shape[0]*100
na = pd.DataFrame([missing, missing_perc], index = ['missing_num', 'missing_perc']).T
na = na.sort_values(by = 'missing_perc', ascending = False)
na


# In[208]:


ratios.fillna(0, inplace=True)


# In[89]:


#Clustering Financial Data

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward') 
hier = AgglomerativeClustering()


# In[90]:


import pandas as pd

hier.fit(ratios)
label = hier.labels_


# In[91]:


from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

def linkage_frame(data):
    row_clusters = linkage(data, method='average', metric='euclidean')
    columns = ['row label 1', 'row label 2', 'distance', 'no. items in clust.']
    index = ['cluster %d' % (i + 1) for i in range(row_clusters.shape[0])]
    linkage_df = pd.DataFrame(row_clusters, columns=columns, index=index)
    return linkage_df


# In[92]:


linkage_df = linkage_frame(ratios.values)
linkage_df.head()


# In[217]:


row_dendr = dendrogram(linkage_df, leaf_rotation=90, leaf_font_size=8)
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()


# In[93]:


row_dendr = dendrogram(linkage_df, leaf_rotation=90, truncate_mode='lastp', p = 20, leaf_font_size=8)
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()


# In[94]:


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
cluster.fit_predict(ratios)


# In[95]:


plt.figure(figsize=(10, 7))
plt.scatter(ratios['total_assets'], ratios['net_cash_from_operating_activities'], c=cluster.labels_, cmap='rainbow')


# In[100]:


df=cluster.fit_predict(ratios)
df = pd.DataFrame(df)
df2=pd.concat([df, data1], axis = 1)


# In[101]:


df2.head()
df2.to_csv('clusters.csv')


# In[99]:


data1.columns


# In[ ]:


ratios.to_csv('ratios.csv')


#K Means Clustering


# In[238]:


ratios = data[['gross_margin','P_E_ttm','EV_ebitda_ttm','FCF_yield','ROA','ROE']].copy()
ratios.to_csv('ratios.csv')


# In[21]:



ratios = pd.read_csv('train_Kmean.csv')


# In[24]:


ratios.columns


# In[4]:


#K Means Clustering
from __future__ import print_function
from sklearn.cluster import KMeans
kmeans = KMeans()


# In[46]:


import pandas as pd
from sklearn.cluster import KMeans
from math import sqrt
import  pylab as pl
import numpy as np

X =  ratios.values #Converting ret_var into nummpy array
sse = []
for k in range(2,15):
    
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(X)
    
    sse.append(kmeans.inertia_) #SSE for each n_clusters
pl.plot(range(2,15), sse)
pl.title("Elbow Curve")
pl.show()


# In[48]:


kmeans.set_params(n_clusters=6)
kmeans.fit(ratios)


# In[248]:


kmeans.cluster_centers_


# In[49]:


kmeans.labels_


# In[52]:


plt.scatter(ratios['total_assets'], ratios['net_cash_from_operating_activities'], c=kmeans.labels_, alpha=0.8)
#plt.scatter(kmeans.cluster_centers_['EV_ebitda_ttm'], kmeans.cluster_centers_['ROA'], marker="+", s=1000, c=[0, 1, 2])
plt.xlabel('total_assets')
plt.ylabel('net_cash_from_operating_activities')
plt.show()


# In[59]:


df=kmeans.labels_
df = pd.DataFrame(df)
df2=pd.concat([df, data1], axis = 1)
df2


# In[40]:


cluster = pd.read_csv('cluster.csv')


# In[ ]:





# In[41]:


np.sum(cluster.isnull())


# In[42]:


kmeans.set_params(n_clusters=10)
kmeans.fit(cluster)


# In[29]:


kmeans.cluster_centers_


# In[30]:


plt.scatter(cluster['Rev_growth'], cluster['ROE'], c=kmeans.labels_, alpha=0.8)
#plt.scatter(kmeans.cluster_centers_['EV_ebitda_ttm'], kmeans.cluster_centers_['ROA'], marker="+", s=1000, c=[0, 1, 2])
plt.xlabel('Rev_growth')
plt.ylabel('ROE')
plt.show()


# In[61]:


df2.to_csv('cluster.csv')