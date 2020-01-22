import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
from scipy import stats
plt.style.use('ggplot')

#load data after imputations have been added
house = pd.read_csv('train_imputed.csv', index_col=0)
missing = house.isna().sum()
missing = missing[missing>0]
missing_perc = missing/house.shape[0]*100
na = pd.DataFrame([missing, missing_perc], index = ['missing_num', 'missing_perc']).T
na = na.sort_values(by = 'missing_perc', ascending = False)
na

# multicollinearity issue. drop the TotRmsAbvGrd, GarageYrBlt and GarageArea features as they have less correlation with the SalePrice compared to GarageCars,YearBuilt and GrLivArea.

#drop columns
house.drop(['TotRmsAbvGrd', 'GarageArea', 'GarageYrBlt'], axis =1, inplace = True)

#variable engineering
mp = {'Ex': 5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'dne':0}
for feat in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
 'HeatingQC', 'KitchenQual','GarageQual', 'GarageCond', ]:
    house[feat] = house[feat].map(mp) 
mp = {'N':0, 'Y':2 , 'P':1}
for feat in ['CentralAir', 'PavedDrive']:
 house[feat] = house[feat].map(mp)
mp = {'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1}
house['Functional'] = house['Functional'].map(mp)
mp = {'Gtl':1 ,'Mod':2 , 'Sev':3}
house['LandSlope'] = house['LandSlope'].map(mp)

#additionaL changes
mp = {'Ex':5 ,'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'dne':0}
house['FireplaceQu'] = house['FireplaceQu'].map(mp)


##variable engineering
house['TotBath'] = house['BsmtFullBath']+house['FullBath']+.5*(house.BsmtHalfBath+house.HalfBath)
house['Total_area'] = house['X1stFlrSF']+house['X2ndFlrSF']+house.TotalBsmtSF
house['Overall_Score'] = house.OverallQual*house.OverallCond
house['Garage_Score'] = house.GarageQual*house.GarageCond
house['Kitchen_Score'] = house.KitchenAbvGr*house.KitchenQual
house['Bsmt_Score'] = house.BsmtQual*house.BsmtCond

#additionaL changes
house['FirePlace Score'] = house.Fireplaces*house.FireplaceQu


#drop features which have been engineered into a single column
house.drop(['BsmtFullBath', 'FullBath','BsmtHalfBath','HalfBath','X1stFlrSF','X2ndFlrSF','TotalBsmtSF', 'GarageQual','GarageCond',
	'KitchenAbvGr','KitchenQual','BsmtQual','BsmtCond','OverallQual', 'OverallCond','BsmtFinSF2','BsmtFinSF1','BsmtUnfSF'], 
	axis =1, inplace = True)
house.drop(['FireplaceQu','Fireplaces'], axis = 1, inplace = True)
house.to_csv('Ames_test_Ihor.csv')

