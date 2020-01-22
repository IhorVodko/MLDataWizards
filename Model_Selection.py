#!/usr/bin/env python
# coding: utf-8

#import 
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
pd.set_option('display.max_columns', 100)

#download prepocessed train data
df_original = pd.read_csv('Final_Train.csv')
df = df_original.copy()
print(df.shape)
df.dropna(inplace=True)
print(df.shape)
df.head()

#download preprocessed test data
df_originalt = pd.read_csv('Final_Test.csv')
dft = df_originalt.copy()
print(dft.shape)
dft.dropna(inplace=True)
print(dft.shape)
dft.head()

#dummify variables
df['label'] = 1
dft['label'] = 2
df_con = pd.concat([df, dft])
df_con['SalePrice'].fillna(100, inplace=True)
features_df_con = pd.get_dummies(df_con, prefix_sep='_', drop_first=True)
df = features_df_con[features_df_con['label'] == 1]
dft = features_df_con[features_df_con['label'] == 2]
df = df.drop(['label'], axis=1)
dft = dft.drop(['label', 'SalePrice'], axis=1)
df.shape
dft.shape

#create x_traain and y_train
x = df.drop(['SalePrice'], axis=1)
y = df['SalePrice']

#examine default Gradient Boosting model
gbm_c = GradientBoostingRegressor()
gbm_c.fit(x, y)
gbm_c.score(x, y)

#tuning of hype_rparameters
one_model = GradientBoostingRegressor()
grid_para_boost = {
    'learning_rate' : [0.01, 0.2, 0.5, 0.8, 1],
    'max_features': [2, 3, 5, 10, 20, 30, 59],
    "n_estimators": [2000, 3000],
    'max_depth' : [2, 3, 5, 8],
    'subsample' : [0.1, 0.3, 0.5, 0.8, 1],
    "random_state": [42]}
grid_search_boost_one = GridSearchCV(one_model, grid_para_boost, cv=3,n_jobs=-1)
get_ipython().run_line_magic('time', 'grid_search_boost_one.fit(x, y)')

grid_search_boost_one.best_params_
grid_search_boost_one.best_score_

#root of mean squared error calculation for the train data
df['PricePred'] = grid_search_boost_one.predict(x)
rmse = np.sqrt(( (np.log(df['PricePred']) - np.log(df['SalePrice']) ) ** 2).mean())

#predicting y_test based on the x_test using tuned model
xt=dft
test_price = list(grid_search_boost_one.predict(xt))
Data = {'Id':range(1461, 2920), 'SalePrice':test_price}
Sub = pd.DataFrame(Data).set_index('Id')
print(Sub.head(3))
Sub.tail(3)

#write submissin file
Sub.to_csv('Submission_gb.csv')

#save tuned GradientBoosting model
with open('my_dumped_regressor_gb.pkl', 'wb') as fid:
    pickle.dump(grid_search_boost_one, fid)   
# # load it again
# with open('my_dumped_regressor_gb.pkl', 'rb') as fid:
#     gnr_loaded = pickle.load(fid)

#feature importance for the GB model
importance = sorted(list(zip(x.columns, list(grid_search_boost_one.best_estimator_.feature_importances_))),
                    key=lambda t:t[1], reverse=True)[:15]

plt.figure(figsize = (12,6))
a, b = zip(*importance)
importance_score = pd.DataFrame({'feature':a, 'score':b})
plt.bar(importance_score['feature'], importance_score['score'])
plt.xticks(rotation = 'vertical')
plt.ylabel('score')
plt.xlabel('feature')


#check Support Vector Regressor model with default parameters
svr = SVR(epsilon = 1e-4, gamma=1)
svr.fit(x,y)
svr.score(x,y)

#hyper parameters tuning 
svr=SVR()
grid_para_svr = {
    'kernel': ['rbf', 'linear'],
    'gamma':np.logspace(3, -3, 3),
    'epsilon':np.logspace(3,-3,3),
    'C':[1e-4, 1, 1e4]
}
grid_search_svr_ = GridSearchCV(svr, grid_para_svr, cv=3,n_jobs=-1)

grid_search_svr_.best_params_
grid_search_svr_.best_score_

#root of mean squared error calculation for the train data 
df['PricePred'] = grid_search_svr_.predict(x)
rmse = np.sqrt(( (np.log(df['PricePred']) - np.log(df['SalePrice']) ) ** 2).mean())
rmse

#predicting y_test based on the x_test using tuned model svr
xt=dft
test_price = list(grid_search_boost_one.predict(xt))
Data = {'Id':range(1461, 2920), 'SalePrice':test_price}
Sub = pd.DataFrame(Data).set_index('Id')
print(Sub.head(3))
Sub.tail(3)

#write submission file
Sub.to_csv('Submission_svr.csv')

# save the model
with open('my_dumped_regressor_svr.pkl', 'wb') as fid:
    pickle.dump(grid_search_svr_, fid)
# # load it again
# with open('my_dumped_regressor_scr.pkl', 'rb') as fid:
#     gnr_loaded = pickle.load(fid)