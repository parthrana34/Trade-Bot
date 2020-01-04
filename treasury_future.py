# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 18:33:03 2019

@author: parth
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
import scipy

#no of days which are used for understanding the trend
n = 7
#no of days for that slope tendency is checked
m = 6
#no of days for that conditions are checked
k=2
#profit in index
p=0.5

data = pd.read_excel("data.xlsx", sheet_name="data_future")

data['slope_10yt'] = float('nan')
data['slope_rate'] = float('nan')
data['slope_10yt_tendency'] = float('nan')
data['slope_rate_tendency'] = float('nan')

for i in range(n,data.shape[0]+1):
    print(i)
    temp = data[i-n:i]
    data.iloc[i-1,10] = linregress(temp['x'],temp['10yt_future']).slope
    data.iloc[i-1,11] = linregress(temp['x'],temp['rate']).slope
    
    if i>m+n:
        temp1 = data[i-m:i]
        data.iloc[i-1,12] = linregress(temp1['x'],temp1['slope_10yt']).slope
        data.iloc[i-1,13] = linregress(temp1['x'],temp1['slope_rate']).slope    

data['intensity_10yt'] = data['slope_10yt'].apply(lambda x: scipy.stats.percentileofscore(abs(data['slope_10yt']), abs(x)))
data['intensity_rate'] = data['slope_rate'].apply(lambda x: scipy.stats.percentileofscore(abs(data['slope_rate']), abs(x)))

data['opportunity'] = (data['slope_10yt']*data['slope_rate'] > 0) & (data['slope_10yt_tendency']*data['slope_rate_tendency'] > 0)
data['rate_dominant'] = abs(data['slope_10yt']) < abs(data['slope_rate'])

data['long/short'] = float('nan')
for i in range(m,data.shape[0]+1):
    print(i)
    temp = data[i-k:i]
    
    if temp[['opportunity']].eq(True).all()[0]:
        
        if data.iloc[i-1,11] < 0:
            data.iloc[i-1,18] = 'long'
        else :
            data.iloc[i-1,18] = 'short'


data['profit'] = float('nan')
data['time'] = float('nan')

for i in range(0,data.shape[0]):
    
    if data.iloc[i,18] == 'long':
            
        temp2 = data[(data['10yt_future'] > data.iloc[i,2] + p ) & (data['date'] > data.iloc[i,0])]
        if temp2.shape[0] > 0:
            data.iloc[i,19] = temp2.iloc[0,2]
            data.iloc[i,20] = temp2.iloc[0,0] - data.iloc[i,0]
    
    elif data.iloc[i,18] == 'short':
        
        temp2 = data[(data['10yt_future'] < data.iloc[i,2] - p ) & (data['date'] > data.iloc[i,0])]
        if temp2.shape[0] > 0:
            data.iloc[i,19] = temp2.iloc[0,2]
            data.iloc[i,20] = temp2.iloc[0,0] - data.iloc[i,0]



#=================================================================ML===========================================================================

from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import train_test_split
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

data_model = data[['mid_mth_ret','10yt','rate','slope_10yt','slope_rate','slope_10yt_tendency','slope_rate_tendency','intensity_10yt','intensity_rate']].dropna(axis='index')

X = data_model[['10yt','rate','slope_10yt','slope_rate','slope_10yt_tendency','slope_rate_tendency','intensity_10yt','intensity_rate']]
y = data_model[['mid_mth_ret']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf.fit(np.asarray(X_train), np.asarray(y_train))  