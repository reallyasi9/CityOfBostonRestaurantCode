# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
from sklearn import linear_model

#%%
business_data = pd.read_csv("processed_data/business_data.csv", index_col="restaurant_id")
business_data.head()

#%% Create dummies for regression
business_data = pd.get_dummies(business_data)
business_data.fillna(0, inplace=True)

#%%
train_labels = pd.read_csv("data/train_labels.csv", index_col=0)
train_labels['date'] = train_labels['date'].map(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d")))
train_labels.head()

#%%
train_data = train_labels.join(business_data, on="restaurant_id", how="left")
train_outcome = train_data[['*', '**', '***']].astype(np.float64)
train_data.drop(['*', '**', '***', 'restaurant_id'], inplace=True, axis=1)
train_data.head()

#%%
regressor = linear_model.LinearRegression()
regressor.fit(train_data,
              train_outcome)
              
#%%