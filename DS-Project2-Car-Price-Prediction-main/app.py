#Car Price Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

#Data Acquisition

df=pd.read_csv(r"C:\Users\ComputerCenter3\Desktop\cpp\audi.csv")
print(df)

#Data Exploration

import pandas_profiling as pf
print(pf.ProfileReport(df))
print(len(df))
df.shape
df.dtypes
df.isna().sum()
df.info()
df.describe()

X = df.iloc[:,[0,1,3,4,5,6,7,8]].values
print(X.shape)
print(X)

Y = df.iloc[:,[2]].values
print(Y.shape)
print(Y)
print(pd.DataFrame(X).head(5))

#Data Pre-processing

from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
X[:,0] = le1.fit_transform(X[:,0])
le2 = LabelEncoder()
X[:,-4] = le2.fit_transform(X[:,-4])
print(X)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[2])],remainder='passthrough')
X = ct.fit_transform(X)
print(X.shape)
print(pd.DataFrame(X))
print(pd.DataFrame(X))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(pd.DataFrame(X))

from sklearn.model_selection import train_test_split
(X_train,X_test,Y_train,Y_test) = train_test_split(X,Y,test_size=0.2,random_state=0)
print(X.shape, Y.shape)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor(random_state=0)
regression.fit(X_train,Y_train)
print(regression)


y_pred = regression.predict(X_test)
print(y_pred)
print(np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1))

from sklearn.metrics import r2_score,mean_absolute_error
print('R2 Score ', r2_score(Y_test, y_pred))
print('Mean Absolute Error', mean_absolute_error(Y_test,y_pred))

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)
print(reg)

y_pred = reg.predict(X_test)
print(y_pred)
print(np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1))

from sklearn.metrics import r2_score,mean_absolute_error
print('R2 Score ', r2_score(Y_test, y_pred))
print('Mean Absolute Error', mean_absolute_error(Y_test,y_pred))

y_pred = reg.predict(X)
print(y_pred)
result = pd.concat([df,pd.DataFrame(y_pred)],axis=1)
print( result)

from sklearn.ensemble import  ExtraTreesRegressor
ET_Model=ExtraTreesRegressor(n_estimators = 120)
ET_Model.fit(X_train,Y_train)
y_predict=ET_Model.predict(X_test)
from sklearn.metrics import r2_score,mean_absolute_error
print('R2 Score ', r2_score(Y_test, y_predict))
print('Mean Absolute Error', mean_absolute_error(Y_test,y_predict))

y_pred = reg.predict(X)
print(y_pred)
result = pd.concat([df,pd.DataFrame(y_pred)],axis=1)
print( result)

from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 80, stop = 1500, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(6, 45, num = 5)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

# create random grid
rand_grid={'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
rf=RandomForestRegressor()
rCV=RandomizedSearchCV(estimator=rf,param_distributions=rand_grid,scoring='neg_mean_squared_error',n_iter=3,cv=3,random_state=42, n_jobs = 1)

print(rCV.fit(X_train,Y_train))

rf_pred=rCV.predict(X_test)
print(rf_pred)

from sklearn.metrics import mean_absolute_error,mean_squared_error
print('MAE',mean_absolute_error(Y_test,rf_pred))
print('MSE',mean_squared_error(Y_test,rf_pred))

print(r2_score(Y_test,rf_pred))

#Model Training
from catboost import CatBoostRegressor
cat=CatBoostRegressor()
print(cat.fit(X_train,Y_train))

cat_pred=cat.predict(X_test)
print(cat_pred)
print(r2_score(Y_test,cat_pred))

#Model Deployment
import pickle 
# Saving model to disk
pickle.dump(cat, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(model.predict (X_train))