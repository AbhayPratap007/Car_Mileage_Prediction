# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 16:28:49 2021

@author: Singh
"""
#-----------------------Car Mileage Prediction--------------------

#import libraries
import pandas as pd
import numpy as nm

#import dataset
dataset = pd.read_csv('E:/Data Analytics/Project/Car_MPG_Prediction/T6_Luxury_Cars.csv')

#see datatype of each var
dataset.info()
#check missing values
dataset.isnull().sum()

#hot-one-encoding, using dummies variable to handale catgoriacal variables
M_Dummy = pd.get_dummies(dataset["Make"],drop_first=True)

T_Dummy = pd.get_dummies(dataset["Type"],drop_first=True)

O_Dummy = pd.get_dummies(dataset["Origin"],drop_first=True)

D_Dummy = pd.get_dummies(dataset["DriveTrain"],drop_first=True)

#concantinate those dummies var into maindataset
dataset = pd.concat([dataset,M_Dummy,T_Dummy,O_Dummy,D_Dummy],axis=1)

#dropping the columns whose dummy var has been created
dataset.drop(['Make','Type','Origin','DriveTrain'],axis=1,inplace=True)
dataset.drop(["Model"],axis=1,inplace=True)

#obtaining dev. and idv. from the dataset
X = dataset.drop('MPG (Mileage)',axis=1)
y = dataset['MPG (Mileage)']

#Spliting the dataset into Training set and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

#Fitting Linear Regressin to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predict train set result
y_predict = regressor.predict(X_test)

#calcualting r square value for accesing accuracy for mulitiple regression model
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)
#Accuracy - 83.73%



#----------------------------Backward elimination(It used to remove those value of dev which has not significant level on dev)-----------------------

#import libraries
import statsmodels.api as sm

dataset_1 = dataset
dataset_1.head(5)

x1=dataset_1.drop("MPG (Mileage)",axis=1)
y1=dataset_1["MPG (Mileage)"]

x1 = nm.append(arr = nm.ones((426,1)).astype(int), values=x1, axis=1)

#create a new feature vector x_opt,which will only contain a set of 
#independent features that are significantly affecting the dependent variable.
x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,
              28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

#remove x20, because it has high significant value in entire dataset

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,
              28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()
#remove x13:

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,21,22,23,24,25,26,27,
              28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,14,16,17,18,19,21,22,23,24,25,26,27,
              28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,14,16,17,18,19,21,22,23,24,25,26,27,
              28,29,30,31,32,33,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,14,16,17,18,19,21,22,23,24,25,26,27,
              28,29,30,31,32,33,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,14,16,17,18,19,21,22,23,24,25,26,27,
              28,29,30,31,33,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,14,16,17,18,19,21,22,23,24,25,26,27,
              28,29,30,31,33,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,14,16,17,19,21,22,23,24,25,26,27,
              28,29,30,31,33,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,16,17,19,21,22,23,24,25,26,27,
              28,29,30,31,33,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,16,19,21,22,23,24,25,26,27,
              28,29,30,31,33,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,16,19,21,22,23,24,25,26,27,
              28,29,30,31,33,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,8,9,10,11,12,16,19,21,22,23,24,25,26,27,
              28,29,30,31,33,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,8,9,10,11,12,16,21,22,23,24,25,26,27,
              28,29,30,31,33,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,8,9,10,11,12,16,21,22,23,24,25,26,27,
              28,29,30,31,33,37,38,40,41,42,43,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,8,9,10,11,12,16,21,22,23,24,25,26,27,
              28,30,31,33,37,38,40,41,42,43,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,8,9,10,11,12,16,21,22,23,24,25,26,27,
              28,30,33,37,38,40,41,42,43,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,8,9,10,11,12,16,21,22,23,24,25,26,27,
              30,33,37,38,40,41,42,43,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,2,3,4,5,6,8,9,10,11,12,16,21,22,23,24,25,26,27,
              30,33,37,38,40,41,42,43,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,2,3,4,5,6,8,9,10,11,12,16,21,22,23,24,25,26,27,
              30,37,38,40,41,42,43,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,2,3,4,6,8,9,10,11,12,16,21,22,23,24,25,26,27,
              30,37,38,40,41,42,43,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,2,3,4,6,8,9,10,11,12,16,21,22,23,24,26,27,
              30,37,38,40,41,42,43,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,2,3,4,6,8,9,10,11,12,16,21,22,23,24,26,27,
              30,37,38,40,41,42,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,2,3,4,6,8,9,10,11,12,16,21,22,24,26,27,
              30,37,38,40,41,42,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,2,3,4,6,8,9,10,11,12,16,21,22,24,27,
              30,37,38,40,41,42,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()
#---------------------------------------
x_opt= x1[:, [0,2,3,4,6,8,9,10,11,12,16,21,22,24,27,30,37,38,41,42,44,45,46,47,48,49,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()
#---------------------------------------
x_opt= x1[:, [0,2,3,4,6,8,10,11,12,16,21,22,24,27,30,37,38,41,42,44,45,46,47,48,49,50,51]]             
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,2,3,4,6,8,10,12,16,21,22,24,27,30,37,38,41,42,44,45,46,47,48,49,50,51]]            
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,2,3,4,6,8,12,16,21,22,24,27,30,37,38,41,42,44,45,46,47,48,49,50,51]]            
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,2,3,4,6,8,12,16,21,22,27,30,37,38,41,42,44,45,46,47,48,49,50,51]]            
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,2,3,4,6,8,12,16,21,22,27,30,38,41,42,44,45,46,47,48,49,50,51]]             
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,2,3,4,6,8,12,16,21,22,27,30,38,41,42,44,45,46,47,48,50,51]]              
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,2,3,4,8,12,16,21,22,27,30,38,41,42,44,45,46,47,48,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,2,3,4,8,12,16,21,22,27,38,41,42,44,45,46,47,48,50,51]]                   
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,2,3,4,8,12,16,21,27,38,41,42,44,45,46,47,48,50,51]]                 
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,2,3,4,8,12,16,21,38,41,42,44,45,46,47,48,50,51]]              
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,2,3,4,8,16,21,38,41,42,44,45,46,47,48,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,2,3,4,8,16,21,41,42,44,45,46,47,48,50,51]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

#---------------------------Building Multiple Regression model-----------------

#data_set= pd.read_csv('E:/Data Analytics/Project/Car_MPG_Prediction/T6_Luxury_Cars.csv') 

#spiting fxn into training and testing
from sklearn.model_selection import train_test_split
x_BE_train, x_BE_test, y_BE_train, y_BE_test= train_test_split(x_opt, y1, test_size= 1/3, random_state=0)

#Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_BE_train, y_BE_train)

#Predicting the Test set result;
y_pred= regressor.predict(x_BE_test)
 
#Calculating the r squared value:
from sklearn.metrics import r2_score
r2_score(y_BE_test,y_pred)
#Accuracy - 84.95%

#Calculating the coefficients:
print(regressor.coef_)

#Calculating the intercept:
print(regressor.intercept_)

#Multiple regression eq. is: 
# Mileage = 59.03 + {(-5.17)*Cylinders + (-1.16)*Horsepower + (-2.82)*Weight (LBS)
#           + (2.05)*BMW + (1.39)*Honda +(2.71)*Jaguar + (2.69)*Toyato + (1.89)*Volkswagen  
#           + (-2.1)*SUV + (-1.7)*Sedan + (-1.83)*Sports + (-2.09)*Truck + (-1.74)*Wagon 
#           + (7.0)*USA + (1.11)*Front}
 
