import pandas as pd
import matplotlib.pylot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm



data=pd.read_csv () #read data set, in the data set, column is (R&D spend-
#money,,,Administrator--amount,,marketing spend-amount,,,state-name,,,profit-amount
data.head(10) # to check the data to select 10 sample

real_x = data.iloc[:,0:4].values #collecting all the rows and column
real_y= data.iloc[:,4].vales #collecting all rows but last column

real_x #to check the all the independent value
real_y # to check profit values only

le = LabelEncoder()
real_x[:,3] = le.fit_transform(real_x[:,3]) # we have done label encode
real_x[:,3] #we got 0,1 and2 form data, but we need data in bonary format
oneHE = OneHotEncoder(categorical_features=[3])
real_x = oneHE.fit_transform(real_x).toarray()
real_x  #now we got the value in 0 and 1 form

real_x =real_x[:,1:]  #to make trummy variable, we need to skip first column

# Now we will split the data to train and test the data for prediction

training_x,test_x,training_y,test_y = train_test_split(real_x,real_y,test_size=0.2 '''indicating for 20% test data, rest will go into the training data''', random_state = 0 ''' Taking 0 difference between our prediction and model predication''')

# now we will do Regression to train the data
MLR = LinearRegression()
MLR.fit(training_x,training_y)  #training data

#now we will all prediction 
pred_y = MLR.predict(test_x)
pred_y  #PREDICTED VALUE
 # now you can compare the predicted value with the exact value
test_y  #real value
MLR.coef_  #to calculate the cofficient value
MLR.intercept_  # to find the intercept value

#now to calculate the value through formula
#y = b0+ b1x1+b2x2......+BnXn   (We have to get b=0, to amke all the zero)

real_x = np.append(arr=np.ones((50,1)).astypes(int),values=real_x,axis=1) # now we got all the values in 1

x_opt= real[:,[0,1,2,3,4,5]]
reg_OLS=sm.OLS(endog=real_y, exog=x_opt).fit()
reg_OLS.summary()

#if the p value would greate then 0.5 then remove that row index value
x_opt= real[:,[0,1,2,3,4,5]] #we will  remove 2 from list, thier value is more than 0.5 p value.
reg_OLS=sm.OLS(endog=real_y, exog=x_opt).fit()
reg_OLS.summary()

