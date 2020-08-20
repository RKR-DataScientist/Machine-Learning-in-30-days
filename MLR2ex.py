import pandas as pd
import matplotlib.pylot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


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

