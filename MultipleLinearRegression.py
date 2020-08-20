import pandas as pd
#going to work on Multiple Linear Regression
data = pd.read_csv()
data.head()

data_x = data.iloc[:,0:,-1].values
data_y = data.iloc[:,-1].values

#now we will split the data
 from sklearn.model_selections import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=42)

 x_train, x_test, y_train, y_test = train_test_split(data_x,data_y,test_size=0.33, random_state=42)

 #now to train the data after splitting

 from sklearn.linear_model import LinearRegression

 reg= LinearRegression()
 reg.fit(x_train,y_train)

 #now to predit the values
 y_pred= reg.predict( x_test)
 y_pred

 #compare with the actual data
 y_test

 #to find the cofficient value, to find predict on the computational method
 print(reg.coef_) # it is the value of b1,b2...so on, or we can say M value
 print(reg.intercept_) #it is the value of C.

 # Y= b0 + b1*x1 + b2*x2 + b3*x3 + b4*x4.....Bn*Xn

 y0 = reg.intercept_ + reg.coef_[0]*x_test[0][0] + reg.coef_[1]*x_test[0][1] + reg.coef_[2]*x_test[0][3]+ reg.coef_[4]*x_test[0][4]

 y0 # to display the predicted value of index 0, on the beses of computational method


 #now to find the score of regression
 print('Trainging Score : ', reg.score(x_train, y_train) * 100)
 print('Testing Score : ', reg.score(x_test, y_test) * 100)
  
 
