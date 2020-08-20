import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# data having detail person = position, leverl and salary
data = pd.read_csv()
real_x = data.iloc[:,1:2] # indepedent variablw
real_y = data.iloc[:,2] # dependent variable

#now to call regression
reg=DecisionTreeRegressor(random_state=0)
#now to train the model
reg.fit(real_x,real_y)

#now to predict the value while calling regressor
y_predict= reg.predict(6) # prediction of the six year
y_pred #to check the prediction

#now to plot the value in graph
x_grid= np.arange(min(real_x),max(real_x),0.01)
x_grid= x_grid.reshape(len(x_grid),1)
plt.scatter(real_x,real_y,color = 'green')
plt.plot(x_grid, reg.predict(x_grid))
plt.title('Decision Tree Regressor')
plt.xlabel('Pos Level')
plt.ylable('salary')
plt.show


