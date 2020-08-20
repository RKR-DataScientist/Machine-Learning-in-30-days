# https://www.youtube.com/watch?v=nH5QVAW73GA&list=PLkPmSWtWNIyQtpYf0Iq-myisH__8gRy4k&index=101
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler  # for feature scaling
from skearn.linear_model import LogisticRegression
from sklear.metrics import confusion_matrix
from matpotlib.colors import ListedColormap

#data column = (userid, gender, age, estiatedsalary,purchased)
data = pd.read_csv()
data.head(10)

real_x = data.iloc[:,[2,3]].values
real_x

real_y = data.iloc[:,4].values
real_y


# now to split the dataset
traning_x, test_x,training_y,test_y = train_test_split(real_x,real_y, test_size=0.25, random_state=0)
#to check the data set
traning_x
test_x

#now we have to do feature scaling (Basically, when there would huge gap defference between independent variable, then feature scalling  will transform the value under -2 to 2.

scaler = StandardScaler()
training_x = scaler.fit_transform(traninig_x)
test_x = scaler.fit_transform(test_x)

# Now we will make classifier
classifier_LR= LogistiRegression(random_state=0)

# now to train the model
classifier_LR.fit(traning_x, training_y)

# now we will apply prediction
y_pred = classifier_LR.predict(test_x)
y_pred

#now to compare the value
test_y

# now got few wrong prediction (means un-matched data), now to get te whole wrong prediction and right pridication detail.
# we will use confusion metrix, it will separate the both wrong and right prediction

c_m = confusion_matrix(test_y, y_pred)
c_m #to check the confusion matrix


# now to plot the Training data

x_set,y_set = training_x, tranining_y
x1,x2 = np.meshgrid(np.arange(start= x_set[:,0].min()-1,stop= x_set[:,0].min()+1,step=0.01)
                    np.arange(start= x_set[:,1].min()-1,stop= x_set[:,1].min()+1,step=0.01)
plt.counterf(x1,x2, classofier_LR.predict(np.array[x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                    alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i, j in enumerate (np.unique(y_set)):
                plt.scatter(x_set[y_set==j,0], x_set[y_set==j,1],
                            c = ListedColormap(('red','green'))(i),label = j)
plt.title('Logistic Regression (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# now to plot the Testing data

x_set,y_set = test_x, test_y
x1,x2 = np.meshgrid(np.arange(start= x_set[:,0].min()-1,stop= x_set[:,0].min()+1,step=0.01)
                    np.arange(start= x_set[:,1].min()-1,stop= x_set[:,1].min()+1,step=0.01)
plt.counterf(x1,x2, classofier_LR.predict(np.array[x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                    alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i, j in enumerate (np.unique(y_set)):
                plt.scatter(x_set[y_set==j,0], x_set[y_set==j,1],
                            c = ListedColormap(('red','green'))(i),label = j)
plt.title('Logistic Regression (Testing Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


