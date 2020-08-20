#https://www.youtube.com/watch?v=q1S-5Z26pXY&list=PLkPmSWtWNIyQtpYf0Iq-myisH__8gRy4k&index=108
import pandas as pd
import pandas as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbours import KNeighbourClassifier
from sklearn.matrics import confusion_matrix
from matpotlib.colors import ListedColormap
#data set data od person=(Userid, gender, age, Estimatedsalery, purchased)
data = pd.read_csv()
data.head(10)

#split the data in infrom of input and output

real_x= data.iloc[:,[2,3]].values #indepedant variable
real_y= data.iloc[:,4].values #depedant variable

#Split the data in training and testing dataset

training_x,test_x,training_y,test_y = train_test_split(real_x,real_y,test_size=0.25, random_state=0)
training_x #to check the value
test_x #to check the value

#now we will use features scaling, to minise or otimise the age and salry data

s_c= StandardScale()
traning_x = s_c.fit_transform(traninig_x)
test_x= s_c.fit_transform(test_x)
training_x #to check the value
test_x #to check the value

#now we will perfrom the KNN while importing classifier
cls=KNeighboursClassifier(n_neighbors=5, metric='minknowski', p=2)
cls.fit(traning_x,traning_y) # traning the data set

# to we will predict the value
y_pred=cls.predict(test_x)
y_pred

#now to compare with the actual value
test_y

# now to check the number of right predictiona and wrong prediction, we will use confusion matrics
c_m=confusion_matrix(test_y,y_pred) #it fuction require actual variable and predict vaiable
c_m # to see the detail

#now to see the training plot data
x_set,y_set = traning_x,traning_y
x1,x2 = np.meshgrid(np.arange(start= x_set[:,0].min()-1,stop= x_set[:,0].min()+1,step=0.01)
                    np.arange(start= x_set[:,1].min()-1,stop= x_set[:,1].min()+1,step=0.01)
                    plt.counterf(x1,x2, cls.predict(np.array[x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                    alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i, j in enumerate (np.unique(y_set)):
                plt.scatter(x_set[y_set==j,0], x_set[y_set==j,1],
                            c = ListedColormap(('red','green'))(i),label = j)
plt.title('K-NN (Traningin Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#now to check the testing data set, you check while un-commenting the data

'''
x_set,y_set = test_x,test_y
x1,x2 = np.meshgrid(np.arange(start= x_set[:,0].min()-1,stop= x_set[:,0].min()+1,step=0.01)
                    np.arange(start= x_set[:,1].min()-1,stop= x_set[:,1].min()+1,step=0.01)
                    plt.counterf(x1,x2, cls.predict(np.array[x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                    alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i, j in enumerate (np.unique(y_set)):
                plt.scatter(x_set[y_set==j,0], x_set[y_set==j,1],
                            c = ListedColormap(('red','green'))(i),label = j)
plt.title('K-NN (Testing Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

'''

