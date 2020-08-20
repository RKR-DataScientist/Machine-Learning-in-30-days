import pandas as pd
import matplotlib.pyplot asplt

data = pd.read_csv('see the screenshot')

#Splitting the data of independent and dependant variable
data_x = data.iloc[:,[2,3]].values  #independent
data_y = data.iloc[:,4].values  #dependant

#now splitting data in training and testing
from sklear.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.25, random_state=0)

#now will perfrom feature scaling data, to get data in +2 to -2

from sklearn.preprocessing import StandardScaler

sc= StandardScaler()
X_train = sc.fit_transform(X_train)
x_test = sc.fit_transform(X_test)

#now will do Naive Bayes Algorithm
from slearn.naive_bayes import GaussianNB

cls = GaussianNB()
cls.fit(X_train, Y_train)  #training the data

y_pred = cls.predict(X_test) #now predicting the value, while passing input variable

y_pred #checking the data

y_test # copmaring the data with real value

# now will use confusion matrix to check the count of correction and wrong
from sklearn.matrics import confusion_matrix

cm = confusion_matrix (y_test, y_pred )

#we can check the accuracy
from sklearn.metrics import accuracy_score

ac = accuracy_score(y_test, y_pred)
ac # to check the value


#To plot the value in Matplot

from matpotlib.colors import ListedColormap

x_set,y_set = training_x,training_y
x1,x2 = np.meshgrid(np.arange(start= x_set[:,0].min()-1,stop= x_set[:,0].min()+1,step=0.01)
                    np.arange(start= x_set[:,1].min()-1,stop= x_set[:,1].min()+1,step=0.01)
                    plt.counterf(x1,x2, cls_svc.predict(np.array[x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                    alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i, j in enumerate (np.unique(y_set)):
                plt.scatter(x_set[y_set==j,0], x_set[y_set==j,1],
                            c = ListedColormap(('red','green'))(i),label = j)
plt.title('Naive Bayes (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# to check the value of testing data
x_set,y_set = testing_x,testing_y
x1,x2 = np.meshgrid(np.arange(start= x_set[:,0].min()-1,stop= x_set[:,0].min()+1,step=0.01)
                    np.arange(start= x_set[:,1].min()-1,stop= x_set[:,1].min()+1,step=0.01)
                    plt.counterf(x1,x2, cls_svc.predict(np.array[x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                    alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i, j in enumerate (np.unique(y_set)):
                plt.scatter(x_set[y_set==j,0], x_set[y_set==j,1],
                            c = ListedColormap(('red','green'))(i),label = j)
plt.title('Naive Bayes (Testing Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

