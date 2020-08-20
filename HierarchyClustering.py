import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#to find the whole data in screnshot
data =pd.read_csv('Data is in screenshot to fit')
data.head()
 

#suppose now we want to sell new car
#now will allocate the data of weitage, now we have only input 3 and 4 column
#no need to split only will take advantages of column

X = data.iloc[:,[3,4]].values

#to do clustering, we can scipy or dendo-gram.
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method ='ward')) #will allow to link the data

#now to do matplot to plot the data
plt.title('Dendogram')
plt.xlabel('custer')
plt.ylabel('Eucliend Distance')
plt.show()

#now will make the clusters
from sklearn.cluster import AgglomerativeClustering #fucntion for hierarchy cluster
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# now to visualoze the cluster
plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1], s=100, c='red' label = 'cluster-1') #it is for the first cluster
plt.scatter(X[y_hc == 1,0], X[y_hc == 1,1], s=100, c='green' label = 'cluster-2')
plt.scatter(X[y_hc == 2,0], X[y_hc == 2,1], s=100, c='blue' label = 'cluster-3')
plt.scatter(X[y_hc == 3,0], X[y_hc == 3,1], s=100, c='orange' label = 'cluster-4')
plt.scatter(X[y_hc == 4,0], X[y_hc == 4,1], s=100, c='yellow' label = 'cluster-5')
plt.title('Cluster of Customer')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score(1-100)')
plt.legend() # we took label, that we have to call this
plt.show()
