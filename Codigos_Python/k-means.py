# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:48:24 2018

@author: f004197
"""

from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets


train, test = train_test_split(c, train_size = 0.8)

kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10) 
kmeans.fit(data)



for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
    data["clusters"] = kmeans.labels_



from scipy import cluster

Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(c.iloc[:,1:]).score(c['fraude']) for i in range(len(kmeans))]
score

plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

