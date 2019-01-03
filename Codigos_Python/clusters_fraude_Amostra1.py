# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:24:52 2018

@author: f004197
"""

train, test = train_test_split(df_processed, train_size = 0.8)

train_x = train.drop(['Fraude_Final','clusters','cluster_center'], axis=1)
train_y = train['Fraude_Final']

test_x = test.drop(['Fraude_Final','clusters','cluster_center'], axis=1)
test_y = test['Fraude_Final']

################################## ELBOW #####################################
kmeans = KMeans(n_clusters=num_clusters, max_iter=1000).fit(df_cluster)

K = range(1,21)
KM = [KMeans(n_clusters=k).fit(np.array(train_x)) for k in K]
centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(train_x, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/train_x.shape[0] for d in dist]



# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(train_x)**2)/train_x.shape[0]
bss = tss-wcss

fig = plt.figure()
ax = fig.add_subplot(111)
axes = plt.gca()
axes.set_xlim([0,20])
axes.set_ylim([0,100])
ax.plot(K, bss/tss*100, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')
plt.show()


## Fraude ou Não Fraude: 2 clusters

#train 

num_clusters = 2

kmeans = KMeans(n_clusters=num_clusters, max_iter=1000).fit(train_x)






series = train_x['clusters'].value_counts()
colors = cm.viridis(np.linspace(0, 1, 9))
series.plot(kind='bar', title='Numero de Cluster', color = colors)


correct = 0
for i in range(len(train_x)):
    predict_me = np.array(np.array(train_x)[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == np.array(train_y)[i]:
        correct += 1

print(correct/len(train_x))


#test
correct = 0
for i in range(len(test_x)):
    predict_me = np.array(np.array(test_x)[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == np.array(test_y)[i]:
        correct += 1

print(correct/len(test_x))



train_x["clusters"] = kmeans.labels_
centers = [kmeans.cluster_centers_]


test_x["clusters"] = kmeans.predict(test_x)

test_x["clusters"].hist()

test_x["clusters"] = kmeans.labels_
centers = [kmeans.cluster_centers_]

test_y.where(test_x["clusters"].values==test_y)

test_y.hist()

test_cluster = pd.concat([test_x,test_y], axis=1 )

corr = test_cluster.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()


#########################################################



################################################### Para representar clusters

from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit(df_processed)
ver_pca = pca.components_


pca_d = pd.DataFrame(pca.transform(df_processed), columns=['PCA%i' % i for i in range(2)], index=df_processed.index)
scat= plt.scatter(pca_d['PCA0'],pca_d['PCA1'], c =kmeans.labels_ )
bounds = np.linspace(0,9,9+1)
cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
cb.set_label('Custom cbar')
plt.show()

############################################################################

    
#REPRESENTAçÃO GRÀFICA 3 PCA do SET de Dados com base nos clusters identifcados

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


###animação
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro', animated=True)

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,
###############
##TESTE COM ANIMAçÂO

pca = PCA(n_components=3).fit(df_processed)
ver_pca = pca.components_

fig = plt.figure()
ax = Axes3D(fig)
pca_d = pd.DataFrame(pca.transform(df_processed), columns=['PCA%i' % i for i in range(3)], index=df_processed.index)
ax.scatter(pca_d['PCA0'],pca_d['PCA1'],pca_d['PCA2'], c =kmeans.labels_ )
bounds = np.linspace(-10,100,110+1)
cb = plt.colorbar(scat, spacing='proportional',ticks=bounds,shrink=0.8 )
cb.set_label('Clusters') 
ani = FuncAnimation(fig, update(pca_d[['PCA0','PCA1']]), frames=100,init_func=init, blit=True)
plt.show()



import seaborn as sns
corr = df_processed.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()


#########plot fraud and clusters


ax = sns.countplot(x="clusters", hue="Fraude_Final", data=df_processed[['clusters', 'Fraude_Final']], palette="Set2")



##########

df_processed['cluster_center'] = [centroids[i] for i in kmeans.labels_]

df_processed.loc[df_processed['cluster_center']== centroids]


########
cluster0_Analysis = df_processed.loc[df_processed['clusters']==0]
cluster0_Analysis.to_csv()

sns.heatmap(cluster0_Analysis.corr(), 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()

##CENTROIDES DOS CLUSTERS


centroids  = kmeans.cluster_centers_ 





