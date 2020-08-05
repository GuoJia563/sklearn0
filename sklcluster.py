# 菜菜sklearn教程 b站视频 cluster kmeans

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X,y = make_blobs(n_samples=500,n_features=2,centers=4,
                 random_state=3)
fig,ax1 = plt.subplots(1)
ax1.scatter(X[:,0],X[:,1]
            ,marker='o'
            ,s=8)

color = ["red","pink","orange","gray","green"]
fig,ax1 = plt.subplots(1)
for i in range(4):
    ax1.scatter(X[y==i,0],X[y==i,1]
                ,marker='o'
                ,s=8
                ,c=color[i]
                )
plt.show()

from sklearn.cluster import KMeans
n_clusters = 5
cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(X)
y_pred = cluster.labels_
centroid = cluster.cluster_centers_
inertia = cluster.inertia_

fig,ax1 = plt.subplots(1)
for i in range(n_clusters):
    ax1.scatter(X[y_pred==i,0],X[y_pred==i,1]
                ,marker='o'
                ,s=8
                ,c=color[i]
                )
ax1.scatter(centroid[:,0],centroid[:,1]
            ,marker="x"
            ,s=15
            ,c="black"
            )
plt.show()

from sklearn.metrics import silhouette_score
from sklearn.metrics import  silhouette_samples
silhouette_score(X,y_pred)  #轮廓系数

from sklearn.metrics import calinski_harabasz_score
calinski_harabasz_score(X,y_pred)



#案例：基于轮廓系数来选择n_clusters
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

for n_clusters in [2,3,4,5,6,7]:
    n_clusters = n_clusters
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, X.shape[0] + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(X)
    cluster_labels = clusterer.labels_
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i)/n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper)
                          ,ith_cluster_silhouette_values
                          ,facecolor=color
                          ,alpha=0.7
                          )
        ax1.text(-0.05
                 , y_lower + 0.5 * size_cluster_i
                 , str(i))
        y_lower = y_upper + 1
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1]
                    ,marker='o'
                    ,s=8
                    ,c=colors
                    )
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='x',
                    c="red", alpha=1, s=200)
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data"
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
        plt.show()


#重要参数init & random_state & n_init :初始质心怎么放好？
plus = KMeans(n_clusters= 4).fit(X)

clu_cens=np.array([[0,1],[2,1],[3,4],[-2,-2]])
clu = KMeans(n_clusters=4,init=clu_cens,n_init=1).fit(X)

random = KMeans(n_clusters= 10,init="random",random_state=420).fit(X)

#重要参数max_iter & tol : 让迭代停下来
random1 = KMeans(n_clusters=10,init="random",max_iter=6,random_state=420).fit(X)
y_pred_max10 = random1.labels_
a=silhouette_score(X,y_pred_max10)

random2 = KMeans(n_clusters= 10,init="random",max_iter=20,random_state=420).fit(X)
y_pred_max20 = random2.labels_
b=silhouette_score(X,y_pred_max20)

#函数clusters.k_means
from sklearn.cluster import k_means
res=k_means(X,4,return_n_iter=True)
a=silhouette_score(X,res[1])