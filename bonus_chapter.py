#菜菜 决策树full version pdf
# Bonus Chapter I 实例：分类树在合成数集上的表现
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons,make_circles,make_classification
from sklearn.tree import DecisionTreeClassifier
X,y=make_classification(n_samples=100
                        ,n_features=2
                        ,n_redundant=0
                        ,n_informative=2
                        ,random_state=1
                        ,n_clusters_per_class=1
                        )
plt.scatter(X[:,0],X[:,1])
rng=np.random.RandomState(2)
X+=2*rng.uniform(size=X.shape)
linearly_separable=(X,y)
plt.scatter(X[:,0],X[:,1])
datasets=[make_moons(noise=0.3,random_state=0),
          make_circles(noise=0.2,factor=0.5,random_state=1),
          linearly_separable]
#画出三种数据集
figure = plt.figure(figsize=(6,9))
i=1
for ds_index,ds in enumerate(datasets):
    X,y=ds
    X=StandardScaler().fit_transform(X)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.4,random_state=42)
    x1_min,x1_max=X[:,0].min() - .5, X[:,0].max() + .5
    x2_min,x2_max=X[:,1].min() - .5, X[:,1].max() + .5
    array1,array2 = np.meshgrid(np.arange(x1_min,x1_max,0.2),
                                np.arange(x2_min,x2_max,0.2))

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000','#0000FF'])
    ax = plt.subplot(len(datasets),2,i)
    if ds_index == 0:
        ax.set_title("input data")

    ax.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=cm_bright,
               edgecolors='k')  #c:在cmap的颜色范围内，调深度
    ax.scatter(X_test[:,0],X_test[:,1],c=y_test,cmap=cm_bright,
               alpha=0.6,edgecolors='k')

    ax.set_xlim(array1.min(),array1.max())
    ax.set_ylim(array2.min(),array2.max())
    ax.set_xticks(())
    ax.set_yticks(())

    i+=1

    ax = plt.subplot(len(datasets),2,i)

    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)

    z = clf.predict_proba(np.c_[array1.ravel(), array2.ravel()])[:, 1]
    z = z.reshape(array1.shape)
    ax.contourf(array1,array2,z,cmap=cm,alpha=0.8)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    ax.scatter(X_test[:,0],X_test[:,1],c=y_test,cmap=cm_bright,
               edgecolors='k',alpha=0.6)
    ax.set_xlim(array1.min(),array1.max())
    ax.set_ylim(array2.min(),array2.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_index ==0:
        ax.set_title("Decision Tree")
    ax.text(array1.max() - .3, array2.min()
            +.3, ('{:.1f}%'.format(score*100)), size=15,
            horizontalalignment='right')
    i += 1
plt.tight_layout()
plt.show()