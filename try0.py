# sklearn库 决策树分类器 wine数据集
from sklearn.datasets import load_wine  # 引入数据集,sklearn包含众多数据集
from sklearn.model_selection import train_test_split   # 将数据分为测试集和训练集
from sklearn import tree   # 利用邻近点方式训练数据\
import matplotlib.pyplot as plt
# 引入数据
wine=load_wine()   # 引入wine数据
X_train,X_test,y_train,y_test=train_test_split(wine.data,wine.target,test_size=0.3)   # 利用train_test_split进行将训练集和测试集进行分开，test_size占30%
# 训练数据
clf=tree.DecisionTreeClassifier(random_state=0)   # 引入训练方法
clf.fit(X_train,y_train)   # 进行填充测试数据进行训练
# 预测数据
print(clf.predict(X_test))
result=clf.score(X_test,y_test)
print('score:',result)
plt.figure(figsize=(15,9))
tree.plot_tree(clf
               ,filled=True
               ,feature_names=wine.feature_names
               ,class_names=wine.target_names
               )



# sklearn库 泰坦尼克号生存者预测
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
data=pd.read_csv("E:/data/titannic_data.csv")



