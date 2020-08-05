# 菜菜 kmeans案例 聚类算法用于降维，keans的矢量量化应用
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle

'''
from PIL import Image
img = Image.open("C:\\Users\\Sci_Dell_Cpu\\Desktop\\aaa.webp")
china0 = np.asarray(img)
'''

china = load_sample_image('china.jpg')

n_clusters = 64
china = np.array(china,dtype=np.float64) / china.max()
w,h,d = original_shape = tuple(china.shape)
assert d == 3
image_array = np.reshape(china,(w*h,d))

image_array_sample = shuffle(image_array,random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_clusters,random_state=0).fit(image_array_sample)

labels = kmeans.predict(image_array)

image_kmeans = image_array.copy()
for i in range(w*h):
    image_kmeans[i] = kmeans.cluster_centers_[labels[i]]

import pandas as pd
pd.DataFrame(image_kmeans).drop_duplicates().shape

image_kmeans = image_kmeans.reshape(w,h,d)

centroid_random = shuffle(image_array,random_state=0)[:n_clusters]
labels_random = pairwise_distances_argmin(centroid_random,image_array,axis=0)
image_random = image_array.copy()
for i in range(w*h):
    image_random[i] = centroid_random[labels_random[i]]
image_random = image_random.reshape(w,h,d)

plt.figure(figsize=(10,10))
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(china)

plt.figure(figsize=(10,10))
plt.axis('off')
plt.title('Quantized image (64 colors,k-means)')
plt.imshow(image_kmeans)

plt.figure(figsize=(10,10))
plt.axis('off')
plt.title('Quantized image (64 colors,random)')
plt.imshow(image_random)
plt.show()

pic=plt.subplot(2,3,1)
pic.imshow(china)
pic=plt.subplot(2,2,2)
pic.imshow(image_kmeans)
pic=plt.subplot(2,2,3)
pic.imshow(image_random)