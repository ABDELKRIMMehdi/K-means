import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster as skc
import scipy.io as sio

from utility import plot_clusters, plot_dendrogram

X = sio.loadmat(r"%path to%george.mat")["george"]
plt.scatter(X[:,0], X[:,1], s=2)

#k-means
K=6
Kmodel = skc.KMeans (n_clusters = K)
assigned_cluster = Kmodel.fit_predict(X) # vecteur qui contient le cluster attribué à chaque # instance de X 

plot_clusters(X, assigned_cluster, title ='Clustering par K-means', symbolsize=6)
centers = Kmodel.cluster_centers_
plt.scatter(centers[:,0], centers[:, 1], marker='s', s=100)

#clusturing hierarchique
K = 6
hmodel = skc.AgglomerativeClustering(n_clusters=K, linkage="complete", affinity="l2")
assigned_cluster = hmodel.fit_predict(X)

plot_clusters(X, assigned_cluster, title ='Clustering Hierarchique', symbolsize=6)

plot_dendrogram(hmodel)

#segmentation d'images

I = plt.imread(r"%PATH TO%seaforest_small.jpeg")
n, m, d= I. shape 
print("Dimensions de l'image sont {} lignes, {} colonnes {} canaux".format(n, m, d)) # afficher image 
plt.imshow((I)); 
plt.title("Original image")
X = np.reshape(I, (n*m, d)) 
print("Dimensions de l'image après vectorisation sont ",X.shape) 
K = 2 
kmeans_image = skc.KMeans (n_clusters = K) 
kmeans_image.fit(X) 
y=kmeans_image.predict(X)
num_pixels = X.shape[0] # nombre de pixels (nombre de ligne de X)) 
centers = kmeans_image.cluster_centers_ 
seg_image=np.empty(X.shape) 
for i in range(num_pixels): 
    seg_image[i,]=centers [y[i], :]

plt.figure(figsize=(18,10)) 
plt.subplot(121) 
plt.imshow((I)) 
plt.title("Image original") 
plt.subplot(122)
plt.imshow(np. uint8(seg_image.reshape(n, m, d)))
plt.title("Image segmentée")
plt.show()