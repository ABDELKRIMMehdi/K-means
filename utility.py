import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from matplotlib.patches import Rectangle

import numpy as np

def plot_dendrogram(model, **kwargs):
    distance = np.arange(model.children_.shape[0])
    position = np.arange(2, model.children_.shape[0]+2)

    linkage_matrix = np.column_stack([model.children_, distance, position]).astype(float)

    fig, ax = plt.subplots(figsize=(15, 7))

    dendrogram(linkage_matrix, orientation='top', **kwargs)

    plt.tick_params(axis='x', bottom='off', top='off', labelbottom='off')
    plt.tight_layout()
    plt.show()

def plot_clusters(X, Y, title = " ", symbolsize=20):
    plt.figure(figsize=(8,6))
    plt.scatter(X[:,0], X[:, 1], c=Y,  s=symbolsize, cmap=plt.cm.Set1)
    plt.title(title)
    plt.colorbar(ticks=np.unique(Y))
    
    
def plot_rectangles_colors(centers):
    fig, ax = plt.subplots(1)
    stride = 1/centers.shape[1];
    width = stride ; height = 1; (x0, y0)=(-width, 0)
    for i in range(centers.shape[1]):
        x0 += width
        rect = Rectangle((x0, y0), width, height, color = centers[i]/255)
        ax.add_patch(rect)
        plt.text(x0+stride/2, 0.5, "Cluster {}".format(i+1), ha="center", family='sans-serif', size=14)