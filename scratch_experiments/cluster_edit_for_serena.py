from sklearn.cluster import KMeans, OPTICS, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.cluster import hierarchy
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import cv2
from skimage import color

# convenience function that wraps several sklearn clusters algorithms and returns labels
# kmeans, gaussian_mixture, agglom, dbscan
def _cluster(values, algo="kmeans", n=3,
             img_colorspace="rgb",
             params_dict={'eps':0.1,'min_samples':24}, show=True):

    # create cluster model
    match algo:
        case "kmeans":
            model = KMeans(n_clusters=n)
        case "gaussian_mixture":
            model = GaussianMixture(n_components=n,covariance_type='diag',reg_covar=1)#,reg_covar=2)
        case "agglom":
            if show:
                clusters = hierarchy.linkage(values, method="ward")
                plt.figure(figsize=(8, 6))
                dendrogram = hierarchy.dendrogram(clusters)
                # Plotting a horizontal line based on the first biggest distance between clusters
                plt.axhline(150, color='red', linestyle='--');
                # Plotting a horizontal line based on the second biggest distance between clusters
                plt.axhline(100, color='crimson');
            model = AgglomerativeClustering(n_clusters=n)
        case "dbscan":
            model = DBSCAN(eps=params_dict['eps'], min_samples=params_dict['min_samples'])#, #algorithm='ball_tree')  # , metric='manhattan')

    labels = model.fit_predict(values)
    print(labels)
    if show:
        # Define distinct markers for each cluster
        markers = ['o', 's', 'D']  # 'o' is circle, 's' is square, 'D' is diamond, add more if needed

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot each cluster using its distinct marker
        for idx, marker in enumerate(markers):
            subset = values[labels == idx]
            colors = subset/255
            if img_colorspace is "hsv":
                colors = color.hsv2rgb(colors)
            ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], c=colors, marker=marker, label=f'Cluster {idx}')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.legend()
        plt.show()
    return labels

img = cv2.imread('D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder/segments/INAT-78524810-1_segment.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
pixels = img.reshape(-1, 3)
_cluster(pixels,algo="kmeans",n=3,img_colorspace="hsv",show=True)