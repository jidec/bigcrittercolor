from sklearn.cluster import KMeans, OPTICS, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# convenience function that wraps several sklearn clusters algorithms and returns labels
# kmeans, gaussian_mixture, agglom, dbscan
def _cluster(values, algo="kmeans", n=3,
             eps = 0.1,
             min_samples = 24,
             show=True,
             outlier_percentile=None, return_fuzzy_probs=False,
             return_values_as_centroids=False):
    if show:
        #if values.shape[1] > 3:
        scaled = StandardScaler().fit_transform(values)

        ## TSNE
        tsne = TSNE(n_components=2)
        transformed_data = tsne.fit_transform(scaled)

        # Visualize the results
        plt.scatter(transformed_data[:, 0], transformed_data[:, 1], edgecolors='k', cmap='viridis')
        plt.title('t-SNE of Dataset')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar()
        plt.grid(True)
        plt.show()

        ## PCA
        # Step 2: Apply PCA and reduce to 2 principal components
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled)

        # Step 3: Visualize the results
        plt.figure(figsize=(10, 6))
        plt.scatter(principal_components[:, 0], principal_components[:, 1], edgecolors='k', cmap="viridis")
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('2D PCA of Multi-dimensional Data')
        plt.grid(True)
        plt.show()

        for i, ev in enumerate(pca.explained_variance_ratio_, start=1):
            print(f"Principal Component {i}: {ev * 100:.2f}% of the variance")

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
                plt.axhline(150, color='red', linestyle='--')
                # Plotting a horizontal line based on the second biggest distance between clusters
                plt.axhline(100, color='crimson')
                plt.show()

            model = AgglomerativeClustering(n_clusters=n)
        case "dbscan":
            model = DBSCAN(eps=eps, min_samples=min_samples)#, #algorithm='ball_tree')  # , metric='manhattan')

    if return_fuzzy_probs:
        # Get the probabilities of belonging to each cluster
        model = model.fit(values)
        probs = model.predict_proba(values)

        print(model.converged_)
        print("Probability of data belonging to each cluster:\n", list(probs))

        if show:
            labels = model.predict(values)
            # Visualize the clustering results
            plt.scatter(values[:, 0], values[:, 1], c=labels, cmap='viridis')
            plt.title('GMM Clustering')
            plt.show()

        return probs

    if outlier_percentile is not None:
        model.fit(values)

        # Compute the distances of each point to its assigned cluster centroid
        distances = np.min(distance.cdist(values, model.cluster_centers_, 'euclidean'), axis=1)

        # Find the 95th percentile threshold
        threshold = np.percentile(distances, outlier_percentile)

        # Label points above the threshold (i.e., the furthest 5%) as outliers (-1)
        labels = np.where(distances > threshold, -1, model.labels_)

        return labels

    labels = model.fit_predict(values)

    if return_values_as_centroids:
        labels = np.array(labels)  # Convert labels to a numpy array
        values = np.array(values)
        unique_labels = np.unique(labels)
        centroids = {}
        print(type(values))
        print(type(labels))
        print(values.shape)
        print(labels.shape)
        for label in unique_labels:
            centroids[label] = np.mean(values[labels == label], axis=0)

        values_centroids = np.array([centroids[label] for label in labels])

        return values_centroids

    return labels