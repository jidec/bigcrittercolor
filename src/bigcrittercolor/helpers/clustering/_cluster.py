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
from bigcrittercolor.helpers import _scatterColors
from sklearn.preprocessing import MinMaxScaler
import cv2
import matplotlib.patches as patches

# convenience function that wraps several sklearn clusters algorithms and returns labels
# kmeans, gaussian_mixture, agglom, dbscan
def _cluster(values, algo="kmeans", n=3,
             eps = 0.1,
             min_samples = 24,
             linkage="ward",
             scale=False, # whether to scale the features between 0 and 1 to equalize the importance of each (happens before weighting if applicable)
             weights=None, # a list of weights, one for each column, that make certain features more or less important
             show_pca_tsne = False,
             show_color_scatter = False,
             input_colorspace = "rgb",
             show=True,
             outlier_percentile=None, return_fuzzy_probs=False,
             merge_with_user_input=False,
             return_values_as_centroids=False):

    if show_pca_tsne:
        scaled = StandardScaler().fit_transform(values)

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

    start_values = np.copy(values)
    if scale:
        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()
        # Scale the data
        values = scaler.fit_transform(values)

    if weights is not None:
        values = values * weights

    # create cluster model
    match algo:
        case "kmeans":
            model = KMeans(n_clusters=n)
        case "gaussian_mixture":
            model = GaussianMixture(n_components=n,covariance_type='diag',reg_covar=1)#,reg_covar=2)
        case "agglom":
            if show:
                dendrogram_sample = np.copy(values)
                if values.shape[0] >= 5000:
                    # Generate a random sample of 1000 unique indices from the range of your array's length
                    indices = np.random.choice(values.shape[0], 5000, replace=False)
                    dendrogram_sample = dendrogram_sample[indices, :]

                clusters = hierarchy.linkage(dendrogram_sample)
                plt.figure(figsize=(8, 6))
                dendrogram = hierarchy.dendrogram(clusters)
                # Plotting a horizontal line based on the first biggest distance between clusters
                plt.axhline(150, color='red', linestyle='--')
                # Plotting a horizontal line based on the second biggest distance between clusters
                plt.axhline(100, color='crimson')
                plt.show()

            model = AgglomerativeClustering(n_clusters=n,linkage=linkage)
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

    # if we scaled, revert back to the unscaled values we had before clustering - the labels stay the same
    if scale:
        values = start_values

    # if merge with user input, take input of the form 1,2;3,4;5,6 to merge clusters
    if merge_with_user_input:
        # Print centroids of each cluster
        centroids = {}

        for label in np.unique(labels):
            centroids[label] = np.mean(values[labels == label], axis=0)

        # Convert centroids to RGB if necessary
        if input_colorspace in ["cielab", "hls"]:
            for label in centroids:
                if input_colorspace == "cielab":
                    # Convert from CIELAB to RGB
                    lab = np.uint8(np.round(centroids[label])).reshape(1, 1, 3)
                    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    centroids[label] = rgb[0, 0]
                elif input_colorspace == "hls":
                    # Convert from HLS to RGB
                    hls = np.uint8(np.round(centroids[label])).reshape(1, 1, 3)
                    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
                    centroids[label] = rgb[0, 0]

        for label in np.unique(labels):
            print(f"Centroid of cluster {label}: {centroids[label]}")

        if show and input_colorspace is not None:

            # Calculate the number of values in each cluster
            cluster_counts = {label: np.sum(labels == label) for label in np.unique(labels)}

            # Print centroids and visualize as squares with cluster counts
            fig, ax = plt.subplots(figsize=(5, len(centroids)))  # Adjusted figure size for better visualization
            ax.set_xlim(0, 5)
            ax.set_ylim(0, len(centroids))

            for i, (label, centroid) in enumerate(centroids.items()):
                color = centroid / 255  # Normalize if your RGB values are in the range 0-255
                rect = patches.Rectangle((1, len(centroids) - i - 1), 1, 1, linewidth=1, edgecolor='black',
                                         facecolor=color)
                ax.add_patch(rect)
                ax.text(2, len(centroids) - i - 0.5, f"{label} (n={cluster_counts[label]})", horizontalalignment='left',
                        verticalalignment='center')

            plt.xticks([])
            plt.yticks([])
            plt.show()

            # Print centroids and visualize as squares
            #fig, ax = plt.subplots(figsize=(3, len(centroids)))
            #ax.set_xlim(0, 3)
            #ax.set_ylim(0, len(centroids))

            #for i, (label, centroid) in enumerate(centroids.items()):
            #    color = centroid / 255  # Normalize if your RGB values are in the range 0-255
            #    rect = patches.Rectangle((1, len(centroids) - i - 1), 1, 1, linewidth=1, edgecolor='black',
            #                             facecolor=color)
            #    ax.add_patch(rect)
            #    ax.text(1.5, len(centroids) - i - 0.5, str(label), horizontalalignment='center',
            #            verticalalignment='center')

            #plt.xticks([])
            #plt.yticks([])
            #plt.show()

        # Take user input for merging clusters
        user_input = input("Enter cluster sets to merge (e.g., '1,2; 3,4'): ")

        if user_input != '':
            cluster_groups = user_input.split(';')

            for group in cluster_groups:
                clusters_to_merge = [int(x.strip()) for x in group.split(',')]
                for i, label in enumerate(labels):
                    if label in clusters_to_merge:
                        labels[i] = clusters_to_merge[0]  # Assign all merged clusters to the first cluster in the group

    if show_color_scatter:
        values = np.array(values)
        _scatterColors._scatterColors(values, input_colorspace=input_colorspace, cluster_labels=labels)

    if return_values_as_centroids:
        labels = np.array(labels)  # Convert labels to a numpy array
        values = np.array(values)
        unique_labels = np.unique(labels)
        centroids = {}
        for label in unique_labels:
            centroids[label] = np.mean(values[labels == label], axis=0)

        values_centroids = np.array([centroids[label] for label in labels])

        return values_centroids

    return labels