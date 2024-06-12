import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.cluster import KMeans, HDBSCAN, OPTICS, SpectralClustering, AgglomerativeClustering, DBSCAN, AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from scipy.cluster import hierarchy
from scipy.spatial import distance

from bigcrittercolor.helpers import _scatterColors, _bprint

# convenience function that wraps several sklearn clusters algorithms and returns labels
# kmeans, gaussian_mixture, agglom, dbscan
def _cluster(values, algo="kmeans", n=3,
             find_n_minmax = None, find_n_metric = "ch",
             eps = 0.1,
             min_samples = 5,
             cluster_selection_method="eom",
             linkage="ward",
             preference=None,
             scale=None, # whether to scale the features between 0 and 1 to equalize the importance of each (happens before weighting if applicable)
             weights=None, # a list of weights, one for each column, that make certain features more or less important
             pca_n=None,
             show_pca_tsne = False,
             show_color_scatter = False,
             show_silhouette = False,
             input_colorspace = "rgb",
             show=True,
             outlier_percentile=None, return_fuzzy_probs=False,
             unique_values_only=False,
             dbscan_outliers_to_nearest_centroid=False,
             show_color_centroids=False,
             merge_with_user_input=False,
             return_values_as_centroids=False,
             print_steps=False,
             print_details=False):

    #_bprint._bprint(print_steps, "Shape of values: " + str(np.shape(values)))
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
    if scale == "minmax":
        scaler = MinMaxScaler()
        # Scale the data
        values = scaler.fit_transform(values)
    if scale == "standard":
        scaler = StandardScaler()
        # Scale the data
        values = scaler.fit_transform(values)

    if weights is not None:
        weights = np.array(weights)
        values = values * weights

    if pca_n is not None:
        pca = PCA(n_components=pca_n)
        values = pca.fit_transform(values)

    if unique_values_only:
        values, indices, inverse = np.unique(values, axis=0, return_index=True, return_inverse=True)
        _bprint._bprint(print_steps, "Using " + str(np.shape(values)[0]) + " unique values only...")

    # find n
    if find_n_minmax is not None:
        _bprint._bprint(print_steps,"Finding cluster N using metric(s)...")

        # if n_components > n_samples, we get an error - avoid this by capping find_n_minmax[1] at n rows of values - 1
        if find_n_minmax[1] >= values.shape[0]:
            find_n_minmax = (find_n_minmax[0], values.shape[0]-1)
        # create a vector of ns to try for knee assessment
        ns = np.arange(find_n_minmax[0], find_n_minmax[1] + 1)

        match algo:
            case "kmeans":
                models = [KMeans(n_clusters=n).fit(values) for n in ns]
            case "gaussian_mixture":
                models = [GaussianMixture(n_components=n, covariance_type='full', reg_covar=1e-5).fit(values) for n in ns]

        # assemble list of lists of labels, one for each model with a different n
        labels_list = [m.fit_predict(values) for m in models]

        # these metrics ONLY work if you're not evaluating n=1
        includes_n1 = find_n_minmax[0] == 1
        if not includes_n1:
            ch_scores = [calinski_harabasz_score(values, l) for l in labels_list]
            db_scores = [davies_bouldin_score(values, l) for l in labels_list]
            sil_scores = [silhouette_score(values, l) for l in labels_list]

        # aics ONLY apply to gaussian mixture models
        if algo == "gaussian_mixture":
            aic_scores = [m.aic(values) for m in models]

        if show:
            if not includes_n1:
                plt.plot(ns, ch_scores)
                plt.title('Calinski-Harabasz Scores')
                plt.xlabel('Number of clusters')
                plt.ylabel('CH score')
                plt.show()

                plt.plot(ns, db_scores)
                plt.title('Inverse Davies-Bouldin Scores')
                plt.xlabel('Number of clusters')
                plt.ylabel('Inverse DB score')
                # Reverse the Y-axis
                plt.gca().invert_yaxis()
                plt.show()

                plt.plot(ns, sil_scores, label='Silhouette score')
                plt.title('Silhouette Scores')
                plt.xlabel('Number of clusters')
                plt.ylabel('Silhouette score')
                plt.show()
            if algo == "gaussian_mixture":
                plt.plot(ns, aic_scores)
                plt.title('Cluster Model AICs')
                plt.xlabel('Number of clusters')
                plt.ylabel('AIC')
                # Reverse the Y-axis
                plt.gca().invert_yaxis()
                plt.show()

        match find_n_metric:
            case "ch":
                n = ns[np.argmax(ch_scores)]
            case "db":
                n = ns[np.argmin(db_scores)]
            case "aic":
                n = ns[np.argmin(aic_scores)]
            case "all":
                def rank_scores(scores):
                    """Rank the scores within a set, lower scores get higher ranks."""
                    sorted_scores = sorted(scores, reverse=True)
                    ranks = [sorted_scores.index(score) + 1 for score in scores]
                    return ranks
                def aggregate_ranks(ranks_set1, ranks_set2, ranks_set3):
                    """Aggregate ranks for each value and sort by total rank."""
                    aggregated_ranks = [sum(ranks) for ranks in zip(ranks_set1, ranks_set2, ranks_set3)]
                    value_ranks_pairs = list(zip(ns, aggregated_ranks))
                    sorted_by_ranks = sorted(value_ranks_pairs, key=lambda x: x[1])
                    return sorted_by_ranks
                def rank_values_by_scores(scores_set1, scores_set2, scores_set3):
                    # Rank the scores within each set
                    ranks_set1 = rank_scores(scores_set1)
                    ranks_set2 = rank_scores(scores_set2)
                    ranks_set3 = rank_scores(scores_set3)
                    #ranks_set4 = rank_scores(scores_set4)

                    # Aggregate and sort the ranks
                    sorted_values_ranks = aggregate_ranks(ranks_set1, ranks_set2, ranks_set3)

                    return sorted_values_ranks
                n = rank_values_by_scores(ch_scores,db_scores,sil_scores)[0][0]

        _bprint._bprint(print_steps, "Using N of " + str(n) + "...")
    # create cluster model
    match algo:
        case "kmeans":
            model = KMeans(n_clusters=n)
        case "gaussian_mixture":
            model = GaussianMixture(n_components=n,covariance_type='full',reg_covar=1e-5)#,reg_covar=2)
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
            eps = findDBSCANeps(values)
            min_samples = np.shape(values)[1] + 1
            model = DBSCAN(eps=eps, min_samples=min_samples)#, #algorithm='ball_tree')  # , metric='manhattan')
        case "hdbscan":
            #eps = findDBSCANeps(values)
            min_samples = np.shape(values)[1] + 1
            #min_samples=25
            model = HDBSCAN(min_samples=min_samples,cluster_selection_method=cluster_selection_method)#cluster_selection_epsilon=eps)#cluster_selection_method="leaf")
        case "affprop":
            if preference is None:
                distances = euclidean_distances(values, squared=True)
                negative_distances = -distances
                preference = np.median(negative_distances)
            model = AffinityPropagation(preference=preference)
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

    # if unique_values_only
    if unique_values_only:
        labels = labels[inverse]
        values = start_values

    # if merging with user input, ALWAYS show the color centroids
    if merge_with_user_input:
        show_color_centroids = True

    # visualize the color clusters by their centroids
    if show_color_centroids:
        # Print centroids of each cluster
        centroids = {}

        labels = np.array(labels)  # Convert labels to a numpy array
        values = np.array(values)

        for label in np.unique(labels):
            centroids[label] = np.mean(values[labels == label], axis=0)

        # Convert centroids to RGB if necessary
        if input_colorspace in ["cielab", "hls"]:
            for label in centroids:
                if input_colorspace == "cielab":
                    # Convert from CIELAB to RGB
                    lab = np.uint8(np.round(centroids[label])).reshape(1, 1, 3)
                    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) #RGB
                    centroids[label] = rgb[0, 0]
                elif input_colorspace == "hls":
                    # Convert from HLS to RGB
                    hls = np.uint8(np.round(centroids[label])).reshape(1, 1, 3)
                    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
                    centroids[label] = rgb[0, 0]
        # print centroids
        for label in np.unique(labels):
            print(f"Centroid of cluster {label}: {centroids[label]}")

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

    # if merge with user input, take input of the form 1,2;3,4;5,6 to merge clusters
    if merge_with_user_input:
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
        scatter_values = np.unique(values, axis=0)
        #values = np.array(values)

        # Find indices where the first column value is 0 or less
        indices_to_remove = [i for i, row in enumerate(scatter_values) if row[0] <= 0]
        # Remove those rows from the 2D array
        values2 = [row for i, row in enumerate(scatter_values) if i not in indices_to_remove]
        values2 = np.array(values2)
        # Remove the same indices from the separate list
        labels2 = [item for i, item in enumerate(labels) if i not in indices_to_remove]

        _scatterColors._scatterColors(values2, input_colorspace=input_colorspace, cluster_labels=labels2)

    # if force_assign_outliers:
    # # Identify points in each cluster and outliers
    # core_samples_mask = np.zeros_like(labels, dtype=bool)
    # core_samples_mask[model.core_sample_indices_] = True
    # unique_labels = set(labels)
    #
    # # Calculate centroids of the clusters
    # centroids = []
    # for k in unique_labels:
    #     if k != -1:  # Ignoring noise if present
    #         class_member_mask = (labels == k)
    #         xy = values[class_member_mask & core_samples_mask]
    #         centroids.append(np.mean(xy, axis=0))
    #
    # # Convert list to array for distance calculation
    # centroids = np.array(centroids)
    #
    # # Forcibly assign outliers to the nearest cluster centroid
    # outlier_indices = np.where(labels == -1)[0]
    # if len(centroids) > 0 and len(outlier_indices) > 0:
    #     closest, _ = pairwise_distances_argmin_min(values[outlier_indices], centroids)
    #     labels[outlier_indices] = [labels[model.core_sample_indices_[closest[i]]] for i in range(len(closest))]

    if show_silhouette:
        silhouette_avg = silhouette_score(values, labels)
        sample_silhouette_values = silhouette_samples(values, labels)

        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(10, 7)

        y_lower = 10
        for i in range(n):
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.nipy_spectral(float(i) / n)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])
        ax1.set_xticks(np.arange(-0.1, 1.1, 0.2))
        plt.show()

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

from kneed import KneeLocator
def findDBSCANeps(data, k=4):
    """
    Finds and visualizes the optimal eps parameter for DBSCAN clustering by identifying the knee point
    in the k-distance plot.

    Parameters:
        data (array-like): The input dataset for which to calculate the eps parameter.
        k (int): The number of nearest neighbors to consider for the k-distance plot.

    Returns:
        float: The recommended eps value for DBSCAN clustering, and visualizes the plot.
    """
    # Compute the nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)

    # Sort the distances to the k-th nearest neighbor
    sorted_distances = np.sort(distances[:, k-1], axis=0)

    # Use KneeLocator to find the knee point
    knee_locator = KneeLocator(range(len(sorted_distances)), sorted_distances, curve='convex', direction='increasing',S=3)

    # Determine the eps value at the knee point
    eps_value = sorted_distances[knee_locator.knee] if knee_locator.knee is not None else None

    # Plotting the k-distance graph and the knee point
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_distances)
    plt.title('k-Distance Plot')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'Distance to {k}-th nearest neighbor')

    # Highlight the knee
    if knee_locator.knee is not None:
        plt.axvline(x=knee_locator.knee, color='red', linestyle='--', label=f'Knee point at index {knee_locator.knee}')
        plt.legend()

    plt.show()

    # subtract 5
    eps_value = eps_value
    return eps_value