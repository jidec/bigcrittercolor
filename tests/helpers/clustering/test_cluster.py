import pytest
import numpy as np
from sklearn.datasets import make_blobs
from bigcrittercolor.helpers.clustering import _cluster  # Adjust the import path as necessary


# Fixture for generating sample data
@pytest.fixture
def sample_data():
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    return X


# Basic functionality tests for each algorithm
@pytest.mark.parametrize("algo,n_clusters", [
    ("kmeans", 3),
    ("gaussian_mixture", 3),
    ("agglom", 3),
    ("dbscan", None),  # DBSCAN does not use the n_clusters argument
    ("affprop", None),  # AffinityPropagation does not use the n_clusters argument
])
def test_cluster_algorithms_basic(sample_data, algo, n_clusters):
    kwargs = {"algo": algo, "show": False}
    if n_clusters is not None:
        kwargs["n"] = n_clusters

    labels = _cluster(sample_data, **kwargs)

    assert len(labels) == 100, f"Expected labels for each sample using {algo}"
    if n_clusters is not None:
        assert len(np.unique(labels)) <= n_clusters, f"Expected no more than {n_clusters} clusters using {algo}"


# Test scaling effect
def test_cluster_scaling_effect(sample_data):
    labels_without_scaling = _cluster(sample_data, algo="kmeans", n=3, scale=False, show=False)
    labels_with_scaling = _cluster(sample_data, algo="kmeans", n=3, scale=True, show=False)

    assert len(labels_without_scaling) == len(labels_with_scaling), "Scaling should not change the number of labels"
    # Further assertions might compare the actual clusters, but due to scaling, the direct comparison may not be meaningful


# Test weights impact
def test_cluster_weights(sample_data):
    weights = np.random.rand(sample_data.shape[1])
    labels = _cluster(sample_data, algo="kmeans", n=3, weights=weights, show=False)

    assert len(labels) == 100, "Expected labels for each sample with weights"


# Test find_n_minmax functionality
def test_find_n_minmax(sample_data):
    labels = _cluster(sample_data, algo="kmeans", find_n_minmax=(2, 5), find_n_metric="ch", show=False)
    # Assert based on the expected behavior, such as the number of clusters being within the specified range
    assert 2 <= len(np.unique(labels)) <= 5, "Expected number of clusters to be within the range of 2 to 5"


# Testing invalid algorithm parameter
def test_invalid_algorithm(sample_data):
    with pytest.raises(ValueError):
        _cluster(sample_data, algo="invalid_algo", n=3, show=False)


# Test outlier detection
def test_outlier_detection(sample_data):
    labels = _cluster(sample_data, algo="kmeans", n=3, outlier_percentile=95, show=False)
    assert -1 in labels, "Expected outliers to be labeled as -1"


# Test return_values_as_centroids
def test_return_values_as_centroids(sample_data):
    centroids = _cluster(sample_data, algo="kmeans", n=3, return_values_as_centroids=True, show=False)
    assert centroids.shape == sample_data.shape, "Expected returned centroids to match input shape"


# Test merge_with_user_input - This test would ideally mock user input to test functionality
# This test requires a mock or adjustment to the function to inject user input programmatically
@pytest.mark.skip(reason="Requires mocking or function adjustment for user input")
def test_merge_with_user_input():
    pass

# Add more tests as needed for other parameters and edge cases