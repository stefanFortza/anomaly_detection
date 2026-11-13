from typing import cast

import numpy as np
import numpy.random
import sklearn
import sklearn.datasets
from matplotlib import pyplot
from numpy.typing import NDArray

from anomaly_detection.lab2.ex4 import predict_train_and_test_from_scores


def run():
    data = sklearn.datasets.make_blobs(
        n_samples=500,
        n_features=2,
        centers=[(0, 0)],
        cluster_std=1.0,
        random_state=42,
    )

    X_train, y_train = cast(tuple[NDArray[np.float64], NDArray[np.int_]], data)
    X_test = np.random.uniform(-3, 3, size=(500, 2))

    unit_vectors = np.random.multivariate_normal(
        mean=np.zeros(2), cov=np.eye(2), size=5
    )
    unit_vectors /= np.linalg.norm(unit_vectors, axis=1)[:, np.newaxis]
    z_matrix = X_train @ unit_vectors.T

    min_value = np.min(z_matrix) - 1
    max_value = np.max(z_matrix) + 1
    num_bins = 10
    bins = np.linspace(min_value, max_value, num_bins)

    histograms: list[np.ndarray] = []

    for i in range(z_matrix.shape[1]):
        histogram, _ = np.histogram(z_matrix[:, i], bins=bins)
        histograms.append(histogram)

    histogram_probabilities: list[np.ndarray] = []

    for histogram in histograms:
        histogram_probability = histogram / np.sum(histogram)
        histogram_probabilities.append(histogram_probability)

    z_test_matrix = X_test @ unit_vectors.T
    test_indices = np.digitize(z_test_matrix, bins) - 1

    test_scores: list[np.ndarray] = []

    for i in range(z_test_matrix.shape[1]):
        test_score = histogram_probabilities[i][test_indices[:, i]]
        test_scores.append(test_score)

    test_scores_matrix = np.column_stack(test_scores)
    test_scores_combined = np.mean(test_scores_matrix, axis=1)
    print(f"Combined test scores: {test_scores_combined}")

    pyplot.scatter(X_test[:, 0], X_test[:, 1], c=test_scores_combined, cmap="viridis")
    pyplot.colorbar(label="Anomaly score")
    pyplot.xlabel("X1")
    pyplot.ylabel("X2")
    pyplot.title("Anomaly Detection using Histogram-based Method")
    pyplot.grid(True, linestyle=":", alpha=0.6)
    pyplot.show()


if __name__ == "__main__":
    run()
