# src/anomaly_detection/lab1/ex2.py
from typing import cast

import numpy as np
import pyod.utils
from matplotlib import pyplot
from numpy.typing import NDArray
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, roc_curve


def run():
    res = pyod.utils.data.generate_data_clusters(
        n_train=400, n_test=200, contamination=0.1, n_clusters=2
    )

    # PyOD returnează 4 tuple când n_test > 0, cast rezolvă type mismatch-ul Pyright
    X_train, X_test, y_train, y_test = cast(
        tuple[NDArray, NDArray, NDArray, NDArray], res
    )

    knn = KNN(contamination=0.1)
    knn.fit(X_train)

    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)

    fig, axes = pyplot.subplots(2, 2, figsize=(12, 5))
    plot_scatter(axes[0, 0], X_train, y_train)
    axes[0, 0].set_title("Train data")
    plot_scatter(axes[0, 1], X_test, y_test)
    axes[0, 1].set_title("Test data")
    plot_scatter(axes[1, 0], X_train, y_train_pred)
    axes[1, 0].set_title("Train data - predicted")
    plot_scatter(axes[1, 1], X_test, y_test_pred)
    axes[1, 1].set_title("Test data - predicted")
    pyplot.grid(True, linestyle=":", alpha=0.6)
    pyplot.show()


def plot_scatter(ax, X: np.ndarray, y: np.ndarray):
    inliers = X[y == 0]
    outliers = X[y == 1]

    ax.plot(inliers[:, 0], inliers[:, 1], color="blue", marker="o", linestyle="None")
    ax.plot(outliers[:, 0], outliers[:, 1], color="red", marker="x", linestyle="None")


if __name__ == "__main__":
    run()
