# src/anomaly_detection/lab1/ex2.py
from typing import cast

import numpy as np
import pyod.utils
import sklearn
import sklearn.datasets
from matplotlib import pyplot
from numpy.typing import NDArray
from pyod.models.base import BaseDetector
from pyod.models.dif import DIF
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF


def fit_and_get_scores(
    model: BaseDetector, X: NDArray[np.float64], y: NDArray[np.int_]
) -> NDArray[np.float64]:
    model.fit(X, y)
    return model.decision_function(X)


def run_2d():
    data = sklearn.datasets.make_blobs(
        n_samples=[500, 500],
        n_features=2,
        centers=[(10, 0), (0, 10)],
        cluster_std=[1, 1],
    )

    X_train, y_train = cast(tuple[NDArray[np.float64], NDArray[np.int_]], data)
    X_test = np.random.uniform(-10, 20, size=(1000, 2))

    models = [
        ("DIF", DIF(contamination=0.02, hidden_neurons=[50, 20, 10])),
        ("IForest", IForest(contamination=0.02)),
        ("LOF", LOF(contamination=0.02)),
    ]

    fig, axes = pyplot.subplots(1, len(models), figsize=(15, 5))

    for ax, (name, model) in zip(axes, models):
        test_scores = fit_and_get_scores(model, X_train, y_train)

        ax.set_title(name)
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.grid(True, linestyle=":", alpha=0.6)

        pyplot.sca(ax)
        pyplot.scatter(
            X_train[:, 0],
            X_train[:, 1],
            c=test_scores,
            cmap="viridis",
        )
        pyplot.colorbar(label="Anomaly score")

    pyplot.show()


def run_3d():
    data = sklearn.datasets.make_blobs(
        n_samples=[500, 500],
        n_features=3,
        centers=[(0, 10, 0), (10, 0, 10)],
        cluster_std=[1, 1],
    )

    X_train, y_train = cast(tuple[NDArray[np.float64], NDArray[np.int_]], data)
    X_test = np.random.uniform(-10, 20, size=(1000, 3))

    models = [
        ("DIF", DIF(contamination=0.02, hidden_neurons=[50, 20, 10])),
        ("IForest", IForest(contamination=0.02)),
        ("LOF", LOF(contamination=0.02)),
    ]

    fig, axes = pyplot.subplots(
        1, 3, figsize=(18, 6), subplot_kw={"projection": "3d"}, constrained_layout=True
    )

    for ax, (name, model) in zip(axes, models):
        test_scores = fit_and_get_scores(model, X_train, y_train)

        ax.set_title(name)
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("X3")

        sc = ax.scatter(
            X_train[:, 0],
            X_train[:, 1],
            X_train[:, 2],
            c=test_scores,
            cmap="viridis",
        )
        fig.colorbar(sc, ax=ax, label="Anomaly score")

    pyplot.show()


def run():
    # run_2d()
    run_3d()


if __name__ == "__main__":
    run()
