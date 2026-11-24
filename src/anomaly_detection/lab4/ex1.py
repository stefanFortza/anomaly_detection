from typing import cast

import numpy as np
import numpy.random
import sklearn
import sklearn.datasets
from matplotlib import pyplot
from numpy.typing import NDArray
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.ocsvm import OCSVM
from pyod.utils.data import generate_data
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def fit_and_get_scores(
    model, X_train: NDArray[np.float64], X_test: NDArray[np.float64]
) -> NDArray[np.float64]:
    model.fit(X_train)
    return model.decision_function(X_test)


def plot_scatter(
    ax,
    X: NDArray[np.float64],
    y: NDArray[np.int_],
    title: str,
    xlabel: str,
    ylabel: str,
):
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="coolwarm", edgecolor="k")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def run():
    data = generate_data(
        n_train=300,
        n_test=200,
        n_features=3,
        contamination=0.15,
        random_state=42,
    )

    X_train: NDArray[np.float64]
    y_train: NDArray[np.int64]
    X_test: NDArray[np.float64]
    y_test: NDArray[np.int64]
    X_train, X_test, y_train, y_test = cast(
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.int64],
            NDArray[np.int64],
        ],
        data,
    )

    model = OCSVM(contamination=0.15, kernel="rbf")

    test_scores = fit_and_get_scores(model, X_train, X_test)
    y_pred = (test_scores > model.threshold_).astype(int)
    ba_test = balanced_accuracy_score(y_test, y_pred)
    auc_test = roc_auc_score(y_test, test_scores)

    print(f"Balanced Accuracy (Test): {ba_test:.4f}")
    print(f"ROC AUC Score (Test): {auc_test:.4f}")

    fig, axes = pyplot.subplots(
        2, 2, figsize=(10, 10), subplot_kw={"projection": "3d"}, constrained_layout=True
    )

    plot_scatter(axes[0, 0], X_train, y_train, "Training Data", "X1", "X2")
    plot_scatter(axes[1, 0], X_train, model.labels_, "Predicted Train", "X1", "X2")
    plot_scatter(axes[0, 1], X_test, y_test, "Test Data", "X1", "X2")
    plot_scatter(axes[1, 1], X_test, y_pred, "Predicted Test", "X1", "X2")

    pyplot.show()

    model = DeepSVDD(3, contamination=0.15, epochs=5)

    test_scores = fit_and_get_scores(model, X_train, X_test)
    y_pred = (test_scores > model.threshold_).astype(int)
    ba_test = balanced_accuracy_score(y_test, y_pred)
    auc_test = roc_auc_score(y_test, test_scores)

    print(f"Balanced Accuracy (Test): {ba_test:.4f}")
    print(f"ROC AUC Score (Test): {auc_test:.4f}")

    fig, axes = pyplot.subplots(
        2, 2, figsize=(10, 10), subplot_kw={"projection": "3d"}, constrained_layout=True
    )

    plot_scatter(axes[0, 0], X_train, y_train, "Training Data", "X1", "X2")
    plot_scatter(axes[1, 0], X_train, model.labels_, "Predicted Train", "X1", "X2")
    plot_scatter(axes[0, 1], X_test, y_test, "Test Data", "X1", "X2")
    plot_scatter(axes[1, 1], X_test, y_pred, "Predicted Test", "X1", "X2")

    pyplot.show()


if __name__ == "__main__":
    run()
