import numpy as np
from matplotlib import pyplot
from pyod.models import knn, lof
from sklearn.datasets import make_blobs


def run():
    # Când n_test=0, PyOD generează doar (X_train, y_train)
    data = make_blobs(
        n_samples=[200, 100],
        n_features=2,
        centers=[(-10, -10), (10, 10)],
        cluster_std=[2.0, 6.0],
        random_state=42,
    )

    X, y = data[0], data[1]

    knn_model = knn.KNN(contamination=0.07)
    knn_model.fit(X)
    y_pred_knn = knn_model.predict(X)

    lof_model = lof.LOF(contamination=0.07)
    lof_model.fit(X)
    y_pred_lof = lof_model.predict(X)

    fig, axes = pyplot.subplots(1, 2, figsize=(12, 5))
    plot_scatter(axes[0], X, y_pred_knn)
    axes[0].set_title("KNN - predicted")
    plot_scatter(axes[1], X, y_pred_lof)
    axes[1].set_title("LOF - predicted")
    pyplot.grid(True, linestyle=":", alpha=0.6)
    pyplot.show()


def plot_scatter(ax, X: np.ndarray, y: np.ndarray):
    inliers = X[y == 0]
    outliers = X[y == 1]

    ax.plot(inliers[:, 0], inliers[:, 1], color="blue", marker="o", linestyle="None")
    ax.plot(outliers[:, 0], outliers[:, 1], color="red", marker="x", linestyle="None")


if __name__ == "__main__":
    run()
