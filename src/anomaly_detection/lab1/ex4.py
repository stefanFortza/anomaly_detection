import numpy as np
from sklearn.datasets import make_spd_matrix
from sklearn.metrics import balanced_accuracy_score


def run():
    rng = np.random.default_rng(seed=42)
    n_samples = 1000
    contamination = 0.1
    n_outliers = int(n_samples * contamination)
    n_inliers = n_samples - n_outliers
    dimension = rng.integers(2, 50)

    sigma: np.ndarray = make_spd_matrix(dimension, random_state=42)
    mu = rng.uniform(-10, 10, size=dimension)
    L: np.ndarray = np.linalg.cholesky(sigma)

    # Generate inliers
    X_std = rng.standard_normal(size=(n_inliers, dimension))
    X_inliers = X_std @ L.T + mu
    y_inliers = np.zeros(n_inliers, dtype=int)

    # Generate outliers
    X_outliers_std = rng.standard_normal(size=(n_outliers, dimension)) * 4.0
    X_outliers = X_outliers_std @ L.T + mu
    y_outliers = np.ones(n_outliers, dtype=int)

    # Combine inliers and outliers
    X = np.vstack((X_inliers, X_outliers))
    y_train = np.concatenate((y_inliers, y_outliers))

    sigma_inv = np.linalg.inv(sigma)
    diff: np.ndarray = X - mu

    z_scores = np.sqrt(np.sum((diff @ sigma_inv) * diff, axis=1))
    threshold = np.quantile(z_scores, 0.90)
    y_pred = (z_scores > threshold).astype(int)

    ba = balanced_accuracy_score(y_train, y_pred)
    print(f"Balanced Accuracy: {ba:.4f}")


if __name__ == "__main__":
    run()
