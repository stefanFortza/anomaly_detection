import numpy as np
import pyod.utils
from sklearn.metrics import balanced_accuracy_score


def run():
    # Când n_test=0, PyOD generează doar (X_train, y_train)
    X_train, _, y_train, _ = pyod.utils.generate_data(
        n_train=1000, n_test=0, n_features=1, contamination=0.1
    )

    x: np.ndarray = X_train.ravel()

    mean = np.mean(x)
    std = np.std(x)
    z_scores = abs(x - mean) / std

    threshold = np.quantile(z_scores, 0.90, axis=0)

    y_pred = (z_scores > threshold).astype(int)

    ba = balanced_accuracy_score(y_train, y_pred)

    print(f"Balanced Accuracy: {ba:.4f}")


if __name__ == "__main__":
    run()
