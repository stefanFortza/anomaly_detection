import warnings

import numpy as np
import scipy
import torch
from pyod.models.dif import DIF
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.utils.data import train_test_split
from pyod.utils.utility import standardizer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

warnings.filterwarnings(
    "ignore",
    message=".*'pin_memory' argument is set as true but no accelerator is found.*",
    category=UserWarning,
)


def split_and_standardize_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_test = standardizer(X_train, X_test)
    return X_train, X_test, y_train, y_test


def fit_and_get_scores(model, X_train, X_test):
    model.fit(X_train)
    return model.decision_function(X_test)


def run(n_runs=10):
    data = scipy.io.loadmat("src/anomaly_detection/lab3/shuttle.mat")
    X = data["X"]
    y = data["y"].ravel()

    contamination = np.mean(y)

    models = [
        (
            "DIF",
            DIF(
                contamination=contamination,
                hidden_neurons=[50, 20, 10],
                device="cuda" if torch.cuda.is_available() else "cpu",
            ),
        ),
        ("LOF", LOF(contamination=contamination)),
        ("LODA", LODA(contamination=contamination)),
    ]

    for name, model in models:
        for i in range(n_runs):
            print(f"Run {i + 1}/{n_runs} for model: {name}")
            X_train, X_test, _, y_test = split_and_standardize_data(
                X, y, random_state=i
            )

            test_scores = fit_and_get_scores(model, X_train, X_test)

            # We don't use predict because it recalculates the decision scores
            # and it would be slower
            y_pred = (test_scores > model.threshold_).astype(int)

            ba_test = balanced_accuracy_score(y_test, y_pred)
            auc_test = roc_auc_score(y_test, test_scores)
            print(f"{name} - Test Balanced Accuracy: {ba_test:.4f}")
            print(f"{name} - Test AUC: {auc_test:.4f}")


if __name__ == "__main__":
    run()
