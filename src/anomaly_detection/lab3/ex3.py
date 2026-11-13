import sys
import time
import warnings

import numpy as np
import scipy.io
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

    contamination = float(np.mean(y))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sys.stderr.write(f"[INFO] Device: {device} | Dataset shape: {X.shape}\n")

    # Stocam metricile pe fiecare run pentru a afisa si media finala
    results = {name: {"ba": [], "auc": []} for name in ["DIF", "LOF", "LODA"]}

    for i in range(n_runs):
        X_train, X_test, _, y_test = split_and_standardize_data(X, y, random_state=i)

        # Instantiere proaspata la fiecare iteratie pentru a preveni acumularea de stari in DIF
        models = [
            (
                "DIF",
                DIF(
                    contamination=contamination,
                    hidden_neurons=[50, 20, 10],
                    batch_size=2048,
                    device=device,
                    random_state=i,
                ),
            ),
            ("LOF", LOF(contamination=contamination, n_jobs=-1)),
            ("LODA", LODA(contamination=contamination)),
        ]

        for name, model in models:
            t0 = time.perf_counter()
            test_scores = fit_and_get_scores(model, X_train, X_test)
            y_pred = (test_scores > model.threshold_).astype(int)

            ba_test = balanced_accuracy_score(y_test, y_pred)
            auc_test = roc_auc_score(y_test, test_scores)
            elapsed = time.perf_counter() - t0

            results[name]["ba"].append(ba_test)
            results[name]["auc"].append(auc_test)

            msg = f"Run {i + 1}/{n_runs} | {name:4s} | BA: {ba_test:.4f} | AUC: {auc_test:.4f} | Time: {elapsed:.2f}s\n"
            print(msg, end="")
            sys.stderr.write(msg)

    print("\n--- Summary (Mean across runs) ---")
    for name, metrics in results.items():
        print(
            f"{name:4s} | Mean BA: {np.mean(metrics['ba']):.4f} | Mean AUC: {np.mean(metrics['auc']):.4f}"
        )


if __name__ == "__main__":
    run()
