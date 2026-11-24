import numpy as np
import scipy
import sklearn
import sklearn.datasets
from numpy.typing import NDArray
from pyod.models.base import BaseDetector
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import OneClassSVM


def fit_and_get_scores(
    model: BaseDetector, X: NDArray[np.float64], y: NDArray[np.int_]
) -> NDArray[np.float64]:
    model.fit(X, y)
    return model.decision_function(X)


def run():
    data = scipy.io.loadmat("src/anomaly_detection/lab2/cardio.mat")
    X = data["X"]
    y = data["y"].ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6, random_state=42
    )

    contamination = np.sum(y_train == 1) / len(y_train)

    y_train = np.where(y_train == 0, 1, -1)
    y_test = np.where(y_test == 0, 1, -1)

    pipe = sklearn.pipeline.Pipeline(
        [
            ("scaler", sklearn.preprocessing.StandardScaler()),
            ("svm", OneClassSVM()),
        ]
    )

    clf = GridSearchCV(
        estimator=pipe,
        param_grid=[
            {
                "svm__kernel": ["rbf", "poly", "sigmoid"],
                "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
                "svm__nu": [0.01, 0.05, contamination, 0.1, 0.2, 0.5],
            },
            {
                "svm__kernel": ["linear"],
                "svm__nu": [0.01, 0.05, contamination, 0.1, 0.2, 0.5],
            },
        ],
        scoring="balanced_accuracy",
    )

    clf = clf.fit(X_train, y_train)

    y_test_pred: np.ndarray = clf.predict(X_test)

    ba_test: np.float64 = balanced_accuracy_score(y_test, y_test_pred)

    print("Best parameters:", clf.best_params_)
    print(f"Best CV Balanced Accuracy:   {clf.best_score_:.4f}")
    print(f"Test Balanced Accuracy:      {ba_test:.4f}")


if __name__ == "__main__":
    run()
