import numpy as np
import scipy.io
from numpy.typing import NDArray
from pyod.models.base import BaseDetector
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


def fit_and_print_scores(
    X_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    y_test: NDArray[np.int64],
    clf,
    label: str = "",
) -> None:
    clf.fit(X_train)

    if isinstance(clf, BaseDetector):
        anomaly_scores = clf.decision_function(X_test)
        y_pred = (anomaly_scores > clf.threshold_).astype(int)
    else:
        anomaly_scores = -clf.decision_function(X_test)
        y_pred = np.where(clf.predict(X_test) == -1, 1, 0)

    name = label or clf.__class__.__name__
    print(f"{name}:")
    print(f"  Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"  AUC:               {roc_auc_score(y_test, anomaly_scores):.4f}\n")


def run():
    data = scipy.io.loadmat("src/anomaly_detection/lab4/shuttle.mat")
    X = data["X"]
    y = data["y"].ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6, random_state=42, stratify=y
    )
    contamination = float(np.mean(y_train))

    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_test = standard_scaler.transform(X_test)

    models = [
        (
            "OneClassSVM (RBF)",
            OneClassSVM(kernel="rbf", gamma="auto", nu=contamination),
        ),
        (
            "DeepSVDD [64, 32]",
            DeepSVDD(
                n_features=X_train.shape[1],
                use_ae=False,
                hidden_neurons=[64, 32],
                hidden_activation="relu",
                output_activation="relu",
                dropout_rate=0.1,
                l2_regularizer=1e-4,
                epochs=25,
                batch_size=64,
                contamination=contamination,
                random_state=42,
                verbose=0,
            ),
        ),
        (
            "DeepSVDD [32, 16]",
            DeepSVDD(
                n_features=X_train.shape[1],
                use_ae=False,
                hidden_neurons=[32, 16],
                hidden_activation="relu",
                output_activation="relu",
                dropout_rate=0.0,
                l2_regularizer=1e-4,
                epochs=15,
                batch_size=64,
                contamination=contamination,
                random_state=42,
                verbose=0,
            ),
        ),
    ]

    for label, model in models:
        fit_and_print_scores(X_train, X_test, y_test, model, label=label)


if __name__ == "__main__":
    run()
