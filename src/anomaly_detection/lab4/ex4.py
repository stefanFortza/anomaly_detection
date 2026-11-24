import numpy as np
import pyod.models.combination
import scipy
from pyod.models.knn import KNN

# from pyod.models.combination import av
from pyod.utils.data import train_test_split
from pyod.utils.utility import standardizer
from sklearn.metrics import balanced_accuracy_score


def run():
    data = scipy.io.loadmat("src/anomaly_detection/lab2/cardio.mat")
    X = data["X"]
    y = data["y"].ravel()

    contamination = np.mean(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_test = standardizer(X_train, X_test)

    n_neighbors = np.linspace(30, 120, 10, dtype=int)

    train_scores_matrix = np.zeros((len(y_train), len(n_neighbors)))
    test_scores_matrix = np.zeros((len(y_test), len(n_neighbors)))

    for i, k in enumerate(n_neighbors):
        model = KNN(contamination=contamination, n_neighbors=k)
        model.fit(X_train)

        test_scores = model.decision_function(X_test)

        # We don't use predict because it recalculates the decision scores
        # and it would be slower
        y_pred = (test_scores > model.threshold_).astype(int)

        train_scores_matrix[:, i] = model.decision_scores_
        test_scores_matrix[:, i] = test_scores

        ba_train = balanced_accuracy_score(y_train, model.labels_)
        ba_test = balanced_accuracy_score(y_test, y_pred)
        print(f"k={k:3d} | Train BA: {ba_train:.4f} | Test BA: {ba_test:.4f}")

    y_pred_train_combined, y_pred_test_combined = predict_train_and_test_from_scores(
        train_scores_matrix,
        test_scores_matrix,
        contamination,
        pyod.models.combination.average,
    )

    print_balanced_accuracy(
        y_train, y_pred_train_combined, y_test, y_pred_test_combined, "Average"
    )

    y_pred_train_combined, y_pred_test_combined = predict_train_and_test_from_scores(
        train_scores_matrix,
        test_scores_matrix,
        contamination,
        pyod.models.combination.maximization,
    )

    print_balanced_accuracy(
        y_train, y_pred_train_combined, y_test, y_pred_test_combined, "Maximization"
    )


def predict_train_and_test_from_scores(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    contamination: float,
    strategy_func,
):
    train_scores_norm, test_scores_norm = standardizer(train_scores, test_scores)

    train_scores_combined = strategy_func(train_scores_norm)
    test_scores_combined = strategy_func(test_scores_norm)

    treshold_combined = np.quantile(train_scores_combined, 1 - contamination)

    y_pred_train_combined = (train_scores_combined > treshold_combined).astype(int)
    y_pred_test_combined = (test_scores_combined > treshold_combined).astype(int)

    return y_pred_train_combined, y_pred_test_combined


def print_balanced_accuracy(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    strategy_name: str,
):
    ba_train = balanced_accuracy_score(y_true_train, y_pred_train)
    ba_test = balanced_accuracy_score(y_true_test, y_pred_test)
    print(f"{strategy_name} - Train Balanced Accuracy: {ba_train:.4f}")
    print(f"{strategy_name} - Test Balanced Accuracy: {ba_test:.4f}")


if __name__ == "__main__":
    run()
