# src/anomaly_detection/lab1/ex2.py
from typing import cast

import pyod.utils
from matplotlib import pyplot
from numpy.typing import NDArray
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, roc_curve


def run():
    res = pyod.utils.generate_data(
        n_train=600, n_test=200, n_features=2, contamination=0.2
    )
    # PyOD returnează 4 tuple când n_test > 0, cast rezolvă type mismatch-ul Pyright
    X_train, X_test, _y_train, y_test = cast(
        tuple[NDArray, NDArray, NDArray, NDArray], res
    )

    knn = KNN(contamination=0.2)
    knn.fit(X_train)

    y_test_pred = knn.predict(X_test)
    y_test_scores = knn.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_test_scores)

    pyplot.figure(figsize=(6, 5))
    pyplot.plot(fpr, tpr, label="ROC curve")
    pyplot.plot(
        [0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance"
    )
    pyplot.xlim((-0.05, 1.0))
    pyplot.ylim((0.0, 1.05))
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.title("ROC Curve")
    pyplot.legend(loc="lower right")
    pyplot.grid(True, linestyle=":", alpha=0.6)
    pyplot.show()

    confusion_matrix_train = confusion_matrix(y_test, y_test_pred)

    true_negative = confusion_matrix_train[0, 0]
    false_positive = confusion_matrix_train[0, 1]
    false_negative = confusion_matrix_train[1, 0]
    true_positive = confusion_matrix_train[1, 1]

    true_positive_rate = true_positive / (true_positive + false_negative)
    true_negative_rate = true_negative / (true_negative + false_positive)
    balanced_accuracy = (true_positive_rate + true_negative_rate) / 2

    print("True Positive Rate: ", true_positive_rate)
    print("True Negative Rate: ", true_negative_rate)
    print("Balanced Accuracy: ", balanced_accuracy)


if __name__ == "__main__":
    run()
