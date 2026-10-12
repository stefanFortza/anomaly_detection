import pyod
from matplotlib import pyplot
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.neighbors import LocalOutlierFactor


def main() -> None:
    print("Lab 1 running...")
    # ex1()
    # ex2()
    ex3()


def ex1():
    data = pyod.utils.generate_data(
        n_train=400, n_test=100, n_features=2, contamination=0.1
    )

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

    y_pred = lof.fit_predict(data[0])
    outliers = data[0][y_pred == -1]
    non_outliers = data[0][y_pred == 1]

    pyplot.scatter(outliers[:, 0], outliers[:, 1], color="red", label="Outliers")
    pyplot.scatter(
        non_outliers[:, 0], non_outliers[:, 1], color="blue", label="Inliers"
    )
    pyplot.show()


def ex2():
    X_train, X_test, y_train, y_test = pyod.utils.generate_data(
        n_train=600, n_test=200, n_features=2, contamination=0.2
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


def ex3():
    X_train, X_test, y_train, y_test = pyod.utils.generate_data(
        n_train=1000, n_test=0, n_features=2, contamination=0.1
    )


if __name__ == "__main__":
    main()
