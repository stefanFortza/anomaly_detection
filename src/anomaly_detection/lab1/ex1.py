import pyod.utils
from matplotlib import pyplot
from sklearn.neighbors import LocalOutlierFactor


def run():
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


if __name__ == "__main__":
    run()
