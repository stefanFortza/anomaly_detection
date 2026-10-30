import numpy as np
from matplotlib import pyplot


def linear_model(x: float, sigma: float = 1.0) -> float:
    epsilon = np.random.standard_normal() * sigma
    return 0.5 * x - 1 + epsilon


def linear_model_2d(X: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    weights = np.array([[0.5], [0.5]])
    noise = np.random.standard_normal(size=(X.shape[0], 1)) * sigma
    return X @ weights - 1 + noise


def run():
    #    run_case_1d()
    run_case_2d()


def run_case_1d():
    # normals
    X_1 = np.random.standard_normal(size=(100, 1)) * 2
    # x high variance
    X_2 = np.random.standard_normal(size=(30, 1)) * 10
    # y high variance
    X_3 = np.random.standard_normal(size=(10, 1))
    # x and y high variance
    X_4 = np.random.standard_normal(size=(10, 1)) * 10

    X = np.concatenate((X_1, X_2, X_3, X_4), axis=0)

    y_1 = np.array([np.array([linear_model(x[0], 1)]) for x in X_1])
    y_2 = np.array([np.array([linear_model(x[0], 1)]) for x in X_2])
    y_3 = np.array([np.array([linear_model(x[0], 5)]) for x in X_3])
    y_4 = np.array([np.array([linear_model(x[0], 10)]) for x in X_4])

    y = np.concatenate((y_1, y_2, y_3, y_4), axis=0)

    data = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)  # add bias term

    # varianta lenta
    #     H = data @ np.linalg.inv(data.T @ data) @ data.T
    #
    #     l_scores = np.diagonal(H)
    #     print("Leverage scores:", l_scores)
    #     print("Leverage scores sum:", np.sum(l_scores))
    # Varianta rapida
    Q, _ = np.linalg.qr(data)
    l_scores = np.sum(Q**2, axis=1)
    print("Leverage scores:", l_scores)
    print("Leverage scores sum:", np.sum(l_scores))

    top_idx = np.argsort(l_scores)[-10:]

    pyplot.scatter(X_1, y_1, color="blue", label="Cluster 1")
    pyplot.scatter(X_2, y_2, color="green", label="Cluster 2")
    pyplot.scatter(X_3, y_3, color="orange", label="Cluster 3")
    pyplot.scatter(X_4, y_4, color="red", label="Cluster 4")

    pyplot.scatter(
        X[top_idx],
        y[top_idx],
        color="black",
        label="Top Leverage Scores",
        marker="x",
        s=100,
    )

    pyplot.xlabel("X")
    pyplot.ylabel("y")
    pyplot.title("Synthetic Data with Outliers")
    pyplot.legend()
    pyplot.show()


def run_case_2d():
    # normals
    X_1 = np.random.standard_normal(size=(100, 2)) * 2
    # x high variance
    X_2 = np.random.standard_normal(size=(30, 2)) * 10
    # y high variance
    X_3 = np.random.standard_normal(size=(10, 2))
    # x and y high variance
    X_4 = np.random.standard_normal(size=(10, 2)) * 10

    X = np.concatenate((X_1, X_2, X_3, X_4), axis=0)

    y_1 = linear_model_2d(X_1, 1)
    y_2 = linear_model_2d(X_2, 1)
    y_3 = linear_model_2d(X_3, 5)
    y_4 = linear_model_2d(X_4, 10)

    y = np.concatenate((y_1, y_2, y_3, y_4), axis=0)

    data = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)  # add bias term

    # varianta lenta
    #     H = data @ np.linalg.inv(data.T @ data) @ data.T
    #
    #     l_scores = np.diagonal(H)
    #     print("Leverage scores:", l_scores)
    #     print("Leverage scores sum:", np.sum(l_scores))
    # Varianta rapida
    Q, _ = np.linalg.qr(data)
    l_scores = np.sum(Q**2, axis=1)
    print("Leverage scores:", l_scores)
    print("Leverage scores sum:", np.sum(l_scores))


if __name__ == "__main__":
    run()
