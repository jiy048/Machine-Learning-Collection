"""
From scratch implementation of Weighted Logistic Regression.

X has shape (m, n), y has shape (m, 1), and weights have
shape (n + 1, 1) after adding the bias column.
"""

import numpy as np


class WeightedLogisticRegression:
    def __init__(
        self,
        X,
        learning_rate=0.1,
        num_iters=10000,
        weight_positive=1.0,
        weight_negative=1.0,
        print_cost=False,
    ):
        self.lr = learning_rate
        self.num_iters = num_iters
        self.weight_positive = weight_positive
        self.weight_negative = weight_negative
        self.print_cost = print_cost

        # m for #training_examples, n for #features
        self.m, self.n = X.shape

    def train(self, X, y):
        y = np.array(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        ones = np.ones((X.shape[0], 1))
        X = np.append(ones, X, axis=1)

        # init weights
        self.weights = np.zeros((self.n + 1, 1))

        for it in range(self.num_iters + 1):
            # calculate hypothesis
            y_predict = 1 / (1 + np.exp(-np.dot(X, self.weights)))

            # weighted logistic regression gradient
            sample_weights = np.where(y == 1, self.weight_positive, self.weight_negative)
            error = y_predict - y
            weighted_error = sample_weights * error
            dw = 1 / self.m * np.dot(X.T, weighted_error)

            # gradient descent update step
            self.weights -= self.lr * dw

            if it % 1000 == 0 and self.print_cost:
                epsilon = 1e-15
                y_predict = np.clip(y_predict, epsilon, 1 - epsilon)
                losses = y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict)
                cost = -1 / self.m * np.sum(sample_weights * losses)
                print(f"Cost after iteration {it}: {cost}")

        return self.weights

    def predict(self, X):
        ones = np.ones((X.shape[0], 1))
        X = np.append(ones, X, axis=1)
        y_predict = 1 / (1 + np.exp(-np.dot(X, self.weights)))
        y_predict_labels = y_predict > 0.5

        return y_predict_labels


if __name__ == "__main__":
    np.random.seed(1)
    X_zeros = np.random.randn(950, 2) + np.array([-2, -2])
    X_ones = np.random.randn(50, 2) + np.array([2, 2])
    X = np.append(X_zeros, X_ones, axis=0)
    y = np.append(np.zeros((950, 1)), np.ones((50, 1)), axis=0)

    logreg = WeightedLogisticRegression(
        X,
        learning_rate=0.1,
        num_iters=5000,
        weight_positive=19.0,
        weight_negative=1.0,
        print_cost=True,
    )
    w = logreg.train(X, y)
    y_predict = logreg.predict(X)

    print(w)
    print(f"Accuracy: {np.sum(y==y_predict)/X.shape[0]}")
