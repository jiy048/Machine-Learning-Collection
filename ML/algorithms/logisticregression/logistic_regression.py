"""
From scratch implementation of Logistic Regression.

X has shape (m, n), y has shape (m, 1), and weights have
shape (n + 1, 1) after adding the bias column.

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-05-24 Initial coding

"""

import numpy as np


class LogisticRegression:
    def __init__(self, X, learning_rate=0.1, num_iters=10000):
        self.lr = learning_rate
        self.num_iters = num_iters

        # m for #training_examples, n for #features
        self.m, self.n = X.shape

    def add_bias_column(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.append(ones, X, axis=1)

    def train(self, X, y):
        X = self.add_bias_column(X)

        # init weights
        self.weights = np.zeros((self.n + 1, 1))

        for it in range(self.num_iters + 1):
            # calculate hypothesis
            y_predict = self.sigmoid(np.dot(X, self.weights))

            # calculate cost
            cost = (
                -1
                / self.m
                * np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))
            )

            # back prop / gradient calculations
            dw = 1 / self.m * np.dot(X.T, (y_predict - y))

            # gradient descent update step
            self.weights -= self.lr * dw

            # print cost sometimes
            if it % 1000 == 0:
                print(f"Cost after iteration {it}: {cost}")

        return self.weights

    def predict(self, X):
        X = self.add_bias_column(X)
        y_predict = self.sigmoid(np.dot(X, self.weights))
        y_predict_labels = y_predict > 0.5

        return y_predict_labels

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


if __name__ == "__main__":
    np.random.seed(1)
    X_zeros = np.random.randn(500, 2) + np.array([-2, -2])
    X_ones = np.random.randn(500, 2) + np.array([2, 2])
    X = np.append(X_zeros, X_ones, axis=0)
    y = np.append(np.zeros((500, 1)), np.ones((500, 1)), axis=0)

    logreg = LogisticRegression(X)
    w = logreg.train(X, y)
    y_predict = logreg.predict(X)

    print(f"Accuracy: {np.sum(y==y_predict)/X.shape[0]}")
