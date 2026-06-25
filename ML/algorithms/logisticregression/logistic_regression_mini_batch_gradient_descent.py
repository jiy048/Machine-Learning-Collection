"""
From scratch implementation of Logistic Regression using Mini-Batch Gradient Descent.

X has shape (m, n), y has shape (m, 1), and weights have
shape (n + 1, 1) after adding the bias column.
"""

import numpy as np


class LogisticRegressionMiniBatch:
    def __init__(
        self,
        X,
        learning_rate=0.1,
        num_epochs=1000,
        num_batches=32,
        print_cost=False,
        random_state=None,
    ):
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.print_cost = print_cost
        self.random_state = random_state

        # m for #training_examples, n for #features
        self.m, self.n = X.shape

    def add_bias_column(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.append(ones, X, axis=1)

    def create_batches(self, X, y, rng):
        # rng keeps its internal state between calls, so each epoch gets a new
        # shuffled order while still being reproducible from random_state.
        indices = rng.permutation(X.shape[0])
        num_batches = min(self.num_batches, X.shape[0])
        batches = []

        for batch_indices in np.array_split(indices, num_batches):
            batches.append((X[batch_indices], y[batch_indices]))

        return batches

    def train(self, X, y):
        X = self.add_bias_column(X)

        # init weights
        self.weights = np.zeros((self.n + 1, 1))
        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.num_epochs + 1):
            for X_batch, y_batch in self.create_batches(X, y, rng):
                batch_size = X_batch.shape[0]

                # calculate hypothesis
                y_predict = self.sigmoid(np.dot(X_batch, self.weights))

                # back prop / gradient calculations
                dw = 1 / batch_size * np.dot(X_batch.T, (y_predict - y_batch))

                # gradient descent update step
                self.weights -= self.lr * dw

            if epoch % 100 == 0 and self.print_cost:
                y_predict = self.sigmoid(np.dot(X, self.weights))
                cost = self.cost(y_predict, y)
                print(f"Cost after epoch {epoch}: {cost}")

        return self.weights

    def predict(self, X):
        X = self.add_bias_column(X)
        y_predict = self.sigmoid(np.dot(X, self.weights))
        y_predict_labels = y_predict > 0.5

        return y_predict_labels

    def cost(self, y_predict, y):
        epsilon = 1e-15
        y_predict = np.clip(y_predict, epsilon, 1 - epsilon)
        return (
            -1
            / self.m
            * np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))
        )

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


if __name__ == "__main__":
    np.random.seed(1)
    X_zeros = np.random.randn(500, 2) + np.array([-2, -2])
    X_ones = np.random.randn(500, 2) + np.array([2, 2])
    X = np.append(X_zeros, X_ones, axis=0)
    y = np.append(np.zeros((500, 1)), np.ones((500, 1)), axis=0)

    logreg = LogisticRegressionMiniBatch(
        X,
        learning_rate=0.1,
        num_epochs=500,
        num_batches=32,
        random_state=1,
    )
    w = logreg.train(X, y)
    y_predict = logreg.predict(X)

    print(f"Accuracy: {np.sum(y==y_predict)/X.shape[0]}")
