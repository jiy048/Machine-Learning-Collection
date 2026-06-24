"""
Implementation of Ridge Regression using Gradient Descent.

Let m = #training examples and n = #number of features. It takes
as input the following: y is R^(m x 1), X is R^(m x n),
w is R^(n x 1).

The optimized cost is mean squared error plus an L2 penalty on
feature weights. The bias term is not regularized.
"""

import numpy as np


class RidgeRegression:
    def __init__(self, learning_rate=0.01, total_iterations=1000, alpha=0.1, print_cost=False):
        self.learning_rate = learning_rate
        self.total_iterations = total_iterations
        self.alpha = alpha
        self.print_cost = print_cost

    def y_hat(self, X, w):
        return np.dot(X, w)

    def cost(self, yhat, y, w):
        mse = 1 / self.m * np.sum(np.power(yhat - y, 2))
        #Ridge regularization 通常只惩罚 feature weights，不惩罚 bias
        l2_penalty = self.alpha * np.sum(np.power(w[1:], 2))

        return mse + l2_penalty

    def gradient_descent(self, w, X, y, yhat):
        dCdW = 2 / self.m * np.dot(X.T, yhat - y)

        l2_gradient = 2 * self.alpha * w
        l2_gradient[0] = 0

        w = w - self.learning_rate * (dCdW + l2_gradient)

        return w

    def main(self, X, y):
        # Add x1 = 1
        ones = np.ones((X.shape[0], 1))
        X = np.append(ones, X, axis=1)

        self.m = X.shape[0]
        self.n = X.shape[1]

        w = np.zeros((self.n, 1))

        for it in range(self.total_iterations + 1):
            yhat = self.y_hat(X, w)
            cost = self.cost(yhat, y, w)

            if it % 2000 == 0 and self.print_cost:
                print(f"Cost at iteration {it} is {cost}")

            w = self.gradient_descent(w, X, y, yhat)

        return w


if __name__ == "__main__":
    X = np.random.rand(500, 3)
    y = 5 + 3 * X[:, [0]] - 2 * X[:, [1]] + np.random.randn(500, 1) * 0.1
    regression = RidgeRegression(learning_rate=0.01, total_iterations=10000, alpha=0.01)
    w = regression.main(X, y)
    print(w)
