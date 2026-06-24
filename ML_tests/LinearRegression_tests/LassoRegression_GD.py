# Import folder where sorting algorithms
import sys
import unittest
import numpy as np

# For importing from different folders
# OBS: This is supposed to be done with automated testing,
# hence relative to folder we want to import from
sys.path.append("ML/algorithms/linearregression")
# If run from local:
# sys.path.append('../../ML/algorithms/linearregression')
from lasso_regression_gradient_descent import LassoRegression


class TestLassoRegression_GradientDescent(unittest.TestCase):
    def test_matches_linear_regression_when_alpha_is_zero(self):
        regression = LassoRegression(
            learning_rate=0.01,
            total_iterations=5000,
            alpha=0,
        )
        X = np.array([[0, 1, 2, 3, 4, 5]]).T
        y = np.array([[1, 2, 3, 4, 5, 6]]).T
        W_correct = np.array([[1, 1]]).T

        W = regression.main(X, y)
        boolean_array = np.isclose(W, W_correct, atol=0.1)

        self.assertTrue(boolean_array.all())

    def test_does_not_regularize_bias(self):
        regression = LassoRegression(
            learning_rate=0.01,
            total_iterations=3000,
            alpha=1,
        )
        X = np.zeros((10, 1))
        y = np.ones((10, 1)) * 5
        W_correct = np.array([[5, 0]]).T

        W = regression.main(X, y)
        boolean_array = np.isclose(W, W_correct, atol=0.1)

        self.assertTrue(boolean_array.all())


if __name__ == "__main__":
    print("Running Lasso Regression Gradient Descent tests:")
    unittest.main()
