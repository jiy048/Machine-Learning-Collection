# Import folder where logistic regression algorithms live
import sys
import unittest
import numpy as np

# For importing from different folders
# OBS: This is supposed to be done with automated testing,
# hence relative to folder we want to import from
sys.path.append("ML/algorithms/logisticregression")
# If run from local:
# sys.path.append('../../ML/algorithms/logisticregression')
from weighted_logistic_regression import WeightedLogisticRegression


class TestWeightedLogisticRegression(unittest.TestCase):
    def test_learns_linearly_separable_data(self):
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([[0], [0], [1], [1]])
        regression = WeightedLogisticRegression(
            X,
            learning_rate=0.1,
            num_iters=2000,
            weight_positive=2.0,
            weight_negative=1.0,
        )

        W = regression.train(X, y)
        y_predict = regression.predict(X)

        self.assertEqual(W.shape, (2, 1))
        self.assertTrue((y_predict == y).all())

    def test_weighted_gradient_matches_manual_formula(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[0], [1], [1]])
        regression = WeightedLogisticRegression(
            X,
            learning_rate=0.1,
            num_iters=0,
            weight_positive=3.0,
            weight_negative=0.5,
        )
        ones = np.ones((X.shape[0], 1))
        X_bias = np.append(ones, X, axis=1)
        weights = np.array([[0.2], [0.4]])
        y_predict = 1 / (1 + np.exp(-np.dot(X_bias, weights)))

        sample_weights = np.where(
            y == 1,
            regression.weight_positive,
            regression.weight_negative,
        )
        error = y_predict - y
        weighted_error = sample_weights * error
        gradient = 1 / X.shape[0] * np.dot(X_bias.T, weighted_error)

        manual_sample_weights = np.array([[0.5], [3.0], [3.0]])
        manual_gradient = (
            1
            / X.shape[0]
            * np.dot(X_bias.T, manual_sample_weights * (y_predict - y))
        )

        np.testing.assert_array_equal(sample_weights, manual_sample_weights)
        np.testing.assert_allclose(gradient, manual_gradient)
        self.assertEqual(gradient.shape, (2, 1))


if __name__ == "__main__":
    print("Running Weighted Logistic Regression tests:")
    unittest.main()
