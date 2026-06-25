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
from logistic_regression_mini_batch_gradient_descent import LogisticRegressionMiniBatch


class TestLogisticRegression_MiniBatchGradientDescent(unittest.TestCase):
    def test_learns_linearly_separable_data(self):
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([[0], [0], [1], [1]])
        regression = LogisticRegressionMiniBatch(
            X,
            learning_rate=0.1,
            num_epochs=2000,
            num_batches=2,
            random_state=1,
        )

        W = regression.train(X, y)
        y_predict = regression.predict(X)

        self.assertEqual(W.shape, (2, 1))
        self.assertTrue((y_predict == y).all())


if __name__ == "__main__":
    print("Running Logistic Regression Mini-Batch Gradient Descent tests:")
    unittest.main()
