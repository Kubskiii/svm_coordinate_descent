from sklearn.linear_model._base import LinearClassifierMixin, SparseCoefMixin
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from scipy.optimize import minimize

import numpy as np


class lbfgsbSVM(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):
    """
    TODO: Add docstrings
    """
    def __init__(self, C=1):
        """
        TODO: Add docstrings
        """
        self.C = C

    def fit(self, X, y, sample_weight=None):
        """
        TODO: Add docstrings
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, accept_sparse=True)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X = X
        self.y = 2 * y - 1

        self.w = np.zeros(X.shape[1])

        self.w = minimize(
            fun=self.func,
            x0=self.w,
            method='L-BFGS-B',
            jac=self.func_prime
        ).x

        self.coef_ = self.w.reshape(1, -1)
        self.intercept_ = 0

        # Return the classifier
        return self

    def func(self, w):
        return 1/2 * w @ w + self.C * np.sum(np.maximum(1 - self.y * (w @ self.X.T), 0)**2, axis=0)

    def func_prime(self, w):
        return w - 2 * self.C * self.X.T @ (self.y * np.maximum(1 - self.y * (w @ self.X.T), 0))

