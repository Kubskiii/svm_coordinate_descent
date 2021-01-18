from sklearn.linear_model._base import LinearClassifierMixin, SparseCoefMixin
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels

import numpy as np


class CoordinateDescentSVC(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):
    """
    TODO: Add docstrings
    """
    def __init__(self, C=1, beta=0.5, sigma=0.01, max_iter=100):
        """
        TODO: Add docstrings
        """
        self.C = C
        self.beta = beta
        self.sigma = sigma
        self.max_iter = max_iter

    def fit(self, X, y, sample_weight=None):
        """
        TODO: Add docstrings
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X = X
        self.y = 2 * y - 1

        self.w = np.zeros(X.shape[1])
        for k in range(self.max_iter):
            self.fit_iteration()
        self.coef_ = self.w.reshape(1, -1)
        self.intercept_ = 0
        # Return the classifier
        return self

    def fit_iteration(self):
        for e in np.identity(self.X.shape[1]):
            l = 1
            xx = np.matmul(self.X, e)
            H = 1 + 2 * self.C * np.sum(xx * xx)
            l_dashed = self.D_prime2(self.w, e, 0) / (H / 2 + self.sigma)
            while l > l_dashed:
                l *= self.beta
            d = - self.D_prime(self.w, e, 0) / self.D_prime2(self.w, e, 0)
            z = l * d
            self.w = self.get_next_w(self.w, e, z)

    def get_next_w(self, w, e, z):
        return w + e * z

    def get_b(self, w):
        return 1 - self.y * np.matmul(self.X, w.transpose())

    def D_prime(self, w, e, z):
        next_w = self.get_next_w(w, e, z)
        b = self.get_b(next_w)
        yxb = self.y * np.matmul(self.X, e) * b
        yxb[b <= 0] = 0
        return np.dot(w, e) + z - 2 * self.C * np.sum(yxb)

    def D_prime2(self, w, e, z):
        next_w = self.get_next_w(w, e, z)
        b = self.get_b(next_w)
        xx = np.matmul(self.X, e)
        xx[np.ravel(b) <= 0] = 0
        return 1 + 2 * self.C * np.sum(xx * xx)
