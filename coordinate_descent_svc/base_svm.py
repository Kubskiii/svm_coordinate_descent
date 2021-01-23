from sklearn.linear_model._base import LinearClassifierMixin, SparseCoefMixin
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from abc import abstractmethod

import numpy as np


class BaseSVM(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.classes_ = unique_labels(y)
        self.w = self.get_w(np.zeros(X.shape[1]), X, 2 * y - 1)
        self.coef_ = self.w.reshape(1, -1)
        self.intercept_ = 0
        return self

    @abstractmethod
    def get_w(self, w0, X, y):
        ...

    def loss(self, w, X, y):
        return 1/2 * w @ w + self.C * np.sum(np.maximum(1 - y * (w @ X.T), 0)**2, axis=0)

    def loss_prime(self, w, X, y):
        return w - 2 * self.C * X.T @ (y * np.maximum(1 - y * (w @ X.T), 0))
