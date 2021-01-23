from .base_svm import BaseSVM

import numpy as np


class CoordinateDescentSVC(BaseSVM):
    def __init__(self, C=1, beta=0.5, sigma=0.01, max_iter=50, callback=None):
        self.C = C
        self.beta = beta
        self.sigma = sigma
        self.max_iter = max_iter
        self.callback = callback
        self.ftol=1e-8

    def get_w(self, w0, X, y):
        w = w0
        prev_loss = self.loss(w0, X, y)
        for k in range(self.max_iter):
            for e in np.identity(X.shape[1]):
                l = 1
                H = 1 + 2 * self.C * np.sum((X @ e)**2)
                l_dashed = self.D_prime2(w, e, 0, X, y) / (H / 2 + self.sigma)
                while l > l_dashed:
                    l *= self.beta
                d = - self.D_prime(w, e, 0, X, y) / self.D_prime2(w, e, 0, X, y)
                z = l * d
                w = self.get_next_w(w, e, z)
            self.w = w
            self.coef_ = w.reshape(1, -1)
            self.callback(self)
            loss = self.loss(w, X, y)
            if (prev_loss - loss) / max(loss, prev_loss, 1) <= self.ftol:
                break
            prev_loss = loss
        return w

    def get_next_w(self, w, e, z):
        return w + e * z

    def get_b(self, w, X, y):
        return 1 - y * (X @ w.transpose())

    def D_prime(self, w, e, z, X, y):
        next_w = self.get_next_w(w, e, z)
        b = self.get_b(next_w, X, y)
        yxb = y * (X @ e) * b
        yxb[b <= 0] = 0
        return np.dot(w, e) + z - 2 * self.C * np.sum(yxb)

    def D_prime2(self, w, e, z, X, y):
        next_w = self.get_next_w(w, e, z)
        b = self.get_b(next_w, X, y)
        xx = X @ e
        xx[np.ravel(b) <= 0] = 0
        return 1 + 2 * self.C * np.sum(xx**2)
