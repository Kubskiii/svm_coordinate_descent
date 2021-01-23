from .base_svm import BaseSVM
from scipy.optimize import minimize


class lbfgsbSVM(BaseSVM):
    def __init__(self, C=1, callback=None):
        self.C = C
        self.callback = callback

    def get_w(self, w0, X, y):
        minimize(
            fun=self.loss,
            x0=w0,
            method='L-BFGS-B',
            jac=self.loss_prime,
            args=(X, y),
            callback=self.iteration_callback,
            options=dict(gtol=float("-inf"))
        )

    def iteration_callback(self, w):
        self.w = w
        self.coef_ = w.reshape(1, -1)
        self.callback(self)
