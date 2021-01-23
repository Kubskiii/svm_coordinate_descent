from .base_svm import BaseSVM
from scipy.optimize import minimize


class lbfgsbSVM(BaseSVM):
    def __init__(self, C=1):
        self.C = C

    def get_w(self, w0, X, y):
        return minimize(
            fun=self.loss,
            x0=w0,
            method='L-BFGS-B',
            jac=self.loss_prime,
            args=(X, y),
        ).x
