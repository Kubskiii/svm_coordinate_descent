from .base_svm import BaseSVM
from scipy.optimize import minimize


class lbfgsbSVM(BaseSVM):
    def __init__(self, C=1, ftol=1e-8, callback=None):
        self.C = C
        self.callback = callback
        self.ftol = ftol

    def get_w(self, w0, X, y):
        minimize(
            fun=self.loss,
            x0=w0,
            method='L-BFGS-B',
            jac=self.loss_prime,
            args=(X, y),
            callback=self.iteration_callback,
            options=dict(gtol=float("-inf"), ftol=self.ftol)
        )

    def iteration_callback(self, w):
        self.w = w
        self.coef_ = w.reshape(1, -1)
        if self.callback:
            self.callback(self)
