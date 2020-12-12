from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels


class CoordinateDescentSVC(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):
    """
    TODO: Add docstrings
    """
    def fit(self, X, y, sample_weight=None):
        """
        TODO: Add docstrings
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self
