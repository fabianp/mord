import numpy as np
from sklearn import linear_model, svm, metrics


class OrdinalRidge(linear_model.Ridge):
    """
    Overwrite Ridge from scikit-learn to use
    the (minus) absolute error as score function.

    (see https://github.com/scikit-learn/scikit-learn/issues/3848
    on why this cannot be accomplished using a GridSearchCV object)
    """

    def fit(self, X, y, **fit_params):
        self.unique_y_ = np.unique(y)
        super(linear_model.Ridge, self).fit(X, y, **fit_params)
        return self

    def predict(self, X):
        pred = np.round(super(linear_model.Ridge, self).predict(X))
        pred = np.clip(pred, 0, self.unique_y_.max())
        return pred

    def score(self, X, y):
        pred = self.predict(X)
        return - metrics.mean_squared_error(pred, y)


if hasattr(svm, 'LinearSVR'):
    class LAD(svm.LinearSVR):
        """
        Least Absolute Deviation
        """

        def fit(self, X, y):
            self.epsilon = 0.
            self.unique_y_ = np.unique(y)
            svm.LinearSVR.fit(self, X, y)
            return self

        def predict(self, X):
            pred = np.round(super(svm.LinearSVR, self).predict(X))
            pred = np.clip(pred, 0, self.unique_y_.max())
            return pred

        def score(self, X, y):
            pred = self.predict(X)
            return - metrics.mean_absolute_error(pred, y)

