"""
some ordinal regression algorithms

This implements the margin-based ordinal regression methods described
in http://arxiv.org/abs/1408.2327
"""
import numpy as np
from scipy import optimize, linalg, stats

from sklearn import base, metrics, svm, linear_model

from joblib import Memory


def sigmoid(t):
    # sigmoid function, 1 / (1 + exp(-t))
    # stable computation
    idx = t > 0
    out = np.zeros_like(t)
    out[idx] = 1. / (1 + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)
    return out


def log_loss(Z):
    # stable computation of the logistic loss
    idx = Z > 0
    out = np.zeros_like(Z)
    out[idx] = np.log(1 + np.exp(-Z[idx]))
    out[~idx] = (-Z[~idx] + np.log(1 + np.exp(Z[~idx])))
    return out


def obj_margin(x0, X, y, alpha, n_class, weights):
    """
    Objective function for the general margin-based formulation
    """

    w = x0[:X.shape[1]]
    c = x0[X.shape[1]:]
    theta = np.cumsum(c)
    loss_fd = weights[y]

    Xw = X.dot(w)
    Alpha = theta[:, None] - Xw  # (n_class - 1, n_samples)
    S = np.sign(np.arange(n_class - 1)[:, None] - y + 0.5)

    obj = np.sum(loss_fd.T * log_loss(S * Alpha)) + \
           alpha * 0.5 * (linalg.norm(w) ** 2)
    return obj


def grad_margin(x0, X, y, alpha, n_class, weights):
    """
    Gradient for the general margin-based formulation
    """

    w = x0[:X.shape[1]]
    c = x0[X.shape[1]:]
    theta = np.cumsum(c)
    #theta = np.sort(theta)
    loss_fd = weights[y]

    Xw = X.dot(w)
    Alpha = theta[:, None] - Xw # (n_class - 1, n_samples)
    S = np.sign(np.arange(n_class - 1)[:, None] - y + 0.5)
    # Alpha[idx] *= -1
    # W[idx.T] *= -1

    Sigma = S * loss_fd.T * sigmoid(-S * Alpha)

    grad_w = X.T.dot(Sigma.sum(0)) / float(X.shape[0]) + alpha * 2 * w

    grad_theta = - Sigma.sum(1) / float(X.shape[0])

    tmp = np.concatenate(([0], grad_theta))
    grad_c = np.sum(grad_theta) - np.cumsum(tmp[:-1])

    return np.concatenate((grad_w, grad_c), axis=0)



def obj_multiclass(x0, X, y, alpha, n_class):
    n_samples, n_features = X.shape
    W = x0.reshape((n_features + 1, n_class-1))
    Wk = - W.sum(1)[:, None]
    W = np.concatenate((W, Wk), axis=1)
    X = np.concatenate((X, np.ones((n_samples, 1))), axis=1)
    Y = np.zeros((n_samples, n_class))
    Y[:] = - 1./(n_class - 1)
    for i in range(n_samples):
        Y[i, y[i]] = 1.

    L = np.abs(np.arange(n_class)[:, None] - np.arange(n_class))
    obj = (L[y] * np.fmax(X.dot(W) - Y, 0)).sum() / float(n_samples)

    Wt = W[:n_features]
    penalty = alpha * np.trace(Wt.T.dot(Wt))
    return obj + penalty




def threshold_fit(X, y, alpha, n_class, mode='AE', verbose=False,
                  maxiter=1000, bounds=False):
    """
    Solve the general threshold-based ordinal regression model
    using the logistic loss as surrogate of the 0-1 loss

    Parameters
    ----------
    mode : string, one of {'AE', '0-1'}

    """

    X = np.asarray(X)
    y = np.asarray(y) # XXX check its made of integers
    n_samples, n_features = X.shape

    if mode == 'AE':
        # loss forward difference
        loss_fd = np.ones((n_class, n_class - 1))
    elif mode == '0-1':
        loss_fd = np.diag(np.ones(n_class - 1)) + \
            np.diag(np.ones(n_class - 2), k=-1)
        loss_fd = np.vstack((loss_fd, np.zeros(n_class -1)))
        loss_fd[-1, -1] = 1
    else:
        raise NotImplementedError

    x0 = np.zeros(n_features + n_class - 1)
    x0[X.shape[1]:] = np.arange(n_class - 1)
    options = {'maxiter' : maxiter}
    sol = optimize.minimize(obj_margin, x0, #jac=grad_margin,
        args=(X, y, alpha, n_class, loss_fd),
        options=options)
    if not sol.success:
        print(sol.message)
    w, c = sol.x[:X.shape[1]], sol.x[X.shape[1]:]
    theta = np.cumsum(c)
    return w, np.sort(theta)


def threshold_predict(X, w, theta):
    """
    Class numbers are assumed to be between 0 and k-1
    """
    idx = np.concatenate((np.argsort(theta), [theta.size]))
    pred = []
    n_samples = X.shape[0]
    Xw = X.dot(w)
    tmp = Xw - theta[:, None]
    pred = np.sum(tmp >= 0, axis=0).astype(np.int)
    return pred


def multiclass_fit(X, y, alpha, n_class, maxiter=100000):
    """
    Multiclass classification with absolute error cost

    Lee, Yoonkyung, Yi Lin, and Grace Wahba. "Multicategory support
    vector machines: Theory and application to the classification of
    microarray data and satellite radiance data." Journal of the
    American Statistical Association 99.465 (2004): 67-81.
    """
    X = np.asarray(X)
    y = np.asarray(y) # XXX check its made of integers
    n_samples, n_features = X.shape

    x0 = np.random.randn((n_features + 1) * (n_class - 1))
    options = {'maxiter' : maxiter}
    sol = optimize.minimize(obj_multiclass, x0, jac=False,
        args=(X, y, alpha, n_class), method='L-BFGS-B',
        options=options)
    if not sol.success:
        print(sol.message)
    W = sol.x.reshape((n_features + 1, n_class-1))
    Wk = - W.sum(1)[:, None]
    W = np.concatenate((W, Wk), axis=1)
    return W

def multiclass_predict(X, W):
    n_samples, n_features = X.shape
    X = np.concatenate((X, np.ones((n_samples, 1))), axis=1)
    XW = X.dot(W)
    return np.argmax(XW, axis=1)


class OrdinalLogistic(base.BaseEstimator):
    """
    Classifier that implements the ordinal logistic model.

    Parameters
    ----------
    alpha: float
        Regularization parameter. Zero is no regularization, higher values
        increate the squared l2 regularization.

    mode: string
        mode='AE' is equivalent to the 'All threshold' formulation
        in the reference while mode='0-1' is equivalent to the 'Immediate
        threshold' formulation in the reference.

    References
    ----------
    J. D. M. Rennie and N. Srebro, "Loss Functions for Preference Levels :
    Regression with Discrete Ordered Labels," in Proceedings of the IJCAI
    Multidisciplinary Workshop on Advances in Preference Handling, 2005.
    """
    def __init__(self, alpha=1., verbose=0, maxiter=10000):
        self.alpha = alpha
        self.verbose = verbose
        self.maxiter = maxiter

    def fit(self, X, y):
        self.n_class = np.unique(y).size
        self.coef_, self.theta_ = threshold_fit(X, y, self.alpha, self.n_class,
            mode='AE', verbose=self.verbose)
        return self

    def predict(self, X):
        return threshold_predict(X, self.coef_, self.theta_)

    def score(self, X, y):
        pred = self.predict(X)
        return - metrics.mean_absolute_error(pred, y)


class MulticlassLogistic(base.BaseEstimator):
    def __init__(self, alpha=1., verbose=0, maxiter=10000):
        self.alpha = alpha
        self.verbose = verbose
        self.maxiter = maxiter

    def fit(self, X, y):
        self.n_class = np.unique(y).size
        self.coef_, self.theta_ = threshold_fit(X, y, self.alpha, self.n_class,
            mode='0-1', verbose=self.verbose)
        return self

    def predict(self, X):
        return threshold_predict(X, self.coef_, self.theta_)

    def score(self, X, y):
        pred = self.predict(X)
        return metrics.accuracy_score(pred, y)


class RidgeOR(linear_model.Ridge):
    """
    Overwrite Ridge from scikit-learn to use
    the (minus) absolute error as score function.

    (see https://github.com/scikit-learn/scikit-learn/issues/3848
    on why this cannot be accomplished using a GridSearchCV object)
    """

    def fit(self, X, y):
        self.unique_y_ = np.unique(y)
        super(linear_model.Ridge, self).fit(X, y)
        return self

    def predict(self, X):
        pred =  np.round(super(linear_model.Ridge, self).predict(X))
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





if __name__ == '__main__':

    np.random.seed(0)
    from sklearn import datasets, metrics, svm, cross_validation
    n_class = 3
    n_samples = 200
    n_dim = 10

    X, y = datasets.make_regression(n_samples=n_samples, n_features=n_dim,
        n_informative=n_dim // 10, noise=20.)

    bins = stats.mstats.mquantiles(y, np.linspace(0, 1, n_class + 1))
    y = np.digitize(y, bins[:-1])
    y -= np.min(y)

    cv = cross_validation.KFold(y.size)

    for clf in [OrdinalLogistic(), RidgeOR(), LAD()]:
        print np.mean(cross_validation.cross_val_score(clf, X, y, cv=cv))


