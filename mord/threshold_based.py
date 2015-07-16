"""
some ordinal regression algorithms

This implements the margin-based ordinal regression methods described
in http://arxiv.org/abs/1408.2327
"""
import numpy as np
from scipy import optimize, linalg, stats

from sklearn import base, metrics

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


def obj_margin(x0, X, y, alpha, n_class, weights, L):
    """
    Objective function for the general margin-based formulation
    """

    w = x0[:X.shape[1]]
    c = x0[X.shape[1]:]
    theta = L.dot(c)
    loss_fd = weights[y]

    Xw = X.dot(w)
    Alpha = theta[:, None] - Xw  # (n_class - 1, n_samples)
    S = np.sign(np.arange(n_class - 1)[:, None] - y + 0.5)

    obj = np.sum(loss_fd.T * log_loss(S * Alpha)) + \
        alpha * 0.5 * (np.dot(w, w))
    return obj


def grad_margin(x0, X, y, alpha, n_class, weights, L):
    """
    Gradient for the general margin-based formulation
    """

    w = x0[:X.shape[1]]
    c = x0[X.shape[1]:]
    theta = L.dot(c)
    loss_fd = weights[y]

    Xw = X.dot(w)
    Alpha = theta[:, None] - Xw  # (n_class - 1, n_samples)
    S = np.sign(np.arange(n_class - 1)[:, None] - y + 0.5)
    # Alpha[idx] *= -1
    # W[idx.T] *= -1

    Sigma = S * loss_fd.T * sigmoid(-S * Alpha)

    grad_w = X.T.dot(Sigma.sum(0)) + alpha * w

    grad_theta = -Sigma.sum(1)
    grad_c = L.T.dot(grad_theta)
    return np.concatenate((grad_w, grad_c), axis=0)


def threshold_fit(X, y, alpha, n_class, mode='AE',
                  maxiter=1000, verbose=False, tol=1e-4):
    """
    Solve the general threshold-based ordinal regression model
    using the logistic loss as surrogate of the 0-1 loss

    Parameters
    ----------
    mode : string, one of {'AE', '0-1'}

    """

    X = np.asarray(X)
    y = np.asarray(y)  # XXX check its made of integers
    n_samples, n_features = X.shape

    # convert from c to theta
    L = np.zeros((n_class - 1, n_class - 1))
    L[np.tril_indices(n_class-1)] = 1.

    if mode == 'AE':
        # loss forward difference
        loss_fd = np.ones((n_class, n_class - 1))
    elif mode == '0-1':
        loss_fd = np.diag(np.ones(n_class - 1)) + \
            np.diag(np.ones(n_class - 2), k=-1)
        loss_fd = np.vstack((loss_fd, np.zeros(n_class -1)))
        loss_fd[-1, -1] = 1  # border case
    elif mode == 'SE':
        a = np.arange(n_class-1)
        b = np.arange(n_class)
        loss_fd = np.abs((a - b[:, None])**2 - (a - b[:, None]+1)**2)
    else:
        raise NotImplementedError

    x0 = np.zeros(n_features + n_class - 1)
    x0[X.shape[1]:] = np.arange(n_class - 1)
    options = {'maxiter' : maxiter, 'disp': verbose}
    if n_class > 2:
        bounds = [(None, None)] * (n_features + 1) + \
                 [(0, None)] * (n_class - 3) + [(1, 1)]
    else:
        bounds = None
    sol = optimize.minimize(obj_margin, x0, method='L-BFGS-B',
        jac=grad_margin, args=(X, y, alpha, n_class, loss_fd, L),
        bounds=bounds, options=options, tol=tol)
    if not sol.success:
        print(sol.message)
    w, c = sol.x[:X.shape[1]], sol.x[X.shape[1]:]
    theta = L.dot(c)
    return w, theta


def threshold_predict(X, w, theta):
    """
    Class numbers are assumed to be between 0 and k-1
    """
    tmp = theta[:, None] - X.dot(w)
    pred = np.sum(tmp < 0, axis=0).astype(np.int)
    return pred



class LogisticAT(base.BaseEstimator):
    """
    Classifier that implements the ordinal logistic model (All-Threshold variant)

    Parameters
    ----------
    alpha: float
        Regularization parameter. Zero is no regularization, higher values
        increate the squared l2 regularization.

    References
    ----------
    J. D. M. Rennie and N. Srebro, "Loss Functions for Preference Levels :
    Regression with Discrete Ordered Labels," in Proceedings of the IJCAI
    Multidisciplinary Workshop on Advances in Preference Handling, 2005.
    """
    def __init__(self, alpha=1., verbose=0, maxiter=1000):
        self.alpha = alpha
        self.verbose = verbose
        self.maxiter = maxiter

    def fit(self, X, y):
        _y = np.array(y).astype(np.int)
        if np.abs(_y - y).sum() > 0.1:
            raise ValueError('y must only contain integer values')
        self.classes_ = np.unique(y)
        self.n_class_ = self.classes_.max() - self.classes_.min() + 1
        y_tmp = y - y.min() # we need classes that start at zero
        self.coef_, self.theta_ = threshold_fit(X, y_tmp, self.alpha, self.n_class_,
                                                mode='AE', verbose=self.verbose)
        return self

    def predict(self, X):
        return threshold_predict(X, self.coef_, self.theta_) + self.classes_.min()

    def score(self, X, y):
        pred = self.predict(X)
        return - metrics.mean_absolute_error(pred, y)



class LogisticIT(base.BaseEstimator):
    """
    Classifier that implements the ordinal logistic model
    (Immediate-Threshold variant).

    Contrary to the OrdinalLogistic model, this variant
    minimizes a convex surrogate of the 0-1 loss, hence
    the score associated with this object is the accuracy
    score, i.e. the same score used in multiclass
    classification methods (sklearn.metrics.accuracy_score).

    Parameters
    ----------
    alpha: float
        Regularization parameter. Zero is no regularization, higher values
        increate the squared l2 regularization.

    References
    ----------
    J. D. M. Rennie and N. Srebro, "Loss Functions for Preference Levels :
    Regression with Discrete Ordered Labels," in Proceedings of the IJCAI
    Multidisciplinary Workshop on Advances in Preference Handling, 2005.
    """
    def __init__(self, alpha=1., verbose=0, maxiter=1000):
        self.alpha = alpha
        self.verbose = verbose
        self.maxiter = maxiter

    def fit(self, X, y):
        _y = np.array(y).astype(np.int)
        if np.abs(_y - y).sum() > 0.1:
            raise ValueError('y must only contain integer values')
        self.classes_ = np.unique(y)
        self.n_class_ = self.classes_.max() - self.classes_.min() + 1
        y_tmp = y - y.min() # we need classes that start at zero
        self.coef_, self.theta_ = threshold_fit(
            X, y_tmp, self.alpha, self.n_class_,
            mode='0-1', verbose=self.verbose, maxiter=self.maxiter)
        return self

    def predict(self, X):
        return threshold_predict(X, self.coef_, self.theta_) + self.classes_.min()

    def score(self, X, y):
        pred = self.predict(X)
        return metrics.accuracy_score(pred, y)


class LogisticSE(base.BaseEstimator):
    """
    Classifier that implements the ordinal logistic model
    (Immediate-Threshold variant).

    Contrary to the OrdinalLogistic model, this variant
    minimizes a convex surrogate of the 0-1 loss, hence
    the score associated with this object is the accuracy
    score, i.e. the same score used in multiclass
    classification methods (sklearn.metrics.accuracy_score).

    Parameters
    ----------
    alpha: float
        Regularization parameter. Zero is no regularization, higher values
        increate the squared l2 regularization.

    References
    ----------
    J. D. M. Rennie and N. Srebro, "Loss Functions for Preference Levels :
    Regression with Discrete Ordered Labels," in Proceedings of the IJCAI
    Multidisciplinary Workshop on Advances in Preference Handling, 2005.
    """
    def __init__(self, alpha=1., verbose=0, maxiter=1000):
        self.alpha = alpha
        self.verbose = verbose
        self.maxiter = maxiter

    def fit(self, X, y):
        _y = np.array(y).astype(np.int)
        if np.abs(_y - y).sum() > 1e-3:
            raise ValueError('y must only contain integer values')
        self.classes_ = np.unique(y)
        self.n_class_ = self.classes_.max() - self.classes_.min() + 1
        y_tmp = y - y.min()  # we need classes that start at zero
        self.coef_, self.theta_ = threshold_fit(
            X, y_tmp, self.alpha, self.n_class_,
            mode='SE', verbose=self.verbose, maxiter=self.maxiter)
        return self

    def predict(self, X):
        return threshold_predict(X, self.coef_, self.theta_) + self.classes_.min()

    def score(self, X, y):
        pred = self.predict(X)
        return - metrics.mean_squared_error(pred, y)





if __name__ == '__main__':

    np.random.seed(0)
    from sklearn import datasets, metrics, svm, cross_validation
    n_class = 3
    n_samples = 200
    n_dim = 100

    X, y = datasets.make_regression(n_samples=n_samples, n_features=n_dim,
                                    n_informative=n_dim // 10, noise=20.)

    bins = stats.mstats.mquantiles(y, np.linspace(0, 1, n_class + 1))
    y = np.digitize(y, bins[:-1])
    y -= np.min(y)

    # test that the gradient is correct
    x0 = np.random.randn(X.shape[1] + n_class - 1)

    loss_fd = np.diag(np.ones(n_class - 1)) + \
        np.diag(np.ones(n_class - 2), k=-1)
    loss_fd = np.vstack((loss_fd, np.zeros(n_class -1)))
    loss_fd[-1, -1] = 1
    print(optimize.check_grad(obj_margin, grad_margin, x0, X, y, 1., n_class,
                              loss_fd))

    cv = cross_validation.KFold(y.size)

    for clf in [LogisticAT(), OrdinalRidge(), LAD()]:
        print(np.mean(cross_validation.cross_val_score(clf, X, y, cv=cv)))

    print()
    for clf in [LogisticAT(mode='0-1'), svm.SVC()]:
        print(np.mean(cross_validation.cross_val_score(clf, X, y, cv=cv)))
