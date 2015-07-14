import numpy as np
from scipy import stats, optimize, linalg
import mord

np.random.seed(0)
from sklearn import datasets, metrics, svm, cross_validation
n_class = 5
n_samples = 200
n_dim = 2

X, y = datasets.make_regression(n_samples=n_samples, n_features=n_dim,
    n_informative=n_dim // 10, noise=20.)
bins = stats.mstats.mquantiles(y, np.linspace(0, 1, n_class + 1))
y = np.digitize(y, bins[:-1])
y -= y.min()

# def test_1():
#     """
#     Test two model in overfit mode
#     """
#     clf = mord.LogisticAT()
#     clf.fit(X, y)
#     # the score is - absolute error, 0 is perfect
#     assert clf.score(X, y) == 0.
#
#     #clf = mord.MulticlassLogistic()
#     #clf.fit(X, y)
#     ## the score is accuracy, 1 is perfect
#     #assert clf.score(X, y) == 1.


def test_grad():
    x0 = np.random.randn(n_dim + n_class - 1)
    x0[n_dim+1:] = np.abs(x0[n_dim+1:])

    loss_fd = np.diag(np.ones(n_class - 1)) + \
        np.diag(np.ones(n_class - 2), k=-1)
    loss_fd = np.vstack((loss_fd, np.zeros(n_class -1)))
    loss_fd[-1, -1] = 1  # border case

    L = np.eye(n_class - 1) - np.diag(np.ones(n_class - 2), k=-1)


    fun = lambda x: mord.threshold_based.obj_margin(
        x, X, y, 1.0, n_class, loss_fd, L)
    grad = lambda x: mord.threshold_based.grad_margin(
        x, X, y, 1.0, n_class, loss_fd, L)
    assert optimize.check_grad(fun, grad, x0) < 1e-3

test_grad()