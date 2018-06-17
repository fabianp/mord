import numpy as np
from scipy import stats, optimize, sparse
import mord
import functools
from nose.tools import assert_almost_equal, assert_greater_equal, assert_less

np.random.seed(0)
from sklearn import datasets, metrics, linear_model

n_class = 5
n_samples = 100
n_dim = 10

X = np.random.randn(n_samples, n_dim)
w = np.random.randn(n_dim)
y = X.dot(w)
bins = stats.mstats.mquantiles(y, np.linspace(0, 1, n_class + 1))
y = np.digitize(y, bins[:-1])
y -= y.min()


# import pylab as plt
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

def test_1():
    """
    Test two model in overfit mode
    """
    clf1 = mord.OrdinalRidge(alpha=0.)
    clf1.fit(X, y)

    clf2 = mord.LogisticAT(alpha=0.)
    clf2.fit(X, y)

    # the score is - absolute error, 0 is perfect
    # assert clf1.score(X, y) < clf2.score(X, y)

    clf3 = mord.LogisticSE(alpha=0.)
    clf3.fit(X, y)
    pred3 = clf3.predict(X)
    pred2 = clf2.predict(X)

    # check that it predicts better than the surrogate
    # for other loss
    assert np.abs(pred2 - y).mean() <= np.abs(pred3 - y).mean()
    # # the score is - absolute error, 0 is perfect
    # assert_almost_equal(clf.score(X, y), 0., places=2)
    #
    # clf = mord.LogisticIT(alpha=0.)
    # clf.fit(X, y)
    # # the score is classification error, 1 is perfect
    # assert_almost_equal(clf.score(X, y), 1., places=2)

    # test on sparse matrices
    X_sparse = sparse.csr_matrix(X)
    clf4 = mord.LogisticAT(alpha=0.)
    clf4.fit(X_sparse, y)
    pred4 = clf4.predict(X_sparse)
    assert metrics.mean_absolute_error(y, pred4) < 1.


def test_grad():
    x0 = np.random.randn(n_dim + n_class - 1)
    x0[n_dim + 1:] = np.abs(x0[n_dim + 1:])

    loss_fd = np.diag(np.ones(n_class - 1)) + \
              np.diag(np.ones(n_class - 2), k=-1)
    loss_fd = np.vstack((loss_fd, np.zeros(n_class - 1)))
    loss_fd[-1, -1] = 1  # border case

    L = np.eye(n_class - 1) - np.diag(np.ones(n_class - 2), k=-1)

    def fun(x, sample_weights=None):
        return mord.threshold_based.obj_margin(
            x, X, y, 100.0, n_class, loss_fd, L, sample_weights)

    def grad(x, sample_weights=None):
        return mord.threshold_based.grad_margin(
            x, X, y, 100.0, n_class, loss_fd, L, sample_weights)

    assert_less(
        optimize.check_grad(fun, grad, x0),
        1e-4,
        msg='unweighted')

    sample_weights = np.random.rand(n_samples)
    assert_less(
        optimize.check_grad(
            functools.partial(fun, sample_weights=sample_weights),
            functools.partial(grad, sample_weights=sample_weights),
            x0),
        1e-4,
        msg='weighted')


def test_binary_class():
    Xc, yc = datasets.make_classification(n_classes=2, n_samples=1000)
    clf = linear_model.LogisticRegression(C=1e6)
    clf.fit(Xc[:500], yc[:500])
    pred_lr = clf.predict(Xc[500:])

    clf = mord.LogisticAT(alpha=1e-6)
    clf.fit(Xc[:500], yc[:500])
    pred_at = clf.predict(Xc[500:])
    assert_almost_equal(np.abs(pred_lr - pred_at).mean(), 0.)

    clf2 = mord.LogisticSE(alpha=1e-6)
    clf2.fit(Xc[:500], yc[:500])
    pred_at = clf2.predict(Xc[500:])
    assert_almost_equal(np.abs(pred_lr - pred_at).mean(), 0.)

# def test_performance():
#     clf1 = mord.LogisticAT()
#     clf1.fit(X, y)
#     assert_almost_equal(clf1.score(X, y) < )


def test_predict_proba_nonnegative():
    """
    Test that predict_proba() function outputs a tuple of non-negative values
    """

    def check_for_negative_prob(proba):
        for p in np.ravel(proba):
            assert_greater_equal(np.round(p,7), 0)

    clf = mord.LogisticAT(alpha=0.)
    clf.fit(X, y)
    check_for_negative_prob(clf.predict_proba(X))

    clf2 = mord.LogisticIT(alpha=0.)
    clf2.fit(X, y)
    check_for_negative_prob(clf2.predict_proba(X))

    clf3 = mord.LogisticSE(alpha=0.)
    clf3.fit(X, y)
    check_for_negative_prob(clf3.predict_proba(X))
