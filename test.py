import numpy as np
from scipy import stats
import mord

np.random.seed(0)
from sklearn import datasets, metrics, svm, cross_validation
n_class = 3
n_samples = 200
n_dim = 100

X, y = datasets.make_regression(n_samples=n_samples, n_features=n_dim,
    n_informative=n_dim // 10, noise=20.)
bins = stats.mstats.mquantiles(y, np.linspace(0, 1, n_class + 1))
y = np.digitize(y, bins[:-1])

def test_1():
    """
    Test two model sin overfit mode
    """
    clf = mord.LogisticAT()
    clf.fit(X, y)
    # the score is - absolute error, 0 is perfect
    assert clf.score(X, y) == 0.

    #clf = mord.MulticlassLogistic()
    #clf.fit(X, y)
    ## the score is accuracy, 1 is perfect
    #assert clf.score(X, y) == 1.


test_1()