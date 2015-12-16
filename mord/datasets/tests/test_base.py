from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_equal

from mord.datasets.base import load_housing


def test_load_housing():
    res = load_housing()
    assert_equal(res.data.shape, (1681, 3))
    assert_equal(res.target.size, 1681)
    assert_equal(len(res.feature_names), 3)
    assert_true(res.DESCR)
