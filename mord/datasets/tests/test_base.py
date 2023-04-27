from mord.datasets.base import load_housing


def test_load_housing():
    res = load_housing()
    assert res.data.shape == (1681, 3)
    assert res.target.size == 1681
    assert len(res.feature_names) == 3
    assert res.DESCR
