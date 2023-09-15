"""
created matt_dumont 
on: 15/09/23
"""
import numpy as np
from non_parametric.non_parametric_stats import _mann_kendall_from_sarray, _make_s_array, _mann_kendall_old


def _quick_test_s():
    x = np.random.rand(100)
    s_new = _make_s_array(x).sum()
    s = 0
    n = len(x)
    for k in range(n - 1):
        for j in range(k + 1, n):
            s += np.sign(x[j] - x[k])
    assert s_new == s


def test_new_old_mann_kendall():
    x = np.random.rand(100)
    trend, h, p, z, s, var_s = _mann_kendall_from_sarray(x)
    trend_old, h_old, p_old, z_old, s_old, var_s_old = _mann_kendall_old(x)
    assert trend == trend_old
    assert h == h_old
    assert p == p_old
    assert z == z_old
    assert s == s_old
    assert var_s == var_s_old


def test_part_mann_kendall():
    x = np.random.rand(500)
    s_array_full = _make_s_array(x)
    for i in np.arange(8, 400):
        old = _mann_kendall_from_sarray(x[i:])
        new = _mann_kendall_from_sarray(x[i:], sarray=s_array_full[i:, i:])
        assert old == new


