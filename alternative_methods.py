"""
created matt_dumont 
on: 14/09/23
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from copy import deepcopy


def pettit_test(): # todo propogate to ksltools
    raise NotImplementedError


def two_part_mann_kendall(x, min_size=10, alpha=0.05): # todo propogate to ksltools
    """
    two part mann kendall test to indentify a change point in a time series
    after Frollini et al., 2020, DOI: 10.1007/s11356-020-11998-0
    :param x: time series data
    :param min_size: minimum size for the first and last section of the time series
    :param alpha: significance level
    :return:
    """
    n = len(x)
    if n / 2 < min_size:
        raise ValueError('the time series is too short for the minimum size')
    s_array = _make_s_array(x)
    p1 = []
    p2 = []
    for i in range(min_size, n - min_size):
        p1.append(mann_kendall_from_sarray(x[:i], alpha=alpha, sarray=s_array[:i, :i]))
        p2.append(mann_kendall_from_sarray(x[i:], alpha=alpha, sarray=s_array[i:, i:]))
    p1 = pd.DataFrame(p1, columns=['trend', 'h', 'p', 'z', 's', 'var_s'])
    p2 = pd.DataFrame(p2, columns=['trend', 'h', 'p', 'z', 's', 'var_s'])

    # todo what happens from here??

    raise NotImplementedError

def _make_s_array(x):
    """
    make the s array for the mann kendall test
    :param x:
    :return:
    """
    n = len(x)
    k_array = np.repeat(x[:, np.newaxis], n, axis=1)
    j_array = k_array.transpose()

    s_stat_array = np.sign(k_array - j_array)
    s_stat_array = np.tril(s_stat_array, -1)  # remove the diagonal and upper triangle
    return s_stat_array


def _quick_test_s():
    x = np.random.rand(100)
    s_new = _make_s_array(x).sum()
    s = 0
    n = len(x)
    for k in range(n - 1):
        for j in range(k + 1, n):
            s += np.sign(x[j] - x[k])
    assert s_new == s


def mann_kendall_from_sarray(x, alpha=0.05, sarray=None):  # todo propogate to ksltools
    """
    code optimised mann kendall
    :param x:
    :param alpha:
    :param sarray:
    :return:
    """

    # calculate the unique data
    x = np.array(x)
    n = len(x)

    # calculate s
    if sarray is None:
        sarray = _make_s_array(x)
    s = sarray.sum()

    # calculate the var(s)
    unique_x, unique_counts = np.unique(x, return_counts=True)
    unique_mod = (unique_counts * (unique_counts - 1) * (2 * unique_counts + 5)).sum() * (unique_counts > 1).sum()
    var_s = (n * (n - 1) * (2 * n + 5) + unique_mod) / 18

    z = np.abs(np.sign(s)) * (s - np.sign(s)) / np.sqrt(var_s)

    # calculate the p_value
    p = 2 * (1 - norm.cdf(abs(z)))  # two tail test
    h = abs(z) > norm.ppf(1 - alpha / 2)

    trend = np.sign(z) * h
    # -1 decreasing, 0 no trend, 1 increasing

    return trend, h, p, z, s, var_s


def test_new_old_mann_kendall():
    x = np.random.rand(100)
    trend, h, p, z, s, var_s = mann_kendall_from_sarray(x)
    trend_old, h_old, p_old, z_old, s_old, var_s_old = _mann_kendall_test(x)
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
        old = mann_kendall_from_sarray(x[i:])
        new = mann_kendall_from_sarray(x[i:], sarray=s_array_full[i:, i:])
        assert old == new


def _mann_kendall_test(x, alpha=0.05):
    """
    the duplicate from above is to return more parameters and put into the mann kendall class
    retrieved from https://mail.scipy.org/pipermail/scipy-dev/2016-July/021413.html
    Input:
        x:   a vector of data
        alpha: significance level (0.05 default)
    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p value of the significance test
        z: normalized test statistics
    """
    x = np.array(x)
    n = len(x)

    # calculate S
    s = 0
    for k in range(n - 1):  # todo this could proably be sped up!!, maybe set up to multiprocess???
        for j in range(k + 1, n):
            s += np.sign(x[j] - x[k])

    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)

    # calculate the var(s)
    if n == g:  # there is no tie
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
    else:  # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(unique_x[i] == x)
        var_s = (n * (n - 1) * (2 * n + 5) + np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s == 0:
        z = 0
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        raise ValueError('shouldnt get here')

    # calculate the p_value
    p = 2 * (1 - norm.cdf(abs(z)))  # two tail test
    h = abs(z) > norm.ppf(1 - alpha / 2)

    if (z < 0) and h:
        trend = -1
    elif (z > 0) and h:
        trend = 1
    else:
        trend = 0

    return trend, h, p, z, s, var_s


if __name__ == '__main__':
    _quick_test_s()
    test_new_old_mann_kendall()
    test_part_mann_kendall()
