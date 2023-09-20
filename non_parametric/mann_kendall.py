"""
created matt_dumont
on: 14/09/23
"""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm, mstats
import warnings
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D


# todo add this to ksl tools or own repo for easy internal use

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


def _seasonal_mann_kendall_from_sarray(x, season_data, alpha=0.05, sarray=None,
                                       freq_limit=0.05):
    """
    calculate the seasonal mann kendall test for a time series
    after: https://doi.org/10.1029/WR020i006p00727
    :param x: the data
    :param season_data: the season data, will be converted to integers
    :param alpha: significance level
    :param sarray: the s array, if None will be calculated from _make_s_array
    :param freq_limit: the maximum difference in frequency between seasons (as a fraction),
                       if greater than this will raise a warning
    :return:
    """
    # calculate the unique data
    x = np.atleast_1d(x)
    season_data = np.atleast_1d(season_data)
    assert np.issubdtype(season_data.dtype, int) or np.issubdtype(season_data.dtype, np.string_), (
        'season data must be a string, or integer to avoid errors associated with float precision'
    )
    # get unique values convert to integers
    unique_seasons, season_data = np.unique(season_data, return_inverse=True)

    # get unique integer values
    unique_season_ints, counts = np.unique(season_data, return_counts=True)

    relaive_freq = np.abs(counts - counts.mean()) / counts.mean()
    if (relaive_freq > freq_limit).any():
        warnings.warn(f'the discrepancy of frequency of seasons is greater than the limit({freq_limit})'
                      f' this may affect the test'
                      f' the frequency of seasons are {counts}')
    assert season_data.shape == x.shape, 'season data and x must be the same shape'
    assert x.ndim == 1
    n = len(x)
    assert n >= 3, 'need at least 3 data points'

    # calculate s
    if sarray is None:
        sarray = _make_s_array(x)
    assert sarray.shape == (n, n)

    # make the season array
    season_k_array = np.repeat(season_data[:, np.newaxis], n, axis=1)
    season_j_array = season_k_array.transpose()

    s = 0
    var_s = 0
    # run the mann kendall for each season
    for season in unique_season_ints:
        season_idx = (season_k_array == season) & (season_j_array == season)
        temp_s = sarray[season_idx].sum()
        temp_x = x[season_data == season]
        n0 = len(temp_x)

        # calculate the var(s)
        unique_x, unique_counts = np.unique(temp_x, return_counts=True)
        unique_mod = (unique_counts * (unique_counts - 1) * (2 * unique_counts + 5)).sum() * (unique_counts > 1).sum()
        temp_var_s = (n0 * (n0 - 1) * (2 * n0 + 5) + unique_mod) / 18

        s += temp_s
        var_s += temp_var_s

    # calculate the z value
    z = np.abs(np.sign(s)) * (s - np.sign(s)) / np.sqrt(var_s)
    p = 2 * (1 - norm.cdf(abs(z)))  # two tail test
    h = abs(z) > norm.ppf(1 - alpha / 2)

    trend = np.sign(z) * h
    # -1 decreasing, 0 no trend, 1 increasing
    return trend, h, p, z, s, var_s


def _mann_kendall_from_sarray(x, alpha=0.05, sarray=None):
    """
    code optimised mann kendall
    :param x:
    :param alpha:
    :param sarray:
    :return:
    """

    # calculate the unique data
    x = np.atleast_1d(x)
    assert x.ndim == 1
    n = len(x)
    assert n >= 3, 'need at least 3 data points'

    # calculate s
    if sarray is None:
        sarray = _make_s_array(x)
    assert sarray.shape == (n, n)
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


def _mann_kendall_old(x, alpha=0.05):
    """
    the duplicate from above is to return more parameters and put into the mann kendall class
    retrieved from https://mail.scipy.org/pipermail/scipy-dev/2016-July/021413.html
    this was depreciated as _mann_kendall_from_sarray is MUCH faster
    Input:
        x:   a vector of data
        alpha: significance level (0.05 default)
    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p value of the significance test
        z: normalized test statistics
    """
    warnings.warn('this function is depreciated, use _mann_kendall_from_sarray', DeprecationWarning)
    x = np.array(x)
    n = len(x)

    # calculate S
    s = 0
    for k in range(n - 1):
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


def _generate_startpoints(n, min_size, nparts, test=False):
    if nparts == 2:
        all_start_points_out = np.arange(min_size, n - min_size)[:, np.newaxis]
        if test:
            assert ((all_start_points_out - 0 >= min_size) & (n - all_start_points_out >= min_size)).all()
    else:
        all_start_points = []
        for part in range(nparts - 1):
            start_points = np.arange(min_size + min_size * part, n - (min_size * (nparts - 1 - part)))
            all_start_points.append(start_points)

        all_start_points_out = np.array(list(itertools.product(*all_start_points)))
        all_start_points_out = all_start_points_out[np.all(
            [all_start_points_out[:, i] < all_start_points_out[:, i + 1] for i in range(nparts - 2)],
            axis=0)]
        temp = np.concatenate((
            np.array(all_start_points_out),
            np.full((len(all_start_points_out), 1), n)

        ), axis=1)
        sizes = np.diff(temp, axis=1)
        all_start_points_out = all_start_points_out[np.all(sizes >= min_size, axis=1)]
        if test:
            temp = np.concatenate((
                np.array(all_start_points_out),
                np.full((len(all_start_points_out), 1), n)

            ), axis=1)
            sizes = np.diff(temp, axis=1)
            assert np.all(sizes >= min_size)

    return all_start_points_out


def _old_smk(df, data_col, season_col, alpha=0.05, rm_na=True):
    warnings.warn('this function is depreciated, use _mann_kendall_from_sarray', DeprecationWarning)
    if rm_na:
        data = df.dropna(subset=[data_col, season_col])
    else:
        data = df.copy(deep=True)
    data_col = data_col
    season_col = season_col
    alpha = alpha

    # get list of seasons
    season_vals = np.unique(data[season_col])

    # calulate the seasonal MK values
    _season_outputs = {}
    s = 0
    var_s = 0
    for season in season_vals:
        tempdata = data[data_col][data[season_col] == season].sort_index()
        _season_outputs[season] = MannKendall(data=tempdata, alpha=alpha, rm_na=rm_na)
        var_s += _season_outputs[season].var_s
        s += _season_outputs[season].s

    # calculate the z value
    z = np.abs(np.sign(s)) * (s - np.sign(s)) / np.sqrt(var_s)

    h = abs(z) > norm.ppf(1 - alpha / 2)
    p = 2 * (1 - norm.cdf(abs(z)))  # two tail test
    trend = np.sign(z) * h
    # -1 decreasing, 0 no trend, 1 increasing
    return trend, h, p, z, s, var_s


def _calc_seasonal_senslope(y, season, x=None, alpha=0.95, method='separate'):
    """
    modified from scipy/stats/_stats_mstats_common.py
    Computes the Theil-Sen estimator for a set of points (x, y).

    `theilslopes` implements a method for robust linear regression.  It
    computes the slope as the median of all slopes between paired values.

    Parameters
    ----------
    y : array_like
        Dependent variable.
    x : array_like or None, optional
        Independent variable. If None, use ``arange(len(y))`` instead.
    alpha : float, optional
        Confidence degree between 0 and 1. Default is 95% confidence.
        Note that `alpha` is symmetric around 0.5, i.e. both 0.1 and 0.9 are
        interpreted as "find the 90% confidence interval".
    method : {'joint', 'separate'}, optional
        Method to be used for computing estimate for intercept.
        Following methods are supported,

            * 'joint': Uses np.median(y - slope * x) as intercept.
            * 'separate': Uses np.median(y) - slope * np.median(x)
                          as intercept.

        The default is 'separate'.

        .. versionadded:: 1.8.0

    Returns
    -------
    result : ``TheilslopesResult`` instance
        The return value is an object with the following attributes:

        slope : float
            Theil slope.
        intercept : float
            Intercept of the Theil line.
        low_slope : float
            Lower bound of the confidence interval on `slope`.
        high_slope : float
            Upper bound of the confidence interval on `slope`.

    See Also
    --------
    siegelslopes : a similar technique using repeated medians

    Notes
    -----
    The implementation of `theilslopes` follows [1]_. The intercept is
    not defined in [1]_, and here it is defined as ``median(y) -
    slope*median(x)``, which is given in [3]_. Other definitions of
    the intercept exist in the literature such as  ``median(y - slope*x)``
    in [4]_. The approach to compute the intercept can be determined by the
    parameter ``method``. A confidence interval for the intercept is not
    given as this question is not addressed in [1]_.

    For compatibility with older versions of SciPy, the return value acts
    like a ``namedtuple`` of length 4, with fields ``slope``, ``intercept``,
    ``low_slope``, and ``high_slope``, so one can continue to write::

        slope, intercept, low_slope, high_slope = theilslopes(y, x)

    References
    ----------
    .. [1] P.K. Sen, "Estimates of the regression coefficient based on
           Kendall's tau", J. Am. Stat. Assoc., Vol. 63, pp. 1379-1389, 1968.
    .. [2] H. Theil, "A rank-invariant method of linear and polynomial
           regression analysis I, II and III",  Nederl. Akad. Wetensch., Proc.
           53:, pp. 386-392, pp. 521-525, pp. 1397-1412, 1950.
    .. [3] W.L. Conover, "Practical nonparametric statistics", 2nd ed.,
           John Wiley and Sons, New York, pp. 493.
    .. [4] https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator
"""
    if method not in ['joint', 'separate']:
        raise ValueError("method must be either 'joint' or 'separate'."
                         "'{}' is invalid.".format(method))
    # We copy both x and y so we can use _find_repeats.
    y = np.array(y).flatten()
    season = np.array(season).flatten()
    if len(season) != len(y):
        raise ValueError("Incompatible lengths ! (%s<>%s)" %
                         (len(y), len(season)))

    if x is None:
        x = np.arange(len(y), dtype=float)
    else:
        x = np.array(x, dtype=float).flatten()
        if len(x) != len(y):
            raise ValueError("Incompatible lengths ! (%s<>%s)" %
                             (len(y), len(x)))
    if len(x) == 1:
        msg = "Theil-Sen estimator is not defined for a single point."
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return np.nan, np.nan, np.nan, np.nan
    # Compute sorted slopes only when deltax > 0
    deltax = x[:, np.newaxis] - x
    deltay = y[:, np.newaxis] - y

    # remove slopes where the seasons do not match
    seasons_array_i = np.repeat(season[:, np.newaxis], len(season), axis=1)
    seasons_array_j = np.repeat(season[np.newaxis, :], len(season), axis=0)
    seasons_array = seasons_array_i == seasons_array_j
    deltax[~seasons_array] = 0
    deltay[~seasons_array] = 0

    slopes = deltay[deltax > 0] / deltax[deltax > 0]
    if not slopes.size:
        msg = "All `x` coordinates are identical."
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    slopes.sort()
    medslope = np.nanmedian(slopes)
    if method == 'joint':
        medinter = np.median(y - medslope * x)
    else:
        medinter = np.median(y) - medslope * np.median(x)
    # Now compute confidence intervals
    if alpha > 0.5:
        alpha = 1. - alpha
    from scipy.stats import distributions

    z = distributions.norm.ppf(alpha / 2.)
    # This implements (2.6) from Sen (1968)
    _, nxreps = _find_repeats(x)
    _, nyreps = _find_repeats(y)
    nt = len(slopes)  # N in Sen (1968)
    ny = len(y)  # n in Sen (1968)
    # Equation 2.6 in Sen (1968):
    sigsq = 1 / 18. * (ny * (ny - 1) * (2 * ny + 5) -
                       sum(k * (k - 1) * (2 * k + 5) for k in nxreps) -
                       sum(k * (k - 1) * (2 * k + 5) for k in nyreps))
    # Find the confidence interval indices in `slopes`
    try:
        sigma = np.sqrt(sigsq)
        Ru = min(int(np.round((nt - z * sigma) / 2.)), len(slopes) - 1)
        Rl = max(int(np.round((nt + z * sigma) / 2.)) - 1, 0)
        delta = slopes[[Rl, Ru]]
    except (ValueError, IndexError):
        delta = (np.nan, np.nan)
    low_slope = delta[0]
    high_slope = delta[1]
    slope = medslope
    intercept = medinter
    return slope, intercept, low_slope, high_slope


def get_colors(vals, cmap='tab10'):
    n_scens = len(vals)
    if n_scens < 20:
        cmap = get_cmap(cmap)
        colors = [cmap(e / (n_scens + 1)) for e in range(n_scens)]
    else:
        colors = []
        i = 0
        cmap = get_cmap(cmap)
        for v in vals:
            colors.append(cmap(i / 20))
            i += 1
            if i == 20:
                i = 0
    return colors


def _find_repeats(arr):
    # taken from scipy.stats._stats_mstats_common._find_repeats
    # This function assumes it may clobber its input.
    if len(arr) == 0:
        return np.array(0, np.float64), np.array(0, np.intp)

    # XXX This cast was previously needed for the Fortran implementation,
    # should we ditch it?
    arr = np.asarray(arr, np.float64).ravel()
    arr.sort()

    # Taken from NumPy 1.9's np.unique.
    change = np.concatenate(([True], arr[1:] != arr[:-1]))
    unique = arr[change]
    change_idx = np.concatenate(np.nonzero(change) + ([arr.size],))
    freq = np.diff(change_idx)
    atleast2 = freq > 1
    return unique[atleast2], freq[atleast2]


class MannKendall(object):
    """
    an object to hold and calculate kendall trends

    :ivar trend: the trend of the data, -1 decreasing, 0 no trend, 1 increasing
    :ivar h: boolean, True if the trend is significant
    :ivar p: the p value of the trend
    :ivar z: the z value of the trend
    :ivar s: the s value of the trend
    :ivar var_s: the variance of the s value
    :ivar alpha: the alpha value used to calculate the trend
    :ivar data: the data used to calculate the trend
    :ivar data_col: the column of the data used to calculate the trend
    """
    "assumes a pandas dataframe or series with a time index"

    def __init__(self, data, alpha=0.05, data_col=None, rm_na=True):
        self.trend_dict = {1: 'increasing', -1: 'decreasing', 0: 'no trend'}
        self.alpha = alpha

        if data_col is not None:
            test_data = data[data_col]
        else:
            test_data = pd.Series(data)
        if rm_na:
            test_data = test_data.dropna(how='any')
        test_data = test_data.sort_index()
        self.data = test_data
        self.data_col = data_col
        self.trend, self.h, self.p, self.z, self.s, self.var_s = _mann_kendall_from_sarray(test_data, alpha=alpha)

    def calc_senslope(self):
        senslope, senintercept, lo_slope, up_slope = mstats.theilslopes(self.data, self.data.index, alpha=self.alpha)
        return senslope, senintercept, lo_slope, up_slope

    def plot_data(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure()

        sslope, sintercept, lo_slope, up_slope = self.calc_senslope()
        # plot the senslope fit and intercept
        x = self.data.index
        y = x * sslope + sintercept
        ax.plot(x, y, color='k', ls='--', label=f'sen slope fit')

        ax.scatter(self.data.index, self.data.values, c='k', label=f'raw data')

        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color='w', ls='--'))
        labels.append(f'sen slope fit: {sslope:.2e}')
        handles.append(Line2D([0], [0], color='w', ls='--'))
        labels.append(f'sen intercept: {sintercept:.2e}')
        handles.append(Line2D([0], [0], color='w', ls='--'))
        labels.append(f'trend: {self.trend_dict[self.trend]}')
        handles.append(Line2D([0], [0], color='w', ls='--'))
        labels.append(f'p: {self.p:0.3f}')

        ax.legend(handles, labels)
        return fig, ax


class SeasonalKendall(object):
    """
    an object to hold and calculate seasonal kendall trends

    :ivar trend: the trend of the data, -1 decreasing, 0 no trend, 1 increasing
    :ivar h: boolean, True if the trend is significant
    :ivar p: the p value of the trend
    :ivar z: the z value of the trend
    :ivar s: the s value of the trend
    :ivar var_s: the variance of the s value
    :ivar alpha: the alpha value used to calculate the trend
    :ivar data: the data used to calculate the trend
    :ivar data_col: the column of the data used to calculate the trend
    :ivar season_col: the column of the season data used to calculate the trend
    :ivar freq_limit: the maximum difference in frequency between seasons (as a fraction),
                        if greater than this will raise a warning
    """

    def __init__(self, df, data_col, season_col, alpha=0.05, rm_na=True,
                 freq_limit=0.05):
        """
        intis and calculate the seasonal mann kendall
        outputs are held by

        :param df: pd.DataFrame holding the season and data columns, expect the index to be the time index
                    e.g. sort_index will sort the data by time
        :param data_col: name of the data column
        :param season_col: name of the season column
        :param alpha: the alpha limit for the p value
        :param rm_na: boolean dropna
        :param freq_limit: the maximum difference in frequency between seasons (as a fraction),
                       if greater than this will raise a warning
        """
        self.trend_dict = {1: 'increasing', -1: 'decreasing', 0: 'no trend'}
        assert isinstance(df, pd.DataFrame), 'df must be a pandas DataFrame'

        self.freq_limit = freq_limit
        if rm_na:
            self.data = df.dropna(subset=[data_col, season_col]).sort_index()
        else:
            self.data = df.copy(deep=True).sort_index()
        self.data_col = data_col
        self.season_col = season_col
        self.alpha = alpha

        x = self.data[data_col]
        self.season_data = season_data = self.data[season_col]

        trend, h, p, z, s, var_s = _seasonal_mann_kendall_from_sarray(x, season_data, alpha=self.alpha, sarray=None,
                                                                      freq_limit=self.freq_limit)
        self.trend = trend
        self.h = h
        self.p = p
        self.z = z
        self.s = s
        self.var_s = var_s

        # -1 decreasing, 0 no trend, 1 increasing

    def calc_senslope(self):
        senslope, senintercept, lo_slope, lo_intercept = _calc_seasonal_senslope(self.data[self.data_col],
                                                                                 self.season_data, x=self.data.index,
                                                                                 alpha=self.alpha)
        return senslope, senintercept, lo_slope, lo_intercept

    def plot_data(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure()

        sslope, sintercept, lo_slope, up_slope = self.calc_senslope()
        # plot the senslope fit and intercept
        x = self.data.index
        y = x * sslope + sintercept
        ax.plot(x, y, color='k', ls='--', label=f'sen slope fit')

        ax.scatter(self.data.index, self.data[self.data_col], c=self.season_data, label=f'raw data')

        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color='w', ls='--'))
        labels.append(f'sen slope fit: {sslope:.2e}')
        handles.append(Line2D([0], [0], color='w', ls='--'))
        labels.append(f'sen intercept: {sintercept:.2e}')
        handles.append(Line2D([0], [0], color='w', ls='--'))
        labels.append(f'trend: {self.trend_dict[self.trend]}')
        handles.append(Line2D([0], [0], color='w', ls='--'))
        labels.append(f'p: {self.p:.3f}')

        ax.legend(handles, labels)
        return fig, ax
