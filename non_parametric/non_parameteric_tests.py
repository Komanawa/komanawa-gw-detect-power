"""
created matt_dumont 
on: 15/09/23
"""
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

from non_parametric.non_parametric_stats import _mann_kendall_from_sarray, _make_s_array, _mann_kendall_old, \
    _seasonal_mann_kendall_from_sarray, _old_smk, MannKendall, SeasonalKendall, MultiPartKendall, \
    SeasonalMultiPartKendall


def _quick_test_s():
    np.random.seed(54)
    x = np.random.rand(100)
    s_new = _make_s_array(x).sum()
    s = 0
    n = len(x)
    for k in range(n - 1):
        for j in range(k + 1, n):
            s += np.sign(x[j] - x[k])
    assert s_new == s


def test_new_old_mann_kendall():
    np.random.seed(54)
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
    np.random.seed(54)
    x = np.random.rand(500)
    s_array_full = _make_s_array(x)
    for i in np.arange(8, 400):
        old = _mann_kendall_from_sarray(x[i:])
        new = _mann_kendall_from_sarray(x[i:], sarray=s_array_full[i:, i:])
        assert old == new


def test_seasonal_kendall_sarray():
    np.random.seed(54)
    x = np.random.rand(500)
    seasons = np.repeat([np.arange(4)], 125, axis=0).flatten()
    new = _seasonal_mann_kendall_from_sarray(x, seasons)
    old = _old_smk(pd.DataFrame(dict(x=x, seasons=seasons)), 'x', 'seasons')
    assert new == old


def make_increasing_decreasing_data(slope=1, noise=1):
    x = np.arange(100).astype(float)
    y = x * slope
    np.random.seed(68)
    noise = np.random.normal(0, noise, len(x))
    y += noise
    return x, y


def test_mann_kendall(show=False):
    test_data_path = Path(__file__).parent.joinpath('test_data', 'test_mk.hdf')
    make_test_data = False
    slopes = [0.1, -0.1, 0]
    noises = [5, 10, 50]
    unsorts = [True, False]
    na_datas = [True, False]
    for slope, noise, unsort, na_data in itertools.product(slopes, noises, unsorts, na_datas):

        x, y = make_increasing_decreasing_data(slope=slope, noise=noise)
        if na_data:
            np.random.seed(868)
            na_idxs = np.random.randint(0, len(y), 10)
            y[na_idxs] = np.nan

        # test passing numpy array
        mk_array = MannKendall(data=y, alpha=0.05, data_col=None, rm_na=True)

        # test passing Series
        test_data = pd.Series(y, index=x)
        if unsort:
            x_use = deepcopy(x)
            np.random.shuffle(x_use)
            test_data = test_data.loc[x_use]
        mk_series = MannKendall(data=test_data, alpha=0.05, data_col=None, rm_na=True)

        # test passing data col (with other noisy cols)
        test_dataframe = pd.DataFrame(index=x, data=y, columns=['y'])
        for col in ['lkj', 'lskdfj', 'laskdfj']:
            test_dataframe[col] = np.random.choice([1, 34.2, np.nan])
        if unsort:
            x_use = deepcopy(x)
            np.random.shuffle(x_use)
            test_dataframe = test_dataframe.loc[x_use]
        mk_df = MannKendall(data=test_dataframe, alpha=0.05, data_col='y', rm_na=True)

        # test results
        assert mk_array.trend == mk_series.trend == mk_df.trend
        assert mk_array.h == mk_series.h == mk_df.h
        assert np.allclose(mk_array.p, mk_series.p)
        assert np.allclose(mk_array.p, mk_df.p)

        assert np.allclose(mk_array.z, mk_series.z)
        assert np.allclose(mk_array.z, mk_df.z)

        assert np.allclose(mk_array.s, mk_df.s)
        assert np.allclose(mk_array.s, mk_df.s)

        assert np.allclose(mk_array.var_s, mk_series.var_s)
        assert np.allclose(mk_array.var_s, mk_df.var_s)

        # senslopes
        array_ss_data = np.array(mk_array.calc_senslope())
        series_ss_data = np.array(mk_series.calc_senslope())
        df_ss_data = np.array(mk_df.calc_senslope())
        assert np.allclose(array_ss_data, series_ss_data)
        assert np.allclose(array_ss_data, df_ss_data)

        got_data = pd.Series(dict(trend=mk_array.trend, h=mk_array.h, p=mk_array.p, z=mk_array.z, s=mk_array.s,
                                  var_s=mk_array.var_s, senslope=array_ss_data[0],
                                  sen_intercept=array_ss_data[1],
                                  lo_slope=array_ss_data[2],
                                  up_slope=array_ss_data[3], ))
        test_name = f'slope_{slope}_noise_{noise}_unsort_{unsort}_na_data_{na_data}'
        # test plot data
        fig, ax = mk_array.plot_data()
        ax.set_title(test_name)
        if show:
            plt.show()
        plt.close('all')
        if not make_test_data:
            test_data = pd.read_hdf(test_data_path, test_name)
            assert isinstance(test_data, pd.Series)
            pd.testing.assert_series_equal(got_data, test_data, check_names=False)
        else:
            got_data.to_hdf(test_data_path, test_name)

        # test that sort vs unsort doesn't change results
    for slope, noise, na_data in itertools.product(slopes, noises, na_datas):
        sort_data = pd.read_hdf(test_data_path, f'slope_{slope}_noise_{noise}_unsort_{True}_na_data_{na_data}')
        unsort_data = pd.read_hdf(test_data_path, f'slope_{slope}_noise_{noise}_unsort_{False}_na_data_{na_data}')
        assert isinstance(sort_data, pd.Series)
        assert isinstance(unsort_data, pd.Series)
        pd.testing.assert_series_equal(sort_data, unsort_data, check_names=False)


def test_seasonal_mann_kendall(show=False):  # todo this is looking good, but I'm getting weird issues with sorting and senslope, shouldnt change input data, but it seems to...
    test_data_path = Path(__file__).parent.joinpath('test_data', 'test_smk.hdf')
    make_test_data = False
    slopes = [0.1, -0.1, 0]
    noises = [5, 10, 50]
    unsorts = [True, False]
    na_datas = [True, False]
    for slope, noise, unsort, na_data in itertools.product(slopes, noises, unsorts, na_datas):

        x, y = make_increasing_decreasing_data(slope=slope, noise=noise)
        assert len(x) % 4 == 0
        # add/reduce data in each season (create bias + +- noise)
        seasons = np.repeat([[1, 2, 3, 4]], len(x) // 4, axis=0).flatten()
        y[seasons == 1] += 0 * noise/2
        y[seasons == 2] += 2 * noise/2
        y[seasons == 3] += 0 * noise/2
        y[seasons == 4] += -2 * noise/2

        if na_data:
            np.random.seed(868)
            na_idxs = np.random.randint(0, len(y), 10)
            y[na_idxs] = np.nan

        # test passing data col (with other noisy cols)
        test_dataframe = pd.DataFrame(index=x, data=y, columns=['y'])
        test_dataframe['seasons'] = seasons
        for col in ['lkj', 'lskdfj', 'laskdfj']:
            test_dataframe[col] = np.random.choice([1, 34.2, np.nan])
        if unsort:
            x_use = deepcopy(x)
            np.random.shuffle(x_use)
            test_dataframe = test_dataframe.loc[x_use]
        mk_df = SeasonalKendall(df=test_dataframe, data_col='y', season_col='seasons', alpha=0.05, rm_na=True,
                                freq_limit=0.05)

        # test results
        df_ss_data = np.array(mk_df.calc_senslope())

        got_data = pd.Series(dict(trend=mk_df.trend, h=mk_df.h, p=mk_df.p, z=mk_df.z, s=mk_df.s,
                                  var_s=mk_df.var_s, senslope=df_ss_data[0],
                                  sen_intercept=df_ss_data[1],
                                  lo_slope=df_ss_data[2],
                                  up_slope=df_ss_data[3], ))
        test_name = f'slope_{slope}_noise_{noise}_unsort_{unsort}_na_data_{na_data}'
        # test plot data
        fig, ax = mk_df.plot_data()
        ax.set_title(test_name)
        if show:
            plt.show()
        plt.close('all')
        if not make_test_data:
            test_data = pd.read_hdf(test_data_path, test_name)
            assert isinstance(test_data, pd.Series)
            pd.testing.assert_series_equal(got_data, test_data, check_names=False)
        else:
            got_data.to_hdf(test_data_path, test_name)

        # test that sort vs unsort doesn't change results
    for slope, noise, na_data in itertools.product(slopes, noises, na_datas):
        sort_data = pd.read_hdf(test_data_path, f'slope_{slope}_noise_{noise}_unsort_{True}_na_data_{na_data}')
        unsort_data = pd.read_hdf(test_data_path, f'slope_{slope}_noise_{noise}_unsort_{False}_na_data_{na_data}')
        assert isinstance(sort_data, pd.Series)
        assert isinstance(unsort_data, pd.Series)
        pd.testing.assert_series_equal(sort_data, unsort_data, check_names=False)


def make_multipart_sharp_change_data():  # todo
    # todo v data (sharp change)
    raise NotImplementedError


def make_multipart_parabolic_data():  # todo
    # todo use a parabola
    raise NotImplementedError


def test_multipart_kendall(show=False):  # todo
    raise NotImplementedError


def test_seasonal_multipart_kendall(show=False):  # todo
    raise NotImplementedError


if __name__ == '__main__':
    # working test
    test_seasonal_mann_kendall(show=False)

    # finished tests
    _quick_test_s()
    test_new_old_mann_kendall()
    test_part_mann_kendall()
    test_seasonal_kendall_sarray()
    test_mann_kendall(show=False)
