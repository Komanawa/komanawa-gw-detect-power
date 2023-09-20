"""
created matt_dumont 
on: 15/09/23
"""
import itertools
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

from non_parametric.non_parametric_stats import _mann_kendall_from_sarray, _make_s_array, _mann_kendall_old, \
    _seasonal_mann_kendall_from_sarray, _old_smk, MannKendall, SeasonalKendall, MultiPartKendall, \
    SeasonalMultiPartKendall, _calc_seasonal_senslope, get_colors, _generate_startpoints


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


def make_seasonal_data(slope, noise, unsort, na_data):
    x, y = make_increasing_decreasing_data(slope=slope, noise=noise)
    assert len(x) % 4 == 0
    # add/reduce data in each season (create bias + +- noise)
    seasons = np.repeat([[1, 2, 3, 4]], len(x) // 4, axis=0).flatten()
    y[seasons == 1] += 0 * noise / 2
    y[seasons == 2] += 2 * noise / 2
    y[seasons == 3] += 0 * noise / 2
    y[seasons == 4] += -2 * noise / 2

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

    return test_dataframe


def test_seasonal_data():
    save_path = Path(__file__).parent.joinpath('test_data', 'test_seasonal_data.hdf')
    write_test_data = False
    slope = 0.1
    noise = 10
    for sort, na_data in itertools.product([True, False], [True, False]):
        test_dataframe = make_seasonal_data(slope, noise, sort, na_data)
        test_name = f'slope_{slope}_noise_{noise}_unsort_{sort}_na_data_{na_data}'.replace('.', '_').replace('-', '_')
        if write_test_data:
            test_dataframe.to_hdf(save_path, test_name)
        else:
            expect = pd.read_hdf(save_path, test_name)
            assert isinstance(expect, pd.DataFrame)
            pd.testing.assert_frame_equal(test_dataframe, expect, check_names=False, obj=test_name)
    for na_data in [True, False]:
        sort_name = f'slope_{slope}_noise_{noise}_unsort_{False}_na_data_{na_data}'.replace('.', '_').replace('-', '_')
        sort_data = pd.read_hdf(save_path, sort_name)
        assert isinstance(sort_data, pd.DataFrame)
        sort_data.name = sort_name
        unsort_name = f'slope_{slope}_noise_{noise}_unsort_{True}_na_data_{na_data}'.replace('.', '_').replace('-', '_')
        unsort_data = pd.read_hdf(save_path, unsort_name)
        assert isinstance(unsort_data, pd.DataFrame)
        unsort_data.name = unsort_name
        unsort_data = unsort_data.sort_index()
        pd.testing.assert_frame_equal(sort_data, unsort_data, check_names=False, obj=sort_name)
        assert np.allclose(sort_data.values, unsort_data.values, equal_nan=True)


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
        got_data = got_data.astype(float)
        test_name = f'slope_{slope}_noise_{noise}_unsort_{unsort}_na_data_{na_data}'.replace('.', '_').replace('-', '_')
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
        sort_name = f'slope_{slope}_noise_{noise}_unsort_{True}_na_data_{na_data}'.replace('.', '_').replace('-', '_')
        sort_data = pd.read_hdf(test_data_path, sort_name)
        unsort_name = f'slope_{slope}_noise_{noise}_unsort_{True}_na_data_{na_data}'.replace('.', '_').replace('-', '_')
        unsort_data = pd.read_hdf(test_data_path, unsort_name)
        assert isinstance(sort_data, pd.Series)
        assert isinstance(unsort_data, pd.Series)
        pd.testing.assert_series_equal(sort_data, unsort_data, check_names=False)


def test_seasonal_senslope():
    seasonal_data_base = make_seasonal_data(slope=0.1, noise=10, unsort=False, na_data=False)
    seasonal_data1 = deepcopy(seasonal_data_base)
    seasonal_data2 = deepcopy(seasonal_data_base)

    t1 = _calc_seasonal_senslope(y=seasonal_data1['y'], season=seasonal_data1['seasons'], x=seasonal_data1.index,
                                 alpha=0.05)
    t2 = _calc_seasonal_senslope(y=seasonal_data2['y'], season=seasonal_data2['seasons'], x=seasonal_data2.index,
                                 alpha=0.05)
    assert np.allclose(t1, t2)

    # test sort vs unsort
    seasonal_data1 = deepcopy(seasonal_data_base)
    seasonal_data2 = make_seasonal_data(slope=0.1, noise=10, unsort=True, na_data=False)
    seasonal_data2 = seasonal_data2.sort_index()
    assert np.allclose(seasonal_data1.values, seasonal_data2.values, equal_nan=True)
    t1 = _calc_seasonal_senslope(y=seasonal_data1['y'], season=seasonal_data1['seasons'], x=seasonal_data1.index,
                                 alpha=0.05)
    t2 = _calc_seasonal_senslope(y=seasonal_data2['y'], season=seasonal_data2['seasons'], x=seasonal_data2.index,
                                 alpha=0.05)
    assert np.allclose(t1, t2)

    # test sort vs unsort with nan
    seasonal_data1 = make_seasonal_data(slope=0.1, noise=10, unsort=False, na_data=True)
    seasonal_data2 = make_seasonal_data(slope=0.1, noise=10, unsort=True, na_data=True)
    seasonal_data2 = seasonal_data2.sort_index()
    assert np.allclose(seasonal_data1.values, seasonal_data2.values, equal_nan=True)
    seasonal_data1 = seasonal_data1.dropna(subset=['y', 'seasons'])
    seasonal_data2 = seasonal_data2.dropna(subset=['y', 'seasons'])
    assert np.allclose(seasonal_data1.values, seasonal_data2.values, equal_nan=True)
    t1 = _calc_seasonal_senslope(y=seasonal_data1['y'], season=seasonal_data1['seasons'], x=seasonal_data1.index,
                                 alpha=0.05)
    t2 = _calc_seasonal_senslope(y=seasonal_data2['y'], season=seasonal_data2['seasons'], x=seasonal_data2.index,
                                 alpha=0.05)
    assert np.allclose(t1, t2)


def test_seasonal_mann_kendall(show=True):
    test_data_path = Path(__file__).parent.joinpath('test_data', 'test_smk.hdf')
    make_test_data = False
    slopes = [0.1, -0.1, 0]
    noises = [5, 10, 50]
    unsorts = [True, False]
    na_datas = [True, False]
    for slope, noise, unsort, na_data in itertools.product(slopes, noises, unsorts, na_datas):
        test_dataframe = make_seasonal_data(slope, noise, unsort, na_data)

        mk_df = SeasonalKendall(df=test_dataframe, data_col='y', season_col='seasons', alpha=0.05, rm_na=True,
                                freq_limit=0.05)

        # test results
        df_ss_data = np.array(mk_df.calc_senslope())

        got_data = pd.Series(dict(trend=mk_df.trend, h=mk_df.h, p=mk_df.p, z=mk_df.z, s=mk_df.s,
                                  var_s=mk_df.var_s, senslope=df_ss_data[0],
                                  sen_intercept=df_ss_data[1],
                                  lo_slope=df_ss_data[2],
                                  up_slope=df_ss_data[3], ))
        got_data = got_data.astype(float)
        test_name = f'slope_{slope}_noise_{noise}_unsort_{unsort}_na_data_{na_data}'.replace('.', '_').replace('-', '_')
        # test plot data
        fig, ax = mk_df.plot_data()
        ax.set_title(test_name)
        if show:
            plt.show()
        plt.close('all')
        if not make_test_data:
            test_data = pd.read_hdf(test_data_path, test_name)
            assert isinstance(test_data, pd.Series)
            pd.testing.assert_series_equal(got_data, test_data, check_names=False, obj=test_name)
        else:
            got_data.to_hdf(test_data_path, test_name, complevel=9, complib='blosc:lz4')

        # test that sort vs unsort doesn't change results
    for slope, noise, na_data in itertools.product(slopes, noises, na_datas):
        sort_name = f'slope_{slope}_noise_{noise}_unsort_{True}_na_data_{na_data}'.replace('.', '_').replace('-', '_')
        sort_data = pd.read_hdf(test_data_path, sort_name)
        sort_data.name = sort_name
        unsort_name = f'slope_{slope}_noise_{noise}_unsort_{False}_na_data_{na_data}'.replace('.', '_').replace('-',
                                                                                                                '_')
        unsort_data = pd.read_hdf(test_data_path, unsort_name)
        unsort_data.name = unsort_name
        assert isinstance(sort_data, pd.Series)
        assert isinstance(unsort_data, pd.Series)
        pd.testing.assert_series_equal(sort_data, unsort_data, check_names=False, obj=f'{sort_name} & {unsort_name}')


def make_multipart_sharp_change_data(slope, noise, unsort, na_data):
    """
    sharp v change positive slope is increasing and then decreasing, negative is opposite
    :param slope:
    :param noise:
    :param unsort:
    :param na_data:
    :return:
    """
    x = np.arange(100)
    y = np.zeros_like(x).astype(float)
    y[:50] = x[:50] * slope + 100
    y[50:] = (x[50:] - x[49].max()) * slope * -1 + y[49]

    np.random.seed(68)
    noise = np.random.normal(0, noise, len(x))
    y += noise

    if na_data:
        np.random.seed(868)
        na_idxs = np.random.randint(0, len(y), 10)
        y[na_idxs] = np.nan

    if unsort:
        x_use = deepcopy(x)
        np.random.shuffle(x_use)
        y = y[x_use]
        x = x[x_use]

    return x, y


def make_multipart_parabolic_data(slope, noise, unsort, na_data):
    """
    note the slope is multiplied by -1 to retain the same standards make_sharp_change_data
    positive slope is increasing and then decreasing, negative is opposite
    :param slope:
    :param noise:
    :param unsort:
    :param na_data:
    :return:
    """

    x = np.arange(100)
    y = slope * -1 * (x - 49) ** 2 + 100.

    np.random.seed(68)
    noise = np.random.normal(0, noise, len(x))
    y += noise

    if na_data:
        np.random.seed(868)
        na_idxs = np.random.randint(0, len(y), 10)
        y[na_idxs] = np.nan

    if unsort:
        x_use = deepcopy(x)
        np.random.shuffle(x_use)
        y = y[x_use]
        x = x[x_use]

    return x, y


def make_seasonal_multipart_parabolic(slope, noise, unsort, na_data):
    x, y = make_multipart_parabolic_data(slope=slope, noise=noise, unsort=False, na_data=False)
    assert len(x) % 4 == 0
    # add/reduce data in each season (create bias + +- noise)
    seasons = np.repeat([[1, 2, 3, 4]], len(x) // 4, axis=0).flatten()
    y[seasons == 1] += 0 + noise / 4
    y[seasons == 2] += 2 + noise / 4
    y[seasons == 3] += 0 + noise / 4
    y[seasons == 4] += -2 + noise / 4

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

    return test_dataframe


def make_seasonal_multipart_sharp_change(slope, noise, unsort, na_data):
    x, y = make_multipart_sharp_change_data(slope=slope, noise=noise, unsort=False, na_data=False)
    assert len(x) % 4 == 0
    # add/reduce data in each season (create bias + +- noise)
    seasons = np.repeat([[1, 2, 3, 4]], len(x) // 4, axis=0).flatten()
    y[seasons == 1] += 0 + noise / 4
    y[seasons == 2] += 2 + noise / 4
    y[seasons == 3] += 0 + noise / 4
    y[seasons == 4] += -2 + noise / 4

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

    return test_dataframe


multipart_sharp_slopes = [0.1, -0.1, 0]
multipart_sharp_noises = [0, 0.5, 1, 5]


def plot_multipart_data_sharp(show=False):
    # sharp change
    f = make_multipart_sharp_change_data
    colors = get_colors(multipart_sharp_noises)
    for slope in multipart_sharp_slopes:
        for noise, c in zip(multipart_sharp_noises, colors):
            fig, ax = plt.subplots()
            x, y = f(slope, noise, unsort=False, na_data=False)
            ax.scatter(x, y, label=f'noise:{noise}', c=c)
            ax.legend()
            ax.set_title(f'slope:{slope}, f:{f.__name__}')
    if show:
        plt.show()
    plt.close('all')


slope_mod = 1e-2
multipart_parabolic_slopes = [1 * slope_mod, -1 * slope_mod, 0]
multipart_parabolic_noises = [0, 1, 5, 10, 20, 50]


def plot_multipart_data_para(show=False):
    # parabolic
    f = make_multipart_parabolic_data
    colors = get_colors(multipart_parabolic_noises)
    for slope in multipart_parabolic_slopes:
        for noise, c in zip(multipart_parabolic_noises, colors):
            fig, ax = plt.subplots()
            x, y = f(slope, noise, unsort=False, na_data=False)
            x0, y0 = f(slope, 0, False, False)
            ax.plot(x0, y0, c='k', label='no noise', ls=':', alpha=0.5)
            ax.scatter(x, y, label=f'noise:{noise}', c=c)
            ax.legend()
            ax.set_title(f'slope:{slope}, f:{f.__name__}')
    if show:
        plt.show()
    plt.close('all')


def plot_seasonal_multipart_para(show=False):
    f = make_seasonal_multipart_parabolic
    for slope in multipart_parabolic_slopes:
        for noise in multipart_parabolic_noises:
            fig, ax = plt.subplots()
            data = f(slope, noise, unsort=False, na_data=False)
            ax.scatter(data.index, data.y, label=f'noise:{noise}', c=data.seasons)
            ax.legend()
            ax.set_title(f'slope:{slope}, f:{f.__name__}')
    if show:
        plt.show()
    plt.close('all')


def plot_seasonal_multipart_sharp(show=False):
    f = make_seasonal_multipart_sharp_change
    for slope in multipart_sharp_slopes:
        for noise in multipart_sharp_noises:
            fig, ax = plt.subplots()
            data = f(slope, noise, unsort=False, na_data=False)
            ax.scatter(data.index, data.y, label=f'noise:{noise}', c=data.seasons)
            ax.legend()
            ax.set_title(f'slope:{slope}, f:{f.__name__}')
    if show:
        plt.show()
    plt.close('all')


def test_generate_startpoints():
    save_path = Path(__file__).parent.joinpath('test_data', 'test_generate_startpoints.npz')
    write_test_data = False

    x, y = make_multipart_sharp_change_data(slope=multipart_sharp_slopes[0], noise=multipart_sharp_noises[1],
                                            unsort=False, na_data=False)
    part4 = _generate_startpoints(n=len(x), min_size=10, nparts=4, test=True)
    part3 = _generate_startpoints(n=len(x), min_size=10, nparts=3, test=True)
    part2 = _generate_startpoints(n=len(x), min_size=10, nparts=2, test=True)

    if write_test_data:
        np.savez_compressed(save_path, part4=part4, part3=part3, part2=part2)
    else:
        expect = np.load(save_path)
        assert np.allclose(part4, expect['part4'])
        assert np.allclose(part3, expect['part3'])
        assert np.allclose(part2, expect['part2'])


def test_multipart_plotting(show=False):
    # 2 part
    x, y = make_multipart_sharp_change_data(slope=multipart_sharp_slopes[0], noise=multipart_sharp_noises[1],
                                            unsort=False, na_data=False)
    data = pd.Series(y, index=x)
    mk = MultiPartKendall(data=data, data_col=None, alpha=0.05, rm_na=True, no_trend_alpha=0.5,
                          nparts=2, expect_part=(1, -1), min_size=10,
                          serialise_path=None, recalc=False, initalize=True)

    acceptble = mk.get_acceptable_matches()
    fig, axs = plt.subplots(nrows=2, figsize=(10, 10))
    data, kstats = mk.get_data_from_breakpoints(50)
    fig, ax = mk.plot_data_from_breakpoints(50, ax=axs[0])
    mk.plot_data_from_breakpoints(25, ax=axs[1], txt_vloc=0.01)
    fig.tight_layout()

    # 3 part
    x, y = make_multipart_parabolic_data(slope=multipart_parabolic_slopes[0], noise=multipart_parabolic_noises[1],
                                         unsort=False, na_data=False)
    data = pd.Series(y, index=x)
    mk_para = MultiPartKendall(data=data, data_col=None, alpha=0.05, rm_na=True, no_trend_alpha=0.5,
                               nparts=3, expect_part=(1, 0, -1), min_size=10,
                               serialise_path=None, recalc=False, initalize=True)
    # plot
    fig, axs = plt.subplots(nrows=2, figsize=(10, 10))
    data, kstats = mk_para.get_data_from_breakpoints([40, 60])
    fig, ax = mk_para.plot_data_from_breakpoints([40, 60], ax=axs[0])
    mk_para.plot_data_from_breakpoints([50, 60], ax=axs[1], txt_vloc=+0.1)
    fig.tight_layout()

    # plot acceptable matches
    for k in ['p', 'z', 's', 'var_s']:
        print(k)
        fig, ax = mk_para.plot_acceptable_matches(key=k)
        ax.set_title('para')
        fig, ax = mk.plot_acceptable_matches(key=k)
        ax.set_title('sharp')
    if show:
        plt.show()
    plt.close('all')


def test_multipart_serialisation():
    with tempfile.TemporaryDirectory() as tdir:
        # 2 part
        tdir = Path(tdir)
        x, y = make_multipart_sharp_change_data(slope=multipart_sharp_slopes[0], noise=multipart_sharp_noises[1],
                                                unsort=False, na_data=False)
        data = pd.Series(y, index=x)
        mk = MultiPartKendall(data=data, nparts=2, expect_part=(1, -1), min_size=10,
                              data_col=None, alpha=0.05, rm_na=True, no_trend_alpha=0.5,
                              serialise_path=tdir.joinpath('test2.hdf'), recalc=False, initalize=True)

        mk1 = MultiPartKendall(data=data, nparts=2, expect_part=(1, -1), min_size=10,
                               data_col=None, alpha=0.05, rm_na=True, no_trend_alpha=0.5,
                               serialise_path=tdir.joinpath('test2.hdf'), recalc=False, initalize=True)

        mk2 = MultiPartKendall.from_file(tdir.joinpath('test2.hdf'))

        assert mk == mk1
        assert mk == mk2
        shutil.copyfile(tdir.joinpath('test2.hdf'), Path.home().joinpath('Downloads', 'mk_test2.hdf'))

        # 3part
        x, y = make_multipart_sharp_change_data(slope=multipart_sharp_slopes[0], noise=multipart_sharp_noises[1],
                                                unsort=False, na_data=False)
        data = pd.Series(y, index=x)
        mk = MultiPartKendall(data=data, nparts=3, expect_part=(1, 0, -1), min_size=10,
                              data_col=None, alpha=0.05, rm_na=True, no_trend_alpha=0.5,
                              serialise_path=tdir.joinpath('test3.hdf'), recalc=False, initalize=True)

        mk1 = MultiPartKendall(data=data, nparts=3, expect_part=(1, 0, -1), min_size=10,
                               data_col=None, alpha=0.05, rm_na=True, no_trend_alpha=0.5,
                               serialise_path=tdir.joinpath('test3.hdf'), recalc=False, initalize=True)

        mk2 = MultiPartKendall.from_file(tdir.joinpath('test3.hdf'))

        assert mk == mk1
        assert mk == mk2
        shutil.copyfile(tdir.joinpath('test3.hdf'), Path.home().joinpath('Downloads', 'mk_test3.hdf'))


def test_seasonal_multipart_serialisation():
    with tempfile.TemporaryDirectory() as tdir:
        # 2 part
        tdir = Path(tdir)
        data = make_seasonal_multipart_sharp_change(slope=multipart_sharp_slopes[0], noise=multipart_sharp_noises[1],
                                                    unsort=False, na_data=False)
        mk = SeasonalMultiPartKendall(data=data, nparts=2, expect_part=(1, -1), min_size=10,
                                      data_col='y', season_col='seasons', alpha=0.05, rm_na=True, no_trend_alpha=0.5,
                                      serialise_path=tdir.joinpath('test2.hdf'), recalc=False, initalize=True)

        mk1 = SeasonalMultiPartKendall(data=data, nparts=2, expect_part=(1, -1), min_size=10,
                                       data_col='y', season_col='seasons', alpha=0.05, rm_na=True, no_trend_alpha=0.5,
                                       serialise_path=tdir.joinpath('test2.hdf'), recalc=False, initalize=True)

        mk2 = SeasonalMultiPartKendall.from_file(tdir.joinpath('test2.hdf'))

        assert mk == mk1
        assert mk == mk2
        shutil.copyfile(tdir.joinpath('test2.hdf'), Path.home().joinpath('Downloads', 'smk_test2.hdf'))

        # 3part
        data = make_seasonal_multipart_sharp_change(slope=multipart_sharp_slopes[0], noise=multipart_sharp_noises[1],
                                                    unsort=False, na_data=False)
        mk = SeasonalMultiPartKendall(data=data, nparts=3, expect_part=(1, 0, -1), min_size=10,
                                      data_col='y', season_col='seasons', alpha=0.05, rm_na=True, no_trend_alpha=0.5,
                                      serialise_path=tdir.joinpath('test3.hdf'), recalc=False, initalize=True)

        mk1 = SeasonalMultiPartKendall(data=data, nparts=3, expect_part=(1, 0, -1), min_size=10,
                                       data_col='y', season_col='seasons', alpha=0.05, rm_na=True, no_trend_alpha=0.5,
                                       serialise_path=tdir.joinpath('test3.hdf'), recalc=False, initalize=True)

        mk2 = SeasonalMultiPartKendall.from_file(tdir.joinpath('test3.hdf'))

        assert mk == mk1
        assert mk == mk2
        shutil.copyfile(tdir.joinpath('test3.hdf'), Path.home().joinpath('Downloads', 'smk_test3.hdf'))


def test_seasonal_multipart_plotting(show=False):
    data = make_seasonal_multipart_sharp_change(slope=multipart_sharp_slopes[0], noise=multipart_sharp_noises[1],
                                                unsort=False, na_data=False)
    mk = SeasonalMultiPartKendall(data=data, data_col='y', season_col='seasons', alpha=0.05, rm_na=True,
                                  no_trend_alpha=0.5,
                                  nparts=2, expect_part=(1, -1), min_size=10,
                                  serialise_path=None, recalc=False, initalize=True)

    acceptble = mk.get_acceptable_matches()
    fig, axs = plt.subplots(nrows=2, figsize=(10, 10))
    data, kstats = mk.get_data_from_breakpoints(50)
    fig, ax = mk.plot_data_from_breakpoints(50, ax=axs[0])
    mk.plot_data_from_breakpoints(25, ax=axs[1], txt_vloc=0.01)
    fig.tight_layout()

    # 3 part
    data = make_seasonal_multipart_parabolic(slope=multipart_parabolic_slopes[0], noise=multipart_parabolic_noises[1],
                                             unsort=False, na_data=False)
    mk_para = SeasonalMultiPartKendall(data=data, data_col='y', season_col='seasons', alpha=0.05, rm_na=True,
                                       no_trend_alpha=0.5,
                                       nparts=3, expect_part=(1, 0, -1), min_size=10,
                                       serialise_path=None, recalc=False, initalize=True)
    # plot
    fig, axs = plt.subplots(nrows=2, figsize=(10, 10))
    data, kstats = mk_para.get_data_from_breakpoints([40, 60])
    fig, ax = mk_para.plot_data_from_breakpoints([40, 60], ax=axs[0])
    mk_para.plot_data_from_breakpoints([50, 60], ax=axs[1], txt_vloc=+0.1)
    fig.tight_layout()

    # plot acceptable matches
    for k in ['p', 'z', 's', 'var_s']:
        print(k)
        fig, ax = mk_para.plot_acceptable_matches(key=k)
        ax.set_title('para')
        fig, ax = mk.plot_acceptable_matches(key=k)
        ax.set_title('sharp')
    if show:
        plt.show()
    plt.close('all')


def test_multipart_kendall(show=False, print_total=False):
    """

    :param show: flag, show the plots of the data
    :param print_total: flag, just count number of iterations
    :return:
    """
    write_test_data = False
    make_functions = [
        make_multipart_parabolic_data,
        make_multipart_sharp_change_data,
    ]
    make_names = [
        'para',
        'sharp',
    ]
    iter_datas = [
        [
            multipart_parabolic_slopes,
            [multipart_parabolic_noises[1], multipart_parabolic_noises[-1]],
            [False],
            [False],
        ],
        [
            multipart_sharp_slopes,
            [multipart_sharp_noises[1], multipart_sharp_noises[-1]],
            [False, True],
            [False, True],
        ],
    ]

    npart_data = [
        (3, (1, 0, -1), [40, 60]),
        (2, (1, -1), [50]),
    ]
    total = 0
    for mfunc, mname, iterdata, (npart, epart, bpoints) in zip(make_functions, make_names, iter_datas, npart_data):
        for slope, noise, unsort, na_data in itertools.product(*iterdata):
            total += 1
            if print_total:
                continue
            title = f'mk_{mname}, {npart=}\n'
            title += f'{slope=}, {noise=}, {unsort=}, {na_data=}\n'
            print(title)
            x, y = mfunc(slope=slope, noise=noise, unsort=unsort, na_data=na_data)
            data = pd.Series(y, index=x)
            if slope != 0:
                use_epart = [e * np.sign(slope) for e in epart]
            else:
                use_epart = epart
            mk = MultiPartKendall(data, nparts=npart,
                                  expect_part=use_epart,
                                  min_size=10,
                                  alpha=0.05, no_trend_alpha=0.5,
                                  data_col=None, rm_na=True,
                                  serialise_path=None)
            title += f'breakpoint: {mk.idx_values[bpoints]}'
            fname = f'mk_{mname}_' + '_'.join([str(i) for i in [
                slope, noise, unsort, na_data]]).replace('.', '_').replace('-', '_')
            fname += f'_npart{npart}.hdf'
            accept = mk.get_acceptable_matches().astype(float)
            bpoint_data, kstats = mk.get_data_from_breakpoints(bpoints)
            hdf_path = Path(__file__).parent.joinpath('test_data', fname)
            if write_test_data:
                mk.to_file(hdf_path)
                accept.to_hdf(hdf_path, 'accept_data', complevel=9, complib='blosc:lz4')
                for i, df in enumerate(bpoint_data):
                    df.to_hdf(hdf_path, f'bp_data_{i}', complevel=9, complib='blosc:lz4')
                fig, ax = mk.plot_data_from_breakpoints(breakpoints=bpoints)
                ax.set_title(title)
                if show:
                    plt.show()
                plt.close('all')
            else:
                mk1 = MultiPartKendall.from_file(hdf_path)
                accept1 = pd.read_hdf(hdf_path, 'accept_data')
                bpoint_data1 = []
                for i in range(npart):
                    bpoint_data1.append(pd.read_hdf(hdf_path, f'bp_data_{i}'))
                assert isinstance(accept1, pd.DataFrame)
                assert mk == mk1
                pd.testing.assert_frame_equal(accept, accept1)
                for i in range(npart):
                    pd.testing.assert_frame_equal(pd.DataFrame(bpoint_data[i]), pd.DataFrame(bpoint_data1[i]))

    print(f'total tests: {total}')

    # test sorting vs unsorting doesnt change anything
    for slope, noise, unsort, na_data in itertools.product(*iter_datas[1]):
        if not unsort:
            continue
        fname_unsort = f'mk_sharp_' + '_'.join([str(i) for i in [
            slope, noise, unsort, na_data]]).replace('.', '_').replace('-', '_')
        fname_unsort += f'_npart2.hdf'

        fname_sort = f'mk_sharp_' + '_'.join([str(i) for i in [
            slope, noise, False, na_data]]).replace('.', '_').replace('-', '_')
        fname_sort += f'_npart2.hdf'

        mk_sort = MultiPartKendall.from_file(Path(__file__).parent.joinpath('test_data', fname_sort))
        mk_unsort = MultiPartKendall.from_file(Path(__file__).parent.joinpath('test_data', fname_unsort))
        assert mk_sort == mk_unsort


def test_seasonal_multipart_kendall(show=False):  # todo start here

    # todo I can pull from the test data for the non seasonal tests
    # todo I don't need to test na and sort for these as that code is from MultiPartKendall
    # only the stats are different

    raise NotImplementedError


def test_get_best_data():
    # todo I may want to make a zero noise version
    # todo I still need to make this function
    # todo get best data... or whatever for both seasonal and non seasonal, npart 2 and 3, sharp and para
    # todo just read the data in via .from_file
    raise NotImplementedError

# todo check issues...
# todo consider making this a git submodule
if __name__ == '__main__':
    # working test
    test_seasonal_multipart_kendall(True)


    # data plots
    plot_seasonal_multipart_sharp(show=False)
    plot_seasonal_multipart_para(show=False)
    plot_multipart_data_para(show=False)
    plot_multipart_data_sharp(show=False)

    # finished tests
    _quick_test_s()
    test_new_old_mann_kendall()
    test_part_mann_kendall()
    test_seasonal_kendall_sarray()
    test_seasonal_senslope()
    test_seasonal_data()
    test_seasonal_mann_kendall(show=False)
    test_mann_kendall(show=False)
    test_multipart_kendall(show=False, print_total=False)
    test_seasonal_multipart_plotting(False)
    test_multipart_plotting(False)
    test_generate_startpoints()
    test_multipart_serialisation()
    test_seasonal_multipart_serialisation()
