"""
created matt_dumont 
on: 21/09/23
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from matplotlib.lines import Line2D
from pathlib import Path
from copy import deepcopy
from scipy.stats import mstats
from non_parametric.mann_kendall import _mann_kendall_from_sarray, _seasonal_mann_kendall_from_sarray, \
    _calc_seasonal_senslope, get_colors, _generate_startpoints, _make_s_array


class MultiPartKendall():
    """
    multi part mann kendall test to indentify a change point(s) in a time series
    after Frollini et al., 2020, DOI: 10.1007/s11356-020-11998-0
    :ivar acceptable_matches: (boolean index) the acceptable matches for the trend, i.e. the trend is the expected trend
    :ivar all_start_points: all the start points for the mann kendall tests
    :ivar alpha: the alpha value used to calculate the trend
    :ivar no_trend_alpha: significance level for no trend e.g. will accept if p> no_trend_alpha
    :ivar data: the data used to calculate the trend
    :ivar data_col: the column of the data used to calculate the trend
    :ivar datasets: a dictionary of the datasets {f'p{i}':pd.DataFrame for i in range(nparts)}
                    each dataset contains the mann kendall results for each part of the time series
                    (trend (1=increasing, -1=decreasing, 0=no trend), h, p, z, s, var_s)
    :ivar expect_part: the expected trend in each part of the time series (1 increasing, -1 decreasing, 0 no trend)
    :ivar idx_values: the index values of the data used to calculate the trend used in internal plotting
    :ivar min_size: the minimum size for all parts of the timeseries
    :ivar n: number of data points
    :ivar nparts: number of parts to split the time series into
    :ivar rm_na: boolean dropna
    :ivar s_array: the s array used to calculate the trend
    :ivar season_col: the column of the season data used to calculate the trend (not used for this class)
    :ivar season_data: the season data used to calculate the trend (not used for this class)
    :ivar serialise: boolean, True if the class is serialised
    :ivar serialise_path: path to the serialised file
    :ivar x: the data
    """

    def __init__(self, data, nparts=2, expect_part=(1, -1), min_size=10,
                 alpha=0.05, no_trend_alpha=0.5,
                 data_col=None, rm_na=True,
                 serialise_path=None, recalc=False, initalize=True):
        """
        multi part mann kendall test to indentify a change point(s) in a time series
        after Frollini et al., 2020, DOI: 10.1007/s11356-020-11998-0
        note where the expected trend is zero the lack of a trend is considered significant if p > 1-alpha
        :param data: time series data, if DataFrame or Series, expects the index to be sample order (will sort on index)
                     if np.array or list expects the data to be in sample order
        :param nparts: number of parts to split the time series into
        :param expect_part: expected trend in each part of the time series (1 increasing, -1 decreasing, 0 no trend)
        :param min_size: minimum size for the first and last section of the time series
        :param alpha: significance level
        :param no_trend_alpha: significance level for no trend e.g. will accept if p> no_trend_alpha
        :param data_col: if data is a DataFrame or Series, the column to use
        :param rm_na: remove na values from the data
        :param serialise_path: path to serialised file (as hdf), if None will not serialise
        :param recalc: if True will recalculate the mann kendall even if the serialised file exists
        :param initalize: if True will initalize the class from the data, only set to False used in self.from_file
        :return:
        """
        self.trend_dict = {1: 'increasing', -1: 'decreasing', 0: 'no trend'}

        if not initalize:
            assert all([e is None for e in
                        [data, nparts, expect_part, min_size, alpha, no_trend_alpha, data_col, rm_na, serialise_path,
                         recalc]])
        else:
            loaded = False
            if serialise_path is not None:
                serialise_path = Path(serialise_path)
                self.serialise_path = serialise_path
                self.serialise = True
                if Path(serialise_path).exists() and not recalc:
                    loaded = True
                    self._set_from_file(
                        data=data,
                        nparts=nparts,
                        expect_part=expect_part,
                        min_size=min_size,
                        alpha=alpha,
                        no_trend_alpha=no_trend_alpha,
                        data_col=data_col,
                        rm_na=rm_na,
                        season_col=None)
            else:
                self.serialise = False
                self.serialise_path = None

            if not loaded:
                self._set_from_data(data=data,
                                    nparts=nparts,
                                    expect_part=expect_part,
                                    min_size=min_size,
                                    alpha=alpha,
                                    no_trend_alpha=no_trend_alpha,
                                    data_col=data_col,
                                    rm_na=rm_na,
                                    season_col=None)

            if self.serialise and not loaded:
                self.to_file()

    def __eq__(self, other):
        out = True
        out *= isinstance(other, self.__class__)
        out *= self.data_col == other.data_col
        out *= self.rm_na == other.rm_na
        out *= self.season_col == other.season_col
        out *= self.nparts == other.nparts
        out *= self.min_size == other.min_size
        out *= self.alpha == other.alpha
        out *= self.no_trend_alpha == other.no_trend_alpha
        out *= all(np.atleast_1d(self.expect_part) == np.atleast_1d(other.expect_part))
        datatype = type(self.data).__name__
        datatype_other = type(other.data).__name__
        out *= datatype == datatype_other

        if datatype == datatype_other:
            try:
                # check datasets
                if datatype == 'DataFrame':
                    pd.testing.assert_frame_equal(self.data, other.data, check_dtype=False, check_like=True)
                elif datatype == 'Series':
                    pd.testing.assert_series_equal(self.data, other.data, check_dtype=False, check_like=True)
                elif datatype == 'ndarray':
                    assert np.allclose(self.data, other.data)
                else:
                    raise AssertionError(f'unknown datatype {datatype}')
            except AssertionError:
                out *= False

        out *= np.allclose(self.x, other.x)
        out *= np.allclose(self.idx_values, other.idx_values)
        out *= np.all(self.acceptable_matches.values == other.acceptable_matches.values)
        if self.season_col is not None:
            out *= np.allclose(self.season_data, other.season_data)

        out *= np.allclose(self.s_array, other.s_array)
        out *= np.allclose(self.all_start_points, other.all_start_points)
        try:
            for part in range(self.nparts):
                pd.testing.assert_frame_equal(self.datasets[f'p{part}'], other.datasets[f'p{part}'])
        except AssertionError:
            out *= False
        return bool(out)

    def get_acceptable_matches(self):
        outdata = self.datasets['p0'].loc[self.acceptable_matches]
        outdata = outdata.set_index([f'split_point_{i}' for i in range(1, self.nparts)])
        outdata.rename(columns={f'{e}': f'{e}_p0' for e in ['trend', 'h', 'p', 'z', 's', 'var_s']}, inplace=True)
        for i in range(1, self.nparts):
            next_data = self.datasets[f'p{i}'].loc[self.acceptable_matches]
            next_data = next_data.set_index([f'split_point_{j}' for j in range(1, self.nparts)])
            next_data.rename(columns={f'{e}': f'{e}_p{i}' for e in ['trend', 'h', 'p', 'z', 's', 'var_s']},
                             inplace=True)
            outdata = pd.merge(outdata, next_data, left_index=True, right_index=True)

        return outdata

    def get_maxz_breakpoints(self, raise_on_none=False):
        """
        get the breakpoints for the maximum z value???
        :param raise_on_none: bool, if True will raise an error if no acceptable matches, otherwise will return None
        :return:
        """
        acceptable = self.get_acceptable_matches()
        if acceptable.empty:
            if raise_on_none:
                raise ValueError('no acceptable matches')
            else:
                return None
        raise NotImplementedError  # todo think about this

    def get_data_from_breakpoints(self, breakpoints):
        """

        :param breakpoints: beakpoints to split the data, e.g. from self.get_acceptable_matches
        :return: outdata: list of dataframes for each part of the time series
                 senslopes: list of senslopes for each part of the time series
                 senintercepts: list of senintercepts for each part of the time series
                 kendal_stats: dataframe of kendal stats for each part of the time series
        """
        breakpoints = np.atleast_1d(breakpoints)
        assert len(breakpoints) == self.nparts - 1
        outdata = []
        kendal_stats = pd.DataFrame(index=[f'p{i}' for i in range(self.nparts)],
                                    columns=['trend', 'h', 'p', 'z', 's', 'var_s', 'senslope',
                                             'senintercept'])
        for p, (pkey, ds) in enumerate(self.datasets.items()):
            assert pkey == f'p{p}'
            temp = ds.set_index([f'split_point_{i}' for i in range(1, self.nparts)])
            outcols = ['trend', 'h', 'p', 'z', 's', 'var_s']
            kendal_stats.loc[f'p{p}', outcols] = temp.loc[tuple(breakpoints), outcols].values

        start = 0
        for i in range(self.nparts):
            if i == self.nparts - 1:
                end = self.n
            else:
                end = breakpoints[i]
            if isinstance(self.data, pd.DataFrame):
                outdata.append(self.data.loc[self.idx_values[start:end]])
            else:
                outdata.append(deepcopy(
                    pd.Series(index=self.idx_values[start:end], data=self.data[self.idx_values[start:end]])))
            start = end

        # calculate the senslope stats
        for i, ds in enumerate(outdata):
            senslope, senintercept = self._calc_senslope(ds)
            kendal_stats.loc[f'p{i}', 'sen_slope'] = senslope
            kendal_stats.loc[f'p{i}', 'sen_intercept'] = senintercept

        return outdata, kendal_stats

    def plot_acceptable_matches(self, key):
        """
        quickly plot the acceptable matches
        :param key: key to plot (one of ['p', 'z', 's', 'var_s'])
        :return:
        """
        assert key in ['p', 'z', 's', 'var_s']
        fig, ax = plt.subplots(figsize=(10, 8))
        acceptable = self.get_acceptable_matches()
        use_keys = [f'{key}_p{i}' for i in range(self.nparts)]
        acceptable = acceptable[use_keys]
        acceptable.plot(ax=ax, ls='none', marker='o')
        return fig, ax

    def plot_data_from_breakpoints(self, breakpoints, ax=None, txt_vloc=-0.05, add_labels=True):
        """
        plot the data from the breakpoints including the senslope fits
        :param breakpoints:
        :param ax: ax to plot on if None then create the ax
        :param txt_vloc: vertical location of the text (in ax.transAxes)
        :param add_labels: boolean, if True add labels (slope, pval) to the plot
        :return:
        """
        breakpoints = np.atleast_1d(breakpoints)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure

        data, kendal_stats = self.get_data_from_breakpoints(breakpoints)
        trans = blended_transform_factory(ax.transData, ax.transAxes)

        # axhlines at breakpoints
        prev_bp = 0
        for i, bp in enumerate(np.concatenate((breakpoints, [self.n]))):
            if not bp == self.n:
                ax.axvline(self.idx_values[bp], color='k', ls=':')
            sslope = kendal_stats.loc[f"p{i}", "sen_slope"]
            sintercept = kendal_stats.loc[f"p{i}", "sen_intercept"]
            if add_labels:
                ax.text((prev_bp + bp) / 2, txt_vloc,
                        f'expected: {self.trend_dict[self.expect_part[i]]}\n'
                        f'got: slope: {sslope:.3e}, '
                        f'pval:{round(kendal_stats.loc[f"p{i}", "p"], 3)}',
                        transform=trans, ha='center', va='top')

            # plot the senslope fit and intercept
            x = self.idx_values[prev_bp:bp]
            y = x * sslope + sintercept
            ax.plot(x, y, color='k', ls='--')
            prev_bp = bp

        if self.season_data is None:
            colors = get_colors(data)
            for i, (ds, c) in enumerate(zip(data, colors)):
                if isinstance(self.data, pd.DataFrame):
                    ax.scatter(ds.index, ds[self.data_col], c=c, label=f'part {i}')
                else:
                    ax.scatter(ds.index, ds, color=c, label=f'part {i}')
        else:
            seasons = np.unique(self.season_data)
            colors = get_colors(seasons)
            for i, ds in enumerate(data):
                for s, c in zip(seasons, colors):
                    temp = ds[ds[self.season_col] == s]
                    ax.scatter(temp.index, temp[self.data_col], color=c, label=f'season: {s}')

        legend_handles = [Line2D([0], [0], color='k', ls=':'),
                          Line2D([0], [0], color='k', ls='--')]

        legend_labels = ['breakpoint', 'sen slope fit', ]
        nhandles, nlabels = ax.get_legend_handles_labels()
        temp = dict(zip(nlabels, nhandles))
        legend_handles.extend(temp.values())
        legend_labels.extend(temp.keys())
        ax.legend(legend_handles, legend_labels, loc='best')
        return fig, ax

    def _set_from_file(self, data, nparts, expect_part, min_size, alpha, no_trend_alpha, data_col, rm_na,
                       season_col=None, check_inputs=True):
        """
        setup the class data from a serialised file, values are passed to ensure they are consistent
        :param check_inputs: bool, if True will check the inputs match the serialised file
        """
        assert self.serialise_path is not None, 'serialise path not set, should not get here'
        params = pd.read_hdf(self.serialise_path, 'params')
        assert isinstance(params, pd.Series)
        # other parameters
        self.alpha = params['alpha']
        self.no_trend_alpha = params['no_trend_alpha']
        self.nparts = int(params['nparts'])
        self.min_size = int(params['min_size'])
        self.rm_na = bool(params['rm_na'])
        self.n = int(params['n'])
        self.expect_part = [int(params[f'expect_part{i}']) for i in range(self.nparts)]

        params_str = pd.read_hdf(self.serialise_path, 'params_str')
        assert isinstance(params_str, pd.Series)
        datatype = params_str['datatype']
        self.season_col = params_str['season_col']
        if self.season_col == 'None':
            self.season_col = None
        self.data_col = params_str['data_col']
        if self.data_col == 'None':
            self.data_col = None

        # d1 data
        d1_data = pd.read_hdf(self.serialise_path, 'd1_data')
        assert isinstance(d1_data, pd.DataFrame)
        self.x = d1_data['x'].values
        self.idx_values = d1_data['idx_values'].values
        self.acceptable_matches = pd.read_hdf(self.serialise_path, 'acceptable_matches')

        if datatype == 'pd.DataFrame':
            self.data = pd.read_hdf(self.serialise_path, 'data')
            assert isinstance(self.data, pd.DataFrame)
        elif datatype == 'pd.Series':
            self.data = pd.read_hdf(self.serialise_path, 'data')
            assert isinstance(self.data, pd.Series)
        elif datatype == 'np.array':
            self.data = pd.read_hdf(self.serialise_path, 'data').values
            assert isinstance(self.data, np.ndarray)
            assert self.data.ndim == 1
        else:
            raise ValueError('unknown datatype, thou shall not pass')

        if self.season_col is not None:
            self.season_data = self.data.loc[self.idx_values, self.season_col]
        else:
            self.season_data = None

        # s array
        self.s_array = pd.read_hdf(self.serialise_path, 's_array').values
        assert self.s_array.shape == (self.n, self.n)

        # all start points
        self.all_start_points = pd.read_hdf(self.serialise_path, 'all_start_points').values
        assert self.all_start_points.shape == (len(self.all_start_points), self.nparts - 1)

        # datasets
        dtypes = {'trend': 'float64', 'h': 'bool', 'p': 'float64',
                  'z': 'float64', 's': 'float64', 'var_s': 'float64'}
        for part in range(1, self.nparts):
            dtypes.update({f'split_point_{part}': 'int64'})
        self.datasets = {}
        for part in range(self.nparts):
            self.datasets[f'p{part}'] = pd.read_hdf(self.serialise_path, f'part{part}').astype(dtypes)

        if check_inputs:
            # check parameters have not changed
            assert self.data_col == data_col, 'data_col does not match'
            assert self.rm_na == rm_na, 'rm_na does not match'
            assert self.season_col == season_col, 'season_col does not match'
            assert self.nparts == nparts, 'nparts does not match'
            assert self.min_size == min_size, 'min_size does not match'
            assert self.alpha == alpha, 'alpha does not match'
            assert self.no_trend_alpha == no_trend_alpha, 'no_trend_alpha does not match'
            assert all(np.atleast_1d(self.expect_part) == np.atleast_1d(expect_part)), 'expect_part does not match'

            # check datasets
            if datatype == 'pd.DataFrame':
                pd.testing.assert_frame_equal(self.data, data, check_dtype=False, check_like=True)
            elif datatype == 'pd.Series':
                pd.testing.assert_series_equal(self.data, data, check_dtype=False, check_like=True)
            elif datatype == 'np.array':
                assert np.allclose(self.data, data)

    def _set_from_data(self, data, nparts, expect_part, min_size, alpha, no_trend_alpha, data_col, rm_na,
                       season_col=None):
        """
        set up the class data from the input data
        :param data:
        :param nparts:
        :param expect_part:
        :param min_size:
        :param alpha:
        :param data_col:
        :param rm_na:
        :param season_col:
        :return:
        """
        self.data = deepcopy(data)
        self.alpha = alpha
        self.no_trend_alpha = no_trend_alpha
        self.nparts = nparts
        self.min_size = min_size
        self.expect_part = expect_part
        self.data_col = data_col
        self.rm_na = rm_na

        assert len(expect_part) == nparts

        # handle data (including options for season)
        self.season_col = season_col
        if season_col is not None:
            assert isinstance(data, pd.DataFrame) or isinstance(data, dict), ('season_col passed but data is not a '
                                                                              'DataFrame or dictionary')
            assert season_col in data.keys(), 'season_col not in data'
            assert data_col is not None, 'data_col must be passed if season_col is passed'
            assert data_col in data.keys(), 'data_col not in data'
            if rm_na:
                data = data.dropna(subset=[data_col, season_col])
            data = data.sort_index()
            self.season_data = data[season_col]
            self.idx_values = data.index.values
            x = np.array(data[data_col])
            self.x = x
        else:
            self.season_data = None
            if data_col is not None:
                x = pd.Series(data[data_col])
            else:
                x = pd.Series(data)
            if rm_na:
                x = x.dropna(how='any')
            x = x.sort_index()
            self.idx_values = x.index.values
            x = np.array(x)
            self.x = x
        assert x.ndim == 1, 'data must be 1d or multi d but with col_name passed'

        n = len(x)
        self.n = n
        if n / self.nparts < min_size:
            raise ValueError('the time series is too short for the minimum size')
        self.s_array = _make_s_array(x)

        all_start_points = _generate_startpoints(n, self.min_size, self.nparts)
        datasets = {f'p{i}': [] for i in range(nparts)}
        self.all_start_points = all_start_points
        self.datasets = datasets

        self._calc_mann_kendall()

        # find all acceptable matches
        idx = np.ones(len(self.all_start_points), bool)
        for part, expect in enumerate(self.expect_part):
            if expect == 0:
                idx = (idx
                       & (self.datasets[f'p{part}'].trend == expect)
                       & (self.datasets[f'p{part}'].p > self.no_trend_alpha)
                       )
            else:
                idx = (idx
                       & (self.datasets[f'p{part}'].trend == expect)
                       & (self.datasets[f'p{part}'].p < self.alpha)
                       )
        self.acceptable_matches = idx

    def _calc_senslope(self, data):

        if isinstance(self.data, pd.DataFrame):
            senslope, senintercept, lo_slope, up_slope = mstats.theilslopes(data[self.data_col], data.index,
                                                                            alpha=self.alpha)
        else:
            senslope, senintercept, lo_slope, up_slope = mstats.theilslopes(data, data.index, alpha=self.alpha)
        return senslope, senintercept

    def _calc_mann_kendall(self):
        """
        acutually calculate the mann kendall from the sarray, this should be the only thing that needs
        to be updated for the seasonal kendall
        :return:
        """
        for sp in np.atleast_2d(self.all_start_points):
            start = 0
            for i in range(self.nparts):
                if i == self.nparts - 1:
                    end = self.n
                else:
                    end = sp[i]
                data = (*sp,
                        *_mann_kendall_from_sarray(self.x[start:end], alpha=self.alpha,
                                                   sarray=self.s_array[start:end, start:end]))
                self.datasets[f'p{i}'].append(data)
                start = end
        for part in range(self.nparts):
            self.datasets[f'p{part}'] = pd.DataFrame(self.datasets[f'p{part}'],
                                                     columns=[f'split_point_{i}' for i in range(1, self.nparts)]
                                                     + ['trend', 'h', 'p', 'z', 's', 'var_s'])

    def to_file(self, save_path=None, complevel=9, complib='blosc:lz4'):
        """
        save the data to a hdf file

        :param save_path: None (save to self.serialise_path) or path to save the file
        :param complevel: compression level for hdf
        :param complib: compression library for hdf
        :return:
        """
        if save_path is None:
            assert self.serialise_path is not None, 'serialise path not set, should not get here'
            save_path = self.serialise_path
        with pd.HDFStore(save_path, 'w') as hdf:
            # setup single value parameters
            params = pd.Series()
            params_str = pd.Series()

            # should be 1d+ of same length
            d1_data = pd.DataFrame(index=range(len(self.x)))
            d1_data['x'] = self.x
            d1_data['idx_values'] = self.idx_values
            d1_data.to_hdf(hdf, 'd1_data')

            self.acceptable_matches.to_hdf(hdf, 'acceptable_matches', complevel=complevel, complib=complib)
            # save as own datasets
            if isinstance(self.data, pd.DataFrame):
                self.data.to_hdf(hdf, 'data', complevel=complevel, complib=complib)
                params_str['datatype'] = 'pd.DataFrame'
            elif isinstance(self.data, pd.Series):
                self.data.to_hdf(hdf, 'data', complevel=complevel, complib=complib)
                params_str['datatype'] = 'pd.Series'
            else:
                params_str['datatype'] = 'np.array'
                pd.Series(self.data).to_hdf(hdf, 'data', complevel=complevel, complib=complib)

            assert isinstance(self.s_array, np.ndarray)
            pd.DataFrame(self.s_array).to_hdf(hdf, 's_array', complevel=complevel, complib=complib)
            assert isinstance(self.all_start_points, np.ndarray)
            pd.DataFrame(self.all_start_points).to_hdf(hdf, 'all_start_points', complevel=complevel, complib=complib)

            for part in range(self.nparts):
                self.datasets[f'p{part}'].astype(float).to_hdf(hdf, f'part{part}', complevel=complevel, complib=complib)

            # other parameters
            params['alpha'] = self.alpha
            params['no_trend_alpha'] = self.no_trend_alpha
            params['nparts'] = float(self.nparts)
            params['min_size'] = float(self.min_size)
            params['rm_na'] = float(self.rm_na)
            params['n'] = float(self.n)
            for i in range(self.nparts):
                params[f'expect_part{i}'] = float(self.expect_part[i])

            params_str['season_col'] = str(self.season_col)
            params_str['data_col'] = str(self.data_col)
            params.to_hdf(hdf, 'params', complevel=complevel, complib=complib)
            params_str.to_hdf(hdf, 'params_str', complevel=complevel, complib=complib)

    @classmethod
    @staticmethod
    def from_file(path):
        """
        load the class from a serialised file
        :param path:
        :return:
        """
        mpk = MultiPartKendall(
            data=None, nparts=None, expect_part=None, min_size=None, alpha=None, no_trend_alpha=None, data_col=None,
            serialise_path=None, recalc=None, rm_na=None, initalize=False)
        mpk.serialise_path = Path(path)
        mpk.serialise = True
        mpk._set_from_file(data=None, nparts=None, expect_part=None, min_size=None, alpha=None, no_trend_alpha=None,
                           data_col=None, rm_na=None, season_col=None, check_inputs=False)
        return mpk


class SeasonalMultiPartKendall(MultiPartKendall):
    """
    multi part mann kendall test to indentify a change point(s) in a time series
    after Frollini et al., 2020, DOI: 10.1007/s11356-020-11998-0
    :ivar acceptable_matches: (boolean index) the acceptable matches for the trend, i.e. the trend is the expected trend
    :ivar all_start_points: all the start points for the mann kendall tests
    :ivar alpha: the alpha value used to calculate the trend
    :ivar no_trend_alpha: significance level for no trend e.g. will accept if p> no_trend_alpha
    :ivar data: the data used to calculate the trend
    :ivar data_col: the column of the data used to calculate the trend
    :ivar datasets: a dictionary of the datasets {f'p{i}':pd.DataFrame for i in range(nparts)}
                    each dataset contains the mann kendall results for each part of the time series
                    (trend (1=increasing, -1=decreasing, 0=no trend), h, p, z, s, var_s)
    :ivar expect_part: the expected trend in each part of the time series (1 increasing, -1 decreasing, 0 no trend)
    :ivar idx_values: the index values of the data used to calculate the trend used in internal plotting
    :ivar min_size: the minimum size for all parts of the timeseries
    :ivar n: number of data points
    :ivar nparts: number of parts to split the time series into
    :ivar rm_na: boolean dropna
    :ivar s_array: the s array used to calculate the trend
    :ivar season_col: the column of the season data used to calculate the trend
    :ivar season_data: the season data used to calculate the trend
    :ivar serialise: boolean, True if the class is serialised
    :ivar serialise_path: path to the serialised file
    :ivar x: the data
    """

    def __init__(self, data, data_col, season_col, nparts=2, expect_part=(1, -1), min_size=10,
                 alpha=0.05, no_trend_alpha=0.5,
                 rm_na=True,
                 serialise_path=None, recalc=False, initalize=True):
        """
        multi part seasonal mann kendall test to indentify a change point(s) in a time series
        after Frollini et al., 2020, DOI: 10.1007/s11356-020-11998-0
        :param data: time series data, if DataFrame or Series, expects the index to be sample order (will sort on index)
                     if np.array or list expects the data to be in sample order
        :param data_col: if data is a DataFrame or Series, the column to use
        :param season_col: the column to use for the season
        :param nparts: number of parts to split the time series into
        :param expect_part: expected trend in each part of the time series (1 increasing, -1 decreasing, 0 no trend)
        :param min_size: minimum size for the first and last section of the time series
        :param alpha: significance level
        :param no_trend_alpha: significance level for no trend e.g. will accept if p> no_trend_alpha
        :param rm_na: remove na values from the data
        :param serialise_path: path to serialised file (as hdf), if None will not serialise
        :param recalc: if True will recalculate the mann kendall even if the serialised file exists
        :param initalize: if True will initalize the class from the data, only set to False used in self.from_file
        :return:
        """
        self.trend_dict = {1: 'increasing', -1: 'decreasing', 0: 'no trend'}

        if not initalize:
            assert all([e is None for e in
                        [data, nparts, expect_part, min_size, alpha, no_trend_alpha, data_col, rm_na, serialise_path,
                         recalc]])
        else:
            loaded = False
            if serialise_path is not None:
                serialise_path = Path(serialise_path)
                self.serialise_path = serialise_path
                self.serialise = True
                if Path(serialise_path).exists() and not recalc:
                    loaded = True
                    self._set_from_file(
                        data=data,
                        nparts=nparts,
                        expect_part=expect_part,
                        min_size=min_size,
                        alpha=alpha,
                        no_trend_alpha=no_trend_alpha,
                        data_col=data_col,
                        rm_na=rm_na,
                        season_col=season_col)
            else:
                self.serialise = False
                self.serialise_path = None

            if not loaded:
                self._set_from_data(data=data,
                                    nparts=nparts,
                                    expect_part=expect_part,
                                    min_size=min_size,
                                    alpha=alpha,
                                    no_trend_alpha=no_trend_alpha,
                                    data_col=data_col,
                                    rm_na=rm_na,
                                    season_col=season_col)

            if self.serialise and not loaded:
                self.to_file()

    @classmethod
    @staticmethod
    def from_file(path):
        """
        load the class from a serialised file
        :param path:
        :return:
        """
        mpk = SeasonalMultiPartKendall(data=None, data_col=None, season_col=None, nparts=None, expect_part=None,
                                       min_size=None, alpha=None, no_trend_alpha=None, rm_na=None,
                                       serialise_path=None, recalc=None, initalize=False)

        mpk.serialise_path = Path(path)
        mpk.serialise = True
        mpk._set_from_file(data=None, nparts=None, expect_part=None, min_size=None,
                           alpha=None, no_trend_alpha=None, data_col=None,
                           rm_na=None, season_col=None, check_inputs=False)
        return mpk

    def _calc_mann_kendall(self):
        """
        acutually calculate the mann kendall from the sarray, this should be the only thing that needs
        to be updated for the seasonal kendall
        :return:
        """

        for sp in self.all_start_points:
            start = 0
            for i in range(self.nparts):
                if i == self.nparts - 1:
                    end = self.n
                else:
                    end = sp[i]
                data = (*sp,
                        *_seasonal_mann_kendall_from_sarray(self.x[start:end], alpha=self.alpha,
                                                            season_data=self.season_data[start:end],
                                                            sarray=self.s_array[start:end,
                                                                                start:end]))  # and passing the s array
                self.datasets[f'p{i}'].append(data)
                start = end
        for part in range(self.nparts):
            self.datasets[f'p{part}'] = pd.DataFrame(self.datasets[f'p{part}'],
                                                     columns=[f'split_point_{i}' for i in range(1, self.nparts)]
                                                             + ['trend', 'h', 'p', 'z', 's', 'var_s'])

    def _calc_senslope(self, data):
        senslope, senintercept, lo_slope, lo_intercept = _calc_seasonal_senslope(data[self.data_col],
                                                                                 data[self.season_col],
                                                                                 x=data.index, alpha=self.alpha)
        return senslope, senintercept
