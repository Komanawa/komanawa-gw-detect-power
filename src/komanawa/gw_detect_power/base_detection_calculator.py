"""
created matt_dumont 
on: 25/01/24
"""
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import multiprocessing
import os
import psutil
import sys
import warnings

# handle import of optional dependencies
age_tools_imported = True
pyhomogeneity_imported = True
kendal_imported = True

try:
    from komanawa.gw_age_tools import binary_exp_piston_flow_cdf, predict_historical_source_conc, make_age_dist, check_age_inputs
except ImportError:
    binary_exp_piston_flow_cdf, get_source_initial_conc_bepm = None, None
    age_tools_imported = False
    warnings.warn(
        'age_tools not installed, age distribution related functions will be unavailable, to install run '
        'pip install git+https://github.com/Komanawa-Solutions-Ltd/gw_age_tools'
    )

try:
    from pyhomogeneity import pettitt_test
except ImportError:
    pettitt_test = None
    pyhomogeneity_imported = False
    warnings.warn(
        'pyhomogeneity not installed, pettitt_test will be unavailable, to install run '
        'pip install pyhomogeneity'
    )

try:
    from komanawa.kendall_stats import MannKendall, MultiPartKendall
except ImportError:
    MannKendall, MultiPartKendall = None, None
    kendal_imported = False
    warnings.warn(
        'kendall_stats not installed, mann_kendall will be unavailable, to install run '
        'pip install git+https://github.com/Komanawa-Solutions-Ltd/kendall_multipart_kendall.git'
    )


class BaseDetectionCalculator:
    """
    Base class for detection power calculations, provides some general methods for power calculations
    """
    implemented_mrt_models = None  # override in subclass
    _power_from_max = False  # override in subclass
    _power_from_min = False  # override in subclass
    implemented_significance_modes = None
    _auto_mode = False
    _counterfactual = False

    def __init__(self):
        raise NotImplementedError('must be implemented in subclass')

    def _round_kwarg_value(self, val, key):
        """
        helper function to round a kwarg value to the precision of the kwarg

        :param val: value to round
        :param key: kwarg name
        :return:
        """
        use_key = key.replace('_vals', '')
        if hasattr(self, f'{use_key}_per'):
            float_percision = getattr(self, f'{use_key}_per')
        else:
            float_percision = 9
        if val is None:
            return val
        else:
            if pd.api.types.is_float(val):
                return np.round(val, float_percision)
            return val

    def _get_id_str(self, val, name):
        """
        helper function to get a string for an idv used to reduce the number of runs for multiprocessing workload
        :param val:
        :param name:
        :param float_percision:
        :return:
        """
        name = name.replace('_vals', '')
        if hasattr(self, f'{name}_per'):
            float_percision = getattr(self, f'{name}_per')
        else:
            float_percision = 9

        if val is None:
            return f'{name}=None'
        else:
            if pd.api.types.is_float(val):
                return f'{name}={val:.{float_percision}f}'
            return f'{name}={val}'

    def _get_key_info(self, key):

        base_data = dict(
            error_vals=(False, False, False),
            error_base_vals=(False, False, False),
            error_alt_vals=(True, False, False),
            seed_alt_vals_vals=(True, True, False),
            seed_base_vals_vals=(True, True, False),
            seed=(True, True, False),
        )

        auto_data = dict(
            samp_years_vals=(False, True, False),
            samp_per_year_vals=(False, True, False),
            implementation_time_vals=(False, True, False),
            initial_conc_vals=(False, False, False),
            target_conc_vals=(False, False, False),
            prev_slope_vals=(False, False, False),
            max_conc_lim_vals=(False, False, False),
            min_conc_lim_vals=(False, False, False),
            mrt_vals=(False, False, False),
            mrt_p1_vals=(True, False, False),
            frac_p1_vals=(True, False, False),
            f_p1_vals=(True, False, False),
            f_p2_vals=(True, False, False),
        )
        if self._auto_mode:
            data = {**base_data, **auto_data}
        else:
            data = base_data

        none_allowed, is_int, is_any = data.get(key, (False, False, True))

        return none_allowed, is_int, is_any

    def _run_multiprocess_pass_conc(self, outpath, idv_vals, run, debug_mode, use_kwargs):
        outpath, idv_vals, use_kwargs = self._multiprocess_checks(outpath, idv_vals, **use_kwargs)

        runs = []
        for i, idv in enumerate(idv_vals):
            kwargs = {k.replace('_vals', ''): v[i] for k, v in use_kwargs.items()}
            kwargs['idv'] = idv
            runs.append(kwargs)

        print(f'running {len(runs)} runs')
        if not run:
            print(f'stopping as {run=}')
            return

        if debug_mode:
            result_data = []
            for run_kwargs in runs:
                print(f'running power calc for: {run_kwargs["idv"]} with debug_mode=True (single process)')
                result_data.append(self.power_calc(**run_kwargs))

        else:
            result_data = _run_multiprocess(self._power_calc_mp, runs, num_cores=self.ncores,
                                            logging_level=self.log_level)
        result_data = pd.DataFrame(result_data)
        result_data.set_index('idv', inplace=True)

        if outpath is not None:
            print(f'saving results to: {outpath}')
            outpath.parent.mkdir(parents=True, exist_ok=True)
            result_data.to_hdf(outpath, 'data')
        return result_data

    def _run_multiprocess_auto(self, outpath, idv_vals, run, debug_mode, use_kwargs):
        outpath, idv_vals, use_kwargs = self._multiprocess_checks(outpath, idv_vals, **use_kwargs)

        # make runs
        runs = []

        if self.condensed_mode:
            if self._counterfactual:
                identifiers = ['error_base', 'error_alt', 'samp_years', 'samp_per_year', 'implementation_time_base',
                               'implementation_time_alt', 'delay_years',
                               'initial_conc',
                               'target_conc_base', 'target_conc_alt', 'prev_slope', 'max_conc_lim', 'min_conc_lim',
                               'mrt_model',
                               'mrt', 'mrt_p1', 'frac_p1', 'f_p1', 'f_p2', 'seed_alt', 'seed_base']
            else:
                identifiers = ['error', 'samp_years', 'samp_per_year', 'implementation_time', 'initial_conc',
                               'target_conc', 'prev_slope', 'max_conc_lim', 'min_conc_lim', 'mrt_model',
                               'mrt', 'mrt_p1', 'frac_p1', 'f_p1', 'f_p2', 'seed']

            all_use_idv = []
            run_list = []
            print('creating and condensing runs')
            use_kwargs = {k: self._round_kwarg_value(v, k) for k, v in use_kwargs.items()}

            for i, idv in enumerate(idv_vals):
                if i % 1000 == 0:
                    print(f'forming/condesing run {i} of {len(idv_vals)}')

                use_idv = '_'.join([self._get_id_str(use_kwargs[e + '_vals'][i], e) for e in identifiers])
                all_use_idv.append(use_idv)
                if use_idv in run_list:
                    continue
                run_list.append(use_idv)
                kwargs = {k.replace('_vals', ''): v[i] for k, v in use_kwargs.items()}
                kwargs['idv'] = use_idv
                runs.append(kwargs)

        else:
            all_use_idv = idv_vals
            for i, idv in enumerate(idv_vals):
                kwargs = {k.replace('_vals', ''): v[i] for k, v in use_kwargs.items()}
                kwargs['idv'] = idv
                runs.append(kwargs)

        if self.condensed_mode:
            print(f'running {len(runs)} runs, simplified from {len(idv_vals)} runs')
        else:
            print(f'running {len(runs)} runs')

        if not run:
            print(f'stopping as {run=}')
            return

        if debug_mode:
            result_data = []
            for run_kwargs in runs:
                print(f'running power calc for: {run_kwargs["idv"]} with debug_mode=True (single process)')
                result_data.append(self.power_calc(**run_kwargs))

        else:
            result_data = _run_multiprocess(self._power_calc_mp, runs, num_cores=self.ncores,
                                            logging_level=self.log_level)
        result_data = pd.DataFrame(result_data)
        result_data.set_index('idv', inplace=True)

        outdata = result_data.loc[all_use_idv]
        outdata.loc[:, 'idv'] = idv_vals
        outdata.set_index('idv', inplace=True, drop=True)

        if outpath is not None:
            outpath.parent.mkdir(parents=True, exist_ok=True)
            outdata.to_hdf(outpath, 'data')
        return outdata

    @staticmethod
    def _adjust_shape(x, shape, none_allowed, is_int, idv, any_val=False):
        """
        helper function to adjust the shape of an input variable
        :param x: input variable
        :param shape: shape needed
        :param none_allowed: Is None allowed as a value
        :param is_int: is it an integer
        :param idv: str name of the input variable for error messages
        :param any_val: if True then any value is allowed
        :return:
        """
        if any_val:
            if hasattr(x, '__iter__') and not isinstance(x, str):
                x = np.atleast_1d(x)
                assert len(x) == shape[0], (f'wrong_shape for {idv} must have shape {shape} or not be iterable '
                                            f'got: shp {x.shape} dtype {x.dtype}')
            else:
                x = np.full(shape, x)
            return x

        if x is None and none_allowed:
            x = np.full(shape, None)
            return x

        if is_int:
            if pd.api.types.is_integer(x):
                x = np.full(shape, x, dtype=int)
            else:
                x = np.atleast_1d(x)
                assert x.shape == shape, (f'wrong_shape for {idv} must have shape {shape} '
                                          f'got: shp {x.shape} dtype {x.dtype}')
                not_bad = [e is None or pd.api.types.is_integer(e) for e in x]
                assert all(not_bad), (f'{idv} must be an integer or None got {x[~np.array(not_bad)]} '
                                      f'at indices {np.where(~np.array(not_bad))[0]}')
        else:
            if pd.api.types.is_number(x):
                x = np.full(shape, x).astype(float)
            else:
                x = np.atleast_1d(x)
                assert x.shape == shape, (f'wrong_shape for {idv} must be a float or have shape {shape} '
                                          f'got: shp {x.shape} dtype {x.dtype}')
                not_bad = [e is None or pd.api.types.is_number(e) for e in x]
                assert all(not_bad), (f'{idv} must be an number or None got {x[~np.array(not_bad)]} '
                                      f'at indices {np.where(~np.array(not_bad))[0]}')
        return x

    @staticmethod
    def truets_from_binary_exp_piston_flow(mrt, mrt_p1, frac_p1, f_p1, f_p2,
                                           initial_conc, target_conc, prev_slope, max_conc, min_conc,
                                           samp_per_year, samp_years, implementation_time, past_source_data=None,
                                           return_extras=False, low_mem=False,
                                           precision=2):
        """
        create a true concentration time series using binary piston flow model for the mean residence time note that this can be really slow for large runs and it may be better to create the source data first and then pass it to the power calcs via pass_true_conc

        :param mrt: mean residence time years
        :param mrt_p1: mean residence time of the first pathway years
        :param frac_p1: fraction of the first pathway
        :param f_p1:  ratio of the exponential volume to the total volume pathway 1
        :param f_p2:  ratio of the exponential volume to the total volume pathway 2
        :param initial_conc: initial concentration
        :param target_conc: target concentration
        :param prev_slope: previous slope of the concentration data
        :param max_conc: maximum concentration limit user specified or None here the maximum concentration is specified as the maximum concentration of the source (before temporal mixing)
        :param min_conc: minimum concentration limit user specified, the lowest concentration for the source
        :param samp_per_year: samples per year
        :param samp_years: number of years to sample
        :param implementation_time: number of years to implement reductions
        :param past_source_data: past source data, if None will use the initial concentration and the previous slope to estimate the past source data, this is only set as an option to allow users to preclude re-running the source data calculations if they have already been done so.  Suggest that users only pass results from get_source_initial_conc_bepm with age_step = 0.01
        :param return_extras: return extra variables for debugging
        :return: true timeseries, max_conc, max_conc_time, frac_p2
        """
        mrt, mrt_p2 = check_age_inputs(mrt=mrt, mrt_p1=mrt_p1, mrt_p2=None, frac_p1=frac_p1,
                                       precision=precision, f_p1=f_p1, f_p2=f_p2)
        # make cdf of age
        age_step, ages, age_fractions = make_age_dist(mrt, mrt_p1, mrt_p2, frac_p1, precision, f_p1, f_p2, start=np.nan)

        ages = np.arange(0, np.nanmax([mrt_p1, mrt_p2]) * 5, age_step).round(precision)  # approximately monthly steps
        age_cdf = binary_exp_piston_flow_cdf(ages, mrt_p1, mrt_p2, frac_p1, f_p1, f_p2)
        age_fractions = np.diff(age_cdf, prepend=0)

        # make historical source concentrations from prev_slope, initial_conc, max_conc
        if past_source_data is not None:
            source_conc_past = past_source_data.sort_index()
            use_max_conc = source_conc_past.iloc[-1]
        else:
            if prev_slope == 0:
                hist_ages = np.arange(0., np.nanmax([mrt_p1, mrt_p2]) * 5 * 2 + age_step, age_step).round(precision)
                source_conc_past = pd.Series(index=hist_ages * -1, data=np.ones(len(hist_ages)) * initial_conc)
                use_max_conc = initial_conc
            else:
                # make a historical source timeseries from preivous slope, inital conc age pdf and max conc
                source_conc_past = predict_historical_source_conc(init_conc=initial_conc,
                                                                  mrt=mrt, mrt_p1=mrt_p1, mrt_p2=mrt_p2,
                                                                  frac_p1=frac_p1, f_p1=f_p1, f_p2=f_p2,
                                                                  prev_slope=prev_slope, max_conc=max_conc,
                                                                  min_conc=min_conc, start_age=np.nan,
                                                                  precision=precision)

                source_conc_past = source_conc_past.sort_index()
                use_max_conc = source_conc_past.iloc[-1]

        # make a future source timeseries from target_conc and implementation_time

        if low_mem:
            fut_idx = np.arange(0, max(implementation_time, samp_years) + 1, 1)
        else:
            fut_idx = np.arange(0, max(implementation_time, samp_years) + age_step, age_step).round(precision)

        future_conc = pd.Series(
            index=fut_idx,
            data=np.nan
        )
        future_conc[future_conc.index >= implementation_time] = target_conc
        future_conc[0] = use_max_conc
        future_conc = future_conc.interpolate(method='linear')
        future_conc = future_conc.sort_index()
        total_source_conc = pd.concat([source_conc_past.drop(index=0), future_conc]).sort_index()

        # sample the source concentration onto the age pdf to return the true timeseries
        out_years = np.arange(0, samp_years, 1 / samp_per_year).round(precision)
        out_conc = np.full_like(out_years, np.nan)
        if low_mem:
            for i, t in enumerate(out_years):
                use_ages = (t - ages).round(precision)
                temp_out = total_source_conc.loc[use_ages.min() - 1:use_ages.max() + 2]
                temp_out = pd.concat((temp_out,
                                      pd.Series(index=use_ages[~np.in1d(use_ages, temp_out.index)], data=np.nan)))
                temp_out = temp_out.sort_index()
                temp_out = temp_out.interpolate(method='linear')
                out_conc[i] = (temp_out.loc[(t - ages).round(precision)] * age_fractions).sum()
        else:
            use_ages = np.repeat(ages[:, np.newaxis], len(out_years), axis=1)
            ags_shp = use_ages.shape
            use_ages = (out_years[np.newaxis] - use_ages).round(precision).flatten()
            out_conc = total_source_conc.loc[use_ages].values.reshape(ags_shp) * age_fractions[:, np.newaxis]
            out_conc = out_conc.sum(axis=0)

        max_conc_time = out_years[out_conc.argmax()]
        conc_max = out_conc.max()
        if return_extras:
            past_years = np.arange(ages.max() * -1, 0., 1 / samp_per_year)
            past_conc = np.full_like(past_years, np.nan)
            for i, t in enumerate(past_years):
                past_conc[i] = (total_source_conc.loc[(t - ages).round(precision)] * age_fractions).sum()
            past_conc = pd.Series(index=past_years, data=past_conc)

            return out_conc, conc_max, max_conc_time, mrt_p2, total_source_conc, age_fractions, out_years, ages, past_conc
        return out_conc, conc_max, max_conc_time, mrt_p2

    @staticmethod
    def truets_from_piston_flow(mrt, initial_conc, target_conc, prev_slope, max_conc, samp_per_year, samp_years,
                                implementation_time):
        """
        piston flow model for the mean residence time

        :param mrt: mean residence time
        :param initial_conc: initial concentration
        :param target_conc: target concentration
        :param prev_slope: previous slope of the concentration data mg/l/yr
        :param max_conc: maximum concentration limit user specified or None
        :param samp_per_year: samples per year
        :param samp_years: number of years to sample
        :param implementation_time: number of years to implement reductions
        :return: true timeseries, max_conc, max_conc_time, frac_p2
        """
        # expand from
        nsamples_imp = samp_per_year * implementation_time
        nsamples_total = samp_per_year * samp_years

        true_conc_ts = []

        # lag period
        if mrt >= 1:
            nsamples_lag = int(round(mrt * samp_per_year))
            temp = np.interp(np.arange(nsamples_lag),
                             [0, nsamples_lag - 1],
                             [initial_conc, initial_conc + prev_slope * mrt])
            if max_conc is not None:
                temp[temp > max_conc] = max_conc
            max_conc_time = np.argmax(temp) / samp_per_year
            max_conc = temp.max()
            true_conc_ts.append(temp)
        else:
            max_conc = initial_conc
            max_conc_time = 0
            nsamples_lag = 0

        # reduction_period
        true_conc_ts.append(
            np.interp(np.arange(nsamples_imp), [0, nsamples_imp - 1], [max_conc, target_conc]))

        if nsamples_total > (nsamples_lag + nsamples_imp):
            true_conc_ts.append(np.ones(nsamples_total - (nsamples_lag + nsamples_imp)) * target_conc)
        true_conc_ts = np.concatenate(true_conc_ts)
        true_conc_ts = true_conc_ts[:nsamples_total]

        frac_p2 = None  # dummy value
        return true_conc_ts, max_conc, max_conc_time, frac_p2

    def time_test_power_calc_itter(self, testnitter=10, **kwargs):
        """
        run a test power calc iteration to check for errors

        :param testnitter: number of iterations to run
        :param kwargs: kwargs for power_calc
        :return: None
        """
        t = time.time()
        use_action = 'default'
        for wv in warnings.filters:
            if str(wv[2]) == str(UserWarning):
                use_action = wv[0]
                break

        warnings.filterwarnings("ignore", category=UserWarning)
        for i in range(testnitter):
            self.power_calc(testnitter=testnitter, **kwargs)

        warnings.filterwarnings(use_action, category=UserWarning)
        temp = (time.time() - t) / testnitter
        print(f'time per iteration: {temp} s. based on {testnitter} iterations\n'
              f'with set number of iterations: {self.nsims} it will take {temp * self.nsims / 60} s to run the power calc')

    def _power_calc_mp(self, kwargs):
        """
        multiprocessing wrapper for power_calc
        :param kwargs:
        :return:
        """
        try:
            out = self.power_calc(**kwargs)
            if self.return_true_conc or self.return_noisy_conc_itters < 0:
                out = out['power']
        except Exception:
            # capture kwargs to make debugging easier
            out = {
                'idv': kwargs['idv'],
                'python_error': traceback.format_exc(),
            }
            for k in kwargs:
                if k not in ['true_conc_ts', 'true_conc_base', 'true_conc_alt', 'idv']:
                    out[k] = kwargs[k]
        out = pd.Series(out)
        return out

    @staticmethod
    def _check_propogate_truets(x, shape):
        if x is None:
            return np.full(shape, None)
        else:
            len_x = len(x)
            assert len_x == shape[0], f'wrong_shape for true_conc_ts_vals must have len {shape[0]} got: shp {len_x}'
            return x

    def _multiprocess_checks(self, outpath, id_vals, **kwargs):
        if self.return_true_conc or self.return_noisy_conc_itters > 0:
            warnings.warn('return_true_conc and return_noisy_conc_itters are not supported for mulitprocess_power_calcs'
                          'only power results will be returned')

        if isinstance(outpath, str):
            outpath = Path(outpath)
        id_vals = np.atleast_1d(id_vals)
        expect_shape = id_vals.shape

        # check other inputs
        for key, value in kwargs.items():
            if key in ['mrt_model_vals', 'true_conc_ts_vals', 'true_conc_base_vals', 'true_conc_alt_vals']:  # todo ensure test passes!
                continue
            none_allowed, is_int, is_any = self._get_key_info(key)
            temp = self._adjust_shape(value, expect_shape, none_allowed=none_allowed, is_int=is_int, idv=key,
                                      any_val=is_any)
            kwargs[key] = temp

        if self._auto_mode:
            mrt_model_vals = kwargs['mrt_model_vals']
            if isinstance(mrt_model_vals, str):
                mrt_model_vals = np.array([mrt_model_vals] * len(id_vals))
            mrt_model_vals = np.atleast_1d(mrt_model_vals)
            assert mrt_model_vals.shape == id_vals.shape, f'mrt_model_vals and mrt_vals must have the same shape'
            assert np.in1d(mrt_model_vals, self.implemented_mrt_models).all(), (
                f'mrt_model_vals must be one of {self.implemented_mrt_models} '
                f'got {np.unique(mrt_model_vals)}')
            if any(mrt_model_vals == 'binary_exponential_piston_flow'):
                assert age_tools_imported, (
                    'cannot run binary_exponential_piston_flow model, age_tools not installed'
                    'to install run:\n'
                    'pip install git+https://github.com/Komanawa-Solutions-Ltd/gw_age_tools')

            kwargs['mrt_model_vals'] = mrt_model_vals

            # check min/max values
            assert 'initial_conc_vals' in kwargs, 'if max_conc_lim_vals is passed initial_conc_vals must be passed'
            not_na_idx = pd.notna(kwargs['max_conc_lim_vals']) & pd.notna(kwargs['initial_conc_vals'])
            assert (kwargs['max_conc_lim_vals'][not_na_idx] >= kwargs['initial_conc_vals'][not_na_idx]).all(), (
                'max_conc_lim must be greater than or equal to initial_conc')

            if self._counterfactual:
                not_na_idx = pd.notna(kwargs['max_conc_lim_vals']) & pd.notna(kwargs['target_conc_base_vals'])
                assert (kwargs['max_conc_lim_vals'][not_na_idx] >= kwargs['target_conc_base_vals'][not_na_idx]).all(), (
                    'max_conc_lim must be greater than or equal to target_conc_base')
                not_na_idx = pd.notna(kwargs['max_conc_lim_vals']) & pd.notna(kwargs['target_conc_alt_vals'])
                assert (kwargs['max_conc_lim_vals'][not_na_idx] >= kwargs['target_conc_alt_vals'][not_na_idx]).all(), (
                    'max_conc_lim must be greater than or equal to target_conc_alt')

            else:
                not_na_idx = pd.notna(kwargs['max_conc_lim_vals']) & pd.notna(kwargs['target_conc_vals'])
                assert (kwargs['max_conc_lim_vals'][not_na_idx] >= kwargs['target_conc_vals'][not_na_idx]).all(), (
                    'max_conc_lim must be greater than or equal to target_conc')

        elif self._counterfactual:
            kwargs['true_conc_base_vals'] = self._check_propogate_truets(kwargs['true_conc_base_vals'], expect_shape)
            kwargs['true_conc_alt_vals'] = self._check_propogate_truets(kwargs['true_conc_alt_vals'], expect_shape)
        else:
            kwargs['true_conc_ts_vals'] = self._check_propogate_truets(kwargs['true_conc_ts_vals'], expect_shape)

        return outpath, id_vals, kwargs


def _run_multiprocess(func, runs, logical=True, num_cores=None, logging_level=logging.INFO):
    """
    count the number of processors and then instiute the runs of a function to
    :param func: function with one argument kwargs.
    :param runs: a list of runs to pass to the function the function is called via func(kwargs)
    :param num_cores: int or None, if None then use all cores (+-logical) if int, set pool size to number of cores
    :param logical: bool if True then add the logical processors to the count
    :param logging_level: logging level to use one of: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR,
                          logging.CRITICAL more info https://docs.python.org/3/howto/logging.html
                          default is logging.INFO
    :return:
    """
    assert isinstance(num_cores, int) or num_cores is None
    multiprocessing.log_to_stderr(logging_level)
    if num_cores is None:
        pool_size = psutil.cpu_count(logical=logical)
    else:
        pool_size = num_cores

    pool = multiprocessing.Pool(processes=pool_size,
                                initializer=_start_process,
                                )

    results = pool.map_async(func, runs)
    pool_outputs = results.get()
    pool.close()  # no more tasks
    pool.join()
    return pool_outputs


def _start_process():
    """
    function to run at the start of each multiprocess sets the priority lower
    :return:
    """
    print('Starting', multiprocessing.current_process().name)
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    if sys.platform == "linux":
        p.nice(19)
        # linux
    elif sys.platform == "darwin":
        # OS X
        p.nice(19)
    elif sys.platform == "win32":
        # Windows...
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        raise ValueError(f'unexpected platform: {sys.platform}')
