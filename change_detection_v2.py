"""
simplification of Mike's code (utils.py power_sims) to propagate the uncertainty from various assumptions to the stats
power calcs
created matt_dumont
on: 18/05/23
"""
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import logging
from exponential_piston_flow import binary_exp_piston_flow_cdf, get_source_initial_conc_bepm
import multiprocessing
import os
import psutil
import sys


class DetectionPowerCalculator:
    implemented_mrt_models = (
        'piston_flow',
        'binary_exponential_piston_flow',
        'pass_true_conc',
    )
    implemented_significance_modes = (
        'linear-regression',
    )

    def __init__(self, significance_mode='linear-regression', nsims=1000, min_p_value=0.05, min_samples=10, ncores=None,
                 log_level=logging.INFO):
        """
        
        :param nsims: number of noise simulations to run for each change detection (e.g. nsims=1000, 
                      power= number of detected changes/1000 noise simulations) 
        :param min_p_value: minimum p value to consider a change detected 
        :param min_samples: minimum number of samples required, less than this number of samples will raise an exception
        :param ncores: number of cores to use for multiprocessing, None will use all available cores
        :param log_level: logging level for multiprocessing subprocesses
        """
        assert significance_mode in self.implemented_significance_modes, (f'significance_mode {significance_mode} not '
                                                                          f'implemented, must be one of '
                                                                          f'{self.implemented_significance_modes}')
        if significance_mode == 'linear-regression':
            self.power_test = self._power_test_lr
        else:
            raise NotImplementedError(f'significance_mode {significance_mode} not implemented, shouldnt get here')
        assert isinstance(nsims, int), 'nsims must be an integer'
        self.nsims = nsims
        assert isinstance(min_samples, int), 'min_samples must be an integer'
        assert min_samples >= 3, ('min_samples must be at least 3 otherwise the slope regresion will either'
                                  'fail or be meaningless')
        self.min_samples = min_samples
        self.min_p_value = min_p_value
        assert self.min_p_value > 0 and self.min_p_value < 1, 'min_p_value must be between 0 and 1'
        assert isinstance(ncores, int) or ncores is None, 'ncores must be an integer or None'
        self.ncores = ncores
        assert log_level in [logging.CRITICAL, logging.FATAL, logging.ERROR, logging.WARNING, logging.WARN,
                             logging.INFO, logging.DEBUG], f'unknown log_level {log_level}'
        self.log_level = log_level

    @staticmethod
    def truets_from_binary_exp_piston_flow(mrt, mrt_p1, frac_p1, f_p1, f_p2,
                                           initial_conc, target_conc, prev_slope, max_conc, min_conc,
                                           samp_per_year, samp_years, implementation_time, past_source_data=None,
                                           return_extras=False, low_mem=False,
                                           precision=2):  # todo this seems really slow optimise
        """
        create a true concentration time series using binary piston flow model for the mean residence time
        :param mrt: mean residence time years
        :param mrt_p1: mean residence time of the first pathway years
        :param frac_p1: fraction of the first pathway
        :param f_p1:  ratio of the exponential volume to the total volume pathway 1
        :param f_p2:  ratio of the exponential volume to the total volume pathway 2
        :param initial_conc: initial concentration
        :param target_conc: target concentration
        :param prev_slope: previous slope of the concentration data
        :param max_conc: maximum concentration limit user specified or None
                         here the maximum concentration is specified as the maximum concentration
                         of the source (before temporal mixing)
        :param min_conc: minimum concentration limit user specified, the lowest concentration for the source
        :param samp_per_year: samples per year
        :param samp_years: number of years to sample
        :param implementation_time: number of years to implement reductions
        :param past_source_data: past source data, if None will use the initial concentration and the previous slope
                                 to estimate the past source data, this is only set as an option to allow users to
                                 preclude re-running the source data calculations if they have already been done
                                 so.  Suggest that users only pass results from get_source_initial_conc_bepm with
                                 age_step = 0.01
        :param return_extras: return extra variables for debugging
        :return: true timeseries, max_conc, max_conc_time, frac_p2
        """
        assert isinstance(precision, int), 'precision must be an integer'
        age_step = 10 ** -precision
        if frac_p1 < 1:
            mrt_p2 = (mrt - (mrt_p1 * frac_p1)) / (1 - frac_p1)
        elif frac_p1 == 1:
            assert mrt_p1 == mrt, 'if frac_p1 == 1, mrt_p1 must equal mrt'
            mrt_p2 = np.nan
        else:
            raise ValueError(f'frac_1 must be between 0 and 1, not {frac_p1}')
        if mrt_p2 < 0:
            raise ValueError(f'mrt_p2 must be greater than 0, when calculated from {mrt=}, {mrt_p1=}, and {frac_p1=}'
                             f'got {mrt_p2=}')
        # make cdf of age

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
                source_conc_past = get_source_initial_conc_bepm(initial_conc, mrt_p1, mrt_p2, age_step, ages,
                                                                age_fractions,
                                                                prev_slope, max_conc, min_conc,
                                                                make_past_conc=return_extras, precision=precision)
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
        """ piston flow model for the mean residence time
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

    def power_calc(self,
                   idv,
                   error: float,
                   mrt_model: str,
                   samp_years: {int, None} = None,
                   samp_per_year: {int, None} = None,
                   implementation_time: {int, None} = None,
                   initial_conc: {float, None} = None,
                   target_conc: {float, None} = None,
                   prev_slope: {float, None} = None,
                   max_conc: {float, None} = None,
                   min_conc: {float, None} = None,
                   mrt: {float, None} = None,
                   # options for binary_exponential_piston_flow model
                   mrt_p1: {float, None} = None,
                   frac_p1: {float, None} = None,
                   f_p1: {float, None} = None,
                   f_p2: {float, None} = None,
                   # options for the pass_true_conc_ts model
                   true_conc_ts: {np.ndarray, None} = None,
                   seed: {int, None} = 5585
                   ):
        """

        :param idv: identifiers for the power calc sites, passed straight through to the output
        :param error: standard deviation of the noise
        :param mrt_model: the model to use for the mean residence time options:
                          * 'piston_flow': use the piston flow model (no mixing, default)
                          * 'binary_exponential_piston_flow': use the binary exponential piston flow model
                          for unitary exponential_piston_flow model set frac_1 = 1 and mrt_p1 = mrt
                          * 'pass_true_conc': pass the true concentration through the model via the
                            "true_conc_ts" kwarg only the 'error', 'mrt, and 'seed' kwargs are used all others must
                            be None. sampling timing and frequency is defined by the points in the true_conc_ts
        :param samp_years: number of years to sample
        :param samp_per_year: number of samples to collect each year
        :param implementation_time: number of years over which reductions are implemented
        :param initial_conc: inital median value of the concentration
        :param target_conc: target concentration to reduce to
        :param prev_slope: slope of the previous data (e.g. prior to the initial concentration)
        :param max_conc: maximum concentration limit user specified or None (default)
        :param min_conc: minimum concentration limit for the source, only used for the
                         binary_exponential_piston_flow model)
        :param mrt: the mean residence time of the site
        :param mrt_p1: the mean residence time of the first piston flow model (only used for
                        binary_exponential_piston_flow model)
        :param frac_p1: the fraction of the first piston flow model (only used for
                        binary_exponential_piston_flow model)
        :param f_p1: the fraction of the first piston flow model (only used for
                        binary_exponential_piston_flow model)
        :param f_p2: the fraction of the first piston flow model (only used for
                        binary_exponential_piston_flow model)
        :param true_conc_ts: the true concentration timeseries (only used for
                        pass_true_conc model)
        :param seed: int or None for random seed
        :return: pd.DataFrame with the power calc results (len=1) note power is percent 0-100
        """

        assert mrt_model in self.implemented_mrt_models, f'mrt_model must be one of: {self.implemented_mrt_models}'
        if mrt_model != 'pass_true_conc':
            assert pd.api.types.is_integer(
                samp_years), 'samp_years must be an integer unless mrt_model="pass_true_conc"'
            assert pd.api.types.is_integer(
                samp_per_year), 'samp_per_year must be an integer unless mrt_model="pass_true_conc"'
            assert pd.api.types.is_number(
                initial_conc), 'initial_conc must be a number unless mrt_model="pass_true_conc"'
            assert pd.api.types.is_number(target_conc), 'target_conc must be a number unless mrt_model="pass_true_conc"'
            assert pd.api.types.is_number(prev_slope), 'prev_slope must be a number unless mrt_model="pass_true_conc"'
            assert pd.api.types.is_number(max_conc), 'max_conc must be a number unless mrt_model="pass_true_conc"'
            assert max_conc >= initial_conc, 'max_conc must be greater than or equal to initial_conc'
            assert max_conc >= target_conc, 'max_conc must be greater than or equal to target_conc'
            assert pd.api.types.is_integer(implementation_time)

        # mange lag
        if mrt_model == 'piston_flow':

            assert true_conc_ts is None, 'true_conc_ts must be None for piston_flow model'
            true_conc_ts, max_conc_val, max_conc_time, mrt_p2 = self.truets_from_piston_flow(mrt,
                                                                                             initial_conc, target_conc,
                                                                                             prev_slope, max_conc,
                                                                                             samp_per_year, samp_years,
                                                                                             implementation_time)
            expect_slope = (target_conc - initial_conc) / implementation_time
        elif mrt_model == 'binary_exponential_piston_flow':
            assert true_conc_ts is None, 'true_conc_ts must be None for binary_exponential_piston_flow model'
            tvs = ['mrt_p1', 'frac_p1', 'f_p1', 'f_p2', 'min_conc']
            bad = []
            for t in tvs:
                if eval(t) is None:
                    bad.append(t)
            if len(bad) > 0:
                raise ValueError(f'for binary_exponential_piston_flow model the following must be specified: {bad}')

            (true_conc_ts, max_conc_val,
             max_conc_time, mrt_p2) = self.truets_from_binary_exp_piston_flow(
                mrt, mrt_p1, frac_p1, f_p1, f_p2,
                initial_conc, target_conc, prev_slope, max_conc, min_conc,
                samp_per_year, samp_years, implementation_time,
                return_extras=False)
            expect_slope = (target_conc - initial_conc) / implementation_time
        elif mrt_model == 'pass_true_conc':
            assert true_conc_ts is not None, 'true_conc_ts must be specified for pass_true_conc model'
            none_params = [
                'samp_years', 'samp_per_year', 'implementation_time', 'initial_conc', 'target_conc', 'prev_slope',
                'max_conc', 'min_conc', 'mrt_p1', 'frac_p1', 'f_p1', 'f_p2',
            ]
            for k in none_params:
                assert eval(k) is None, f'{k} must be None for pass_true_conc model'
            max_conc_val = np.max(true_conc_ts)
            max_conc_time = None
            mrt_p2 = None
            expect_slope = stats.linregress(true_conc_ts.index, true_conc_ts.values).slope

        else:
            raise NotImplementedError(f'mrt_model {mrt_model} not currently implemented')

        nsamples = len(true_conc_ts)
        if nsamples < self.min_samples:
            raise ValueError(f'nsamples must be greater than {self.min_samples}, you can change the '
                             f'minimum number of samples in the DetectionPowerCalculator class init')
        # tile to nsims
        rand_shape = (self.nsims, nsamples)
        conc_with_noise = np.tile(true_conc_ts, self.nsims).reshape(rand_shape)

        # generate noise
        np.random.seed(seed)
        all_seeds = list(np.random.randint(21, 54762438, 2))
        np.random.seed(all_seeds.pop(0))
        noise = np.random.normal(0, error, rand_shape)
        conc_with_noise += noise

        # run slope test
        power = self.power_test(conc_with_noise,
                                expected_slope=expect_slope,  # just used for sign
                                )

        out = pd.Series({'idv': idv,
                         'power': power,
                         'max_conc': max_conc_val,
                         'max_conc_time': max_conc_time,
                         'error': error,
                         'mrt_model': mrt_model,
                         'samp_years': samp_years,
                         'samp_per_year': samp_per_year,
                         'implementation_time': implementation_time,
                         'initial_conc': initial_conc,
                         'target_conc': target_conc,
                         'previous_slope': prev_slope,
                         'max_conc_lim': max_conc,
                         'min_conc_lim': min_conc,
                         'mrt': mrt,
                         'mrt_p1': mrt_p1,
                         'frac_p1': frac_p1,
                         'f_p1': f_p1,
                         'f_p2': f_p2,
                         'seed': seed,
                         'mrt_p2': mrt_p2,
                         'python_error': None
                         })

        return out

    def _power_test_lr(self, y, expected_slope):  # todo add a mann kendal test option, and a ttest option?
        """
        power calculations, probability of detecting a change (slope is significant and in the correct direction)
        :param y: np.array of shape (nsims, n_samples)
        :param expected_slope: used to determine sign of slope predicted is same as expected
        :return:
        """
        n_sims, n_samples = y.shape
        x = np.arange(n_samples)
        p_list = []
        for y in y:
            o2 = stats.linregress(x, y)
            sign_corr = np.sign(o2.slope) == np.sign(expected_slope)
            p_corr = (o2.pvalue < self.min_p_value)
            p_list.append(p_corr & sign_corr)
        power = sum(p_list) / n_sims * 100

        return power

    def _power_calc_mp(self, kwargs):
        """
        multiprocessing wrapper for power_calc
        :param kwargs:
        :return:
        """
        try:
            out = self.power_calc(**kwargs)
        except Exception:
            # capture kwargs to make debugging easier
            out = {
                'idv': kwargs['idv'],
                'python_error': traceback.format_exc(),
                'true_conc_ts_none': kwargs['true_conc_ts'] is None,
            }
            for k in kwargs:
                if k not in ['true_conc_ts', 'idv']:
                    out[k] = kwargs[k]
        out = pd.Series(out)
        return out

    @staticmethod
    def _get_id_str(val, name, float_percision=1):
        """
        helper function to get a string for an idv used to reduce the number of runs for multiprocessing workload
        :param val:
        :param name:
        :param float_percision:
        :return:
        """
        if val is None:
            return f'{name}=None'
        else:
            if pd.api.types.is_float(val):
                return f'{name}={val:.{float_percision}f}'
            return f'{name}={val}'

    @staticmethod
    def _adjust_shape(x, shape, none_allowed, is_int, idv):
        """
        helper function to adjust the shape of an input variable
        :param x: input variable
        :param shape: shape needed
        :param none_allowed: Is None allowed as a value
        :param is_int: is it an integer
        :param idv: str name of the input variable for error messages
        :return:
        """
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
    def _check_propogate_truets(x, shape):
        if x is None:
            return np.full(shape, None)
        else:
            len_x = len(x)
            assert len_x == shape[0], f'wrong_shape for true_conc_ts_vals must have len {shape[0]} got: shp {len_x}'
            return x

    def mulitprocess_power_calcs(
            self,
            outpath: {Path, None, str},
            id_vals: np.ndarray,
            error_vals: {np.ndarray, float},
            samp_years_vals: {np.ndarray, int, None} = None,
            samp_per_year_vals: {np.ndarray, int, None} = None,
            implementation_time_vals: {np.ndarray, int, None} = None,
            initial_conc_vals: {np.ndarray, float, None} = None,
            target_conc_vals: {np.ndarray, float, None} = None,
            previous_slope_vals: {np.ndarray, float, None} = None,
            max_conc_vals: {np.ndarray, float, None} = None,
            min_conc_vals: {np.ndarray, float, None} = None,
            mrt_model_vals: {np.ndarray, str} = 'piston_flow',
            mrt_vals: {np.ndarray, float} = 0.0,
            mrt_p1_vals: {np.ndarray, float, None} = None,
            frac_p1_vals: {np.ndarray, float, None} = None,
            f_p1_vals: {np.ndarray, float, None} = None,
            f_p2_vals: {np.ndarray, float, None} = None,
            true_conc_ts_vals: {np.ndarray, list, None} = None,
            seed: {np.ndarray, int, None} = 5585,
            run=True,
    ):
        """
        multiprocessing wrapper for power_calc, see power_calc for details
        note that if a given run raises and exception the traceback for the exception will be included in the
        returned dataset under the column 'python_error' if 'python_error' is None then the run was successful
        to change the number of cores used pass n_cores to the constructor init

        :param outpath: path to save results to
        :param id_vals: id values for each simulation
        All values from here on out should be either a single value or an array of values with the same shape as id_vals
        :param error_vals: standard deviation of noise to add for each simulation
        :param samp_years_vals: number of years to sample for each simulation
        :param samp_per_year_vals: number of samples per year for each simulation
        :param implementation_time_vals: the length of implementation of reduction for each simulation
        :param initial_conc_vals: the initial concentration for each simulation
        :param target_conc_vals: the target concentration for each simulation
        :param previous_slope_vals: the previous slope of the source concentration
               for each simulation units conc/yr
        :param max_conc_vals: the maximum concentration for each simulation if None then no maximum
                              (based solely on the previous slope and lag time)
        :param min_conc_vals: the minimum concentration for each simulation only used for exponential piston flow
        :param mrt_model_vals: the mrt model to use for each simulation
        :param mrt_vals: the mean residence time for each simulation
        :param mrt_p1_vals: the mean residence time for the first piston for each simulation (only used for
                            bianary exponential piston flow)
        :param frac_p1_vals:  the fraction of the total mass in the first piston for each simulation (only used for
                              bianary exponential piston flow)
                              set to 1 for an exponential piston flow model, otherwise set to None
        :param f_p1_vals: the fraction of the total mass in the first piston for each simulation (only used for
                            bianary exponential piston flow) otherwise set to None
        :param f_p2_vals: the fraction of the total mass in the first piston for each simulation (only used for
                            bianary exponential piston flow) otherwise set to None
        :param true_conc_ts_vals: the true concentration time series for each simulation only used for the
                                    'pass_true_conc' mrt_model, note that this can be a list of arrays of different
                                    lengths for each simulation, Numpy does not support jagged arrays
        :param seed: ndarray (integer seeds), None (no seeds), or int (1 seed for all simulations)
        :param run: if True run the simulations, if False just build  the run_dict and print the number of simulations
        :return: dataframe with input data and the results of all of the power calcs. note power is percent 0-100
        """
        if isinstance(outpath, str):
            outpath = Path(outpath)
        id_vals = np.atleast_1d(id_vals)
        expect_shape = id_vals.shape

        if isinstance(mrt_model_vals, str):
            mrt_model_vals = np.array([mrt_model_vals] * len(id_vals))
        mrt_model_vals = np.atleast_1d(mrt_model_vals)
        assert mrt_model_vals.shape == id_vals.shape, f'mrt_model_vals and mrt_vals must have the same shape'
        assert np.in1d(mrt_model_vals, self.implemented_mrt_models).all(), (
            f'mrt_model_vals must be one of {self.implemented_mrt_models} '
            f'got {np.unique(mrt_model_vals)}')

        # manage multiple size options
        error_vals = self._adjust_shape(error_vals, expect_shape, none_allowed=False, is_int=False,
                                        idv='error_vals')
        samp_years_vals = self._adjust_shape(samp_years_vals, expect_shape, none_allowed=True, is_int=True,
                                             idv='samp_years_vals')
        samp_per_year_vals = self._adjust_shape(samp_per_year_vals, expect_shape, none_allowed=True, is_int=True,
                                                idv='samp_per_year_vals')
        implementation_time_vals = self._adjust_shape(implementation_time_vals, expect_shape, none_allowed=True,
                                                      is_int=True,
                                                      idv='implementation_time_vals')
        initial_conc_vals = self._adjust_shape(initial_conc_vals, expect_shape, none_allowed=True, is_int=False,
                                               idv='initial_conc_vals')
        target_conc_vals = self._adjust_shape(target_conc_vals, expect_shape, none_allowed=True, is_int=False,
                                              idv='target_conc_vals')
        previous_slope_vals = self._adjust_shape(previous_slope_vals, expect_shape, none_allowed=True, is_int=False,
                                                 idv='previous_slope_vals')
        max_conc_vals = self._adjust_shape(max_conc_vals, expect_shape, none_allowed=True, is_int=False,
                                           idv='max_conc_vals')
        min_conc_vals = self._adjust_shape(min_conc_vals, expect_shape, none_allowed=True, is_int=False,
                                           idv='min_conc_vals')
        mrt_vals = self._adjust_shape(mrt_vals, expect_shape, none_allowed=True, is_int=False, idv='mrt_vals')
        mrt_p1_vals = self._adjust_shape(mrt_p1_vals, expect_shape, none_allowed=True, is_int=False, idv='mrt_p1_vals')
        frac_p1_vals = self._adjust_shape(frac_p1_vals, expect_shape, none_allowed=True, is_int=False,
                                          idv='frac_p1_vals')
        f_p1_vals = self._adjust_shape(f_p1_vals, expect_shape, none_allowed=True, is_int=False, idv='f_p1_vals')
        f_p2_vals = self._adjust_shape(f_p2_vals, expect_shape, none_allowed=True, is_int=False, idv='f_p2_vals')
        true_conc_ts_vals = self._check_propogate_truets(true_conc_ts_vals, expect_shape)
        use_seeds = self._adjust_shape(seed, expect_shape, none_allowed=True, is_int=True, idv='seed')
        not_na_idx = pd.notna(max_conc_vals) & pd.notna(initial_conc_vals)
        assert (max_conc_vals[not_na_idx] >= initial_conc_vals[
            not_na_idx]).all(), 'max_conc must be greater than or equal to initial_conc'
        not_na_idx = pd.notna(max_conc_vals) & pd.notna(target_conc_vals)
        assert (max_conc_vals[not_na_idx] >= target_conc_vals[
            not_na_idx]).all(), 'max_conc must be greater than or equal to target_conc'

        run_vals = [
            id_vals,
            error_vals,
            samp_years_vals,
            samp_per_year_vals,
            implementation_time_vals,
            initial_conc_vals,
            target_conc_vals,
            previous_slope_vals,
            max_conc_vals,
            min_conc_vals,
            mrt_model_vals,
            mrt_vals,
            mrt_p1_vals,
            frac_p1_vals,
            f_p1_vals,
            f_p2_vals,
            true_conc_ts_vals,
            use_seeds

        ]
        runs = []
        run_list = []
        all_use_idv = []
        # make runs
        print('creating and condensing runs')
        for i, (
                idv, error, samp_years, samp_per_year, implementation_time,
                inital_conc, target_conc, previous_slope, max_conc, min_conc,
                mrt_model, mrt, mrt_p1, frac_p1, f_p1, f_p2, true_conc_ts, seed_val
        ) in enumerate(zip(*run_vals)):
            # create unique value so that if multiple sites are passed with the same values (within rounding error)
            # they are only run once
            if true_conc_ts is None:
                ptis = None
            else:
                ptis = i
            if i % 1000 == 0:
                print(f'forming/condesing run {i} of {len(id_vals)}')

            use_idv = (
                self._get_id_str(error, 'error'),
                self._get_id_str(samp_years, 'samp_years'),
                self._get_id_str(samp_per_year, 'samp_per_year'),
                self._get_id_str(implementation_time, 'implementation_time'),
                self._get_id_str(inital_conc, 'inital_conc'),
                self._get_id_str(target_conc, 'target_conc'),
                self._get_id_str(previous_slope, 'previous_slope', 2),
                self._get_id_str(max_conc, 'max_conc'),
                self._get_id_str(min_conc, 'min_conc'),
                self._get_id_str(mrt_model, 'mrt_model'),
                self._get_id_str(mrt, 'mrt', 0),
                self._get_id_str(mrt_p1, 'mrt_p1'),
                self._get_id_str(frac_p1, 'frac_p1', 2),
                self._get_id_str(f_p1, 'f_p1', 2),
                self._get_id_str(f_p2, 'f_p2', 2),
                self._get_id_str(ptis, 'ptis'),
            )
            use_idv = '_'.join(use_idv)
            all_use_idv.append(use_idv)
            if use_idv in run_list:
                continue
            run_list.append(use_idv)

            runs.append(dict(
                idv=use_idv,
                error=error,
                samp_years=samp_years,
                samp_per_year=samp_per_year,
                implementation_time=implementation_time,
                initial_conc=inital_conc,
                target_conc=target_conc,
                prev_slope=previous_slope,
                max_conc=max_conc,
                min_conc=min_conc,
                mrt_model=mrt_model,
                mrt=mrt,
                mrt_p1=mrt_p1,
                frac_p1=frac_p1,
                f_p1=f_p1,
                f_p2=f_p2,
                true_conc_ts=true_conc_ts,
                seed=seed_val,

            ))

        # run
        print(f'running {len(runs)} runs, simplified from {len(id_vals)} runs')
        if not run:
            print(f'stopping as {run=}')
            return
        result_data = _run_multiprocess(self._power_calc_mp, runs, num_cores=self.ncores,
                                        logging_level=self.log_level)
        result_data = pd.DataFrame(result_data)
        result_data.set_index('idv', inplace=True)

        outdata = result_data.loc[all_use_idv]
        outdata.loc[:, 'idv'] = id_vals
        outdata.set_index('idv', inplace=True)

        if outpath is not None:
            outpath.parent.mkdir(parents=True, exist_ok=True)
            outdata.to_hdf(outpath, 'data')
        return outdata


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
