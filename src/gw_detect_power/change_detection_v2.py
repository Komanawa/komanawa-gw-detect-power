"""
simplification of Mike's code (utils.py power_sims) to propagate the uncertainty from various assumptions to the stats
power calcs
created matt_dumont
on: 18/05/23
"""
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import logging
import multiprocessing
import os
import psutil
import sys
import warnings
from gw_detect_power.base_detection_calculator import BaseDetectionCalculator, _run_multiprocess

# handle import of optional dependencies
age_tools_imported = True
pyhomogeneity_imported = True
kendal_imported = True

try:
    from gw_age_tools import binary_exp_piston_flow_cdf, predict_historical_source_conc, make_age_dist, check_age_inputs
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
    from kendall_stats import MannKendall, MultiPartKendall
except ImportError:
    MannKendall, MultiPartKendall = None, None
    kendal_imported = False
    warnings.warn(
        'kendall_stats not installed, mann_kendall will be unavailable, to install run '
        'pip install git+https://github.com/Komanawa-Solutions-Ltd/kendall_multipart_kendall.git'
    )


class DetectionPowerCalculator:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError  # Keynote as a place holder to refactor tests


class DetectionPowerSlope(BaseDetectionCalculator):
    implemented_mrt_models = ()
    implemented_significance_modes = (
        'linear-regression',
        'linear-regression-from-max',
        'linear-regression-from-min',
        'mann-kendall',
        'mann-kendall-from-max',
        'mann-kendall-from-min',
        'n-section-mann-kendall',
        'pettitt-test',
    )

    def __init__(self, significance_mode='linear-regression', nsims=1000, min_p_value=0.05, min_samples=10,
                 expect_slope='auto', efficent_mode=True, nparts=None, min_part_size=10, no_trend_alpha=0.50,
                 mpmk_check_step=1, mpmk_efficent_min=10, mpmk_window=0.05,
                 nsims_pettit=2000,
                 ncores=None, log_level=logging.INFO, return_true_conc=False, return_noisy_conc_itters=0,
                 only_significant_noisy=False, print_freq=None):
        """

        :param significance_mode: significance mode to use, options:
                 * linear-regression: linear regression of the concentration data from time 0 to the end
                                      change detected if p < min_p_value
                 * linear-regression-from-[max|min]: linear regression of the concentration data from the
                                               maximum concentration of the noise free concentration data to the end
                                               change detected if p < min_p_value
                 * mann-kendall: mann-kendall test of the concentration data from time 0 to the end,
                                 change detected if p < min_p_value
                 * mann-kendall-from-[max|min]: mann-kendall test of the concentration data from the maximum/minimum
                                              of the noisefree concentration data to the end,
                                              change detected if p < min_p_value
                 * n-section-mann-kendall: 2+ part mann-kendall test to identify change points. if change points are
                                           detected then a change is detected
                 * pettitt-test: pettitt test to identify change points. if change points are detected then a change is
                                detected
        :param nsims: number of noise simulations to run for each change detection (e.g. nsims=1000,
                      power= number of detected changes/1000 noise simulations)
        :param min_p_value: minimum p value to consider a change detected
        :param min_samples: minimum number of samples required, less than this number of samples will raise an exception
        :param expect_slope: expected slope of the concentration data, use depends on significance mode:
                              * linear-regression, linear-regression-from-max, mann-kendall, mann-kendall-from-max:
                                 one of 1 (increasing), -1 (decreasing), or 'auto' will match the slope of the
                                 concentration data before noise is added
                              * n-section-mann-kendall: expected trend in each part of the time series
                                 (1 increasing, -1 decreasing, 0 no trend)
                              * pettitt-test: not used.
        :param efficent_mode: bool, default = True, if True then
                             For linear regression and MannKendall based tests:  run the test on the noise free data
                               to see if any change can be detected, if no change is detected then the test will not be
                               on the noisy data

                             For MultiPartMannKendall test: the test will be run on the noise free data
                               to detect best change points and then the test will be run on the
                               noisy data for a smaller window centered on the True change point
                               see: * mpmk_efficent_min, * mpmk_window

                            For Pettitt Test:  Not implemented, will be ignored and a waring passed
        :param nparts: number of parts to use for the n-section-mann-kendall test (not used for other tests)
        :param min_part_size: minimum number of samples in each part for the n-section-mann-kendall test (not used for
                                other tests)
        :param no_trend_alpha: alpha value to use for the no trend sections in the n-section-mann-kendall test
                                trendless sections are only accepted if p > no_trend_alpha (not used for other tests)
        :param mpmk_check_step: int or function, default = 1, number of samples to check for a change point in the
                                MultiPartMannKendall test, used in both efficent_mode=True and efficent_mode=False
                                if mpmk is a function it must take a single argument (n, number of samples) and return
                                an integer check step
        :param mpmk_efficent_min: int, default = 10, minimum number of possible change points to assess
                                  only used if efficent_mode = True  The minimum number of breakpoints to test
                                  (mpmk_efficent_min) is always respected (i.e. if the window size is less than the
                                  minimum number of breakpoints to test, then the window size will be increased
                                  to the minimum number of breakpoints to test, but the space between breakpoints
                                  will still be defined by check_step). You can specify the exact number of breakpoints
                                  to check by setting mpmk_efficent_min=n breakpoints and setting mpmk_window=0
        :param mpmk_window: float, default = 0.05, define the window around the true detected change point to run the
                                   MultiPartMannKendall.  The detction window is defined as:
                                   (cp - mpmk_window*n, cp + mpmk_window*n) where cp is the detected change point and n
                                   is the number of samples in the time series
                                   Where both a mpmk_window and a check_step>1 is passed the mpmk_window will be
                                   used to define the window size and the check_step will be used to define the
                                   step size within the window.
        :param nsims_pettit: number of simulations to run for calculating the pvalue of the pettitt test
                             (not used for other tests)
        :param ncores: number of cores to use for multiprocessing, None will use all available cores
        :param log_level: logging level for multiprocessing subprocesses
        :param return_true_conc: return the true concentration time series for each simulation with power calcs
                                 (not supported with multiprocessing power calcs)
        :param return_noisy_conc_itters: int <= nsims, default = 0 Number of noisy simulations to return
                                         if 0 then no noisy simulations are returned, not supported with multiprocessing
                                            power calcs
        :param only_significant_noisy: bool if True then only return noisy simulations where a change was detected if
                                       there are fewer noisy simulations with changes detected than return_noisy_conc_itters
                                       all significant simulations will be returned. if there are no noisy simulations
                                       with changes detected then and empty dataframe is returned
        :param print_freq: None or int:  if None then no progress will be printed, if int then progress will be printed
                            every print_freq simulations (n%print_freq==0)
        """

        assert print_freq is None or isinstance(print_freq, int), 'print_freq must be None or an integer'
        self.print_freq = print_freq
        assert significance_mode in self.implemented_significance_modes, (f'significance_mode {significance_mode} not '
                                                                          f'implemented, must be one of '
                                                                          f'{self.implemented_significance_modes}')
        self.nsims_pettitt = nsims_pettit
        if significance_mode in ['linear-regression', 'linear-regression-from-max', 'linear-regression-from-min',
                                 'mann-kendall', 'mann-kendall-from-max', 'mann-kendall-from-min']:
            assert expect_slope in ['auto', 1, -1], 'expect_slope must be "auto", 1, or -1'
            assert isinstance(efficent_mode, bool), 'efficent_mode must be a boolean'
            self.efficent_mode = efficent_mode

        assert isinstance(only_significant_noisy, bool), 'only_significant_noisy must be a boolean'
        self.only_significant_noisy = only_significant_noisy
        assert isinstance(return_true_conc, bool), 'return_true_conc must be a boolean'
        self.return_true_conc = return_true_conc
        assert isinstance(return_noisy_conc_itters, int), 'return_noisy_conc_itters must be an integer'
        assert return_noisy_conc_itters <= nsims, 'return_noisy_conc_itters must be <= nsims'
        assert return_noisy_conc_itters >= 0, 'return_noisy_conc_itters must be >= 0'
        self.return_noisy_conc_itters = return_noisy_conc_itters

        self._power_from_max = False
        self._power_from_min = False
        if significance_mode == 'linear-regression':
            self.power_test = self._power_test_lr
        elif significance_mode == 'linear-regression-from-max':
            self._power_from_max = True
            self.power_test = self._power_test_lr
        elif significance_mode == 'linear-regression-from-min':
            self._power_from_min = True
            self.power_test = self._power_test_lr
        elif significance_mode == 'mann-kendall':
            assert kendal_imported, (
                'cannot run mann-kendall test, kendall_stats not installed'
                'to install run:\n'
                'pip install git+https://github.com/Komanawa-Solutions-Ltd/kendall_multipart_kendall.git')
            self.power_test = self._power_test_mann_kendall
        elif significance_mode == 'mann-kendall-from-max':
            self._power_from_max = True
            assert kendal_imported, (
                'cannot run mann-kendall test, kendall_stats not installed'
                'to install run:\n'
                'pip install git+https://github.com/Komanawa-Solutions-Ltd/kendall_multipart_kendall.git')
            self.power_test = self._power_test_mann_kendall
        elif significance_mode == 'mann-kendall-from-min':
            self._power_from_min = True
            assert kendal_imported, (
                'cannot run mann-kendall test, kendall_stats not installed'
                'to install run:\n'
                'pip install git+https://github.com/Komanawa-Solutions-Ltd/kendall_multipart_kendall.git')
            self.power_test = self._power_test_mann_kendall
        elif significance_mode == 'n-section-mann-kendall':
            assert kendal_imported, (
                'cannot run mann-kendall test, kendall_stats not installed'
                'to install run:\n'
                'pip install git+https://github.com/Komanawa-Solutions-Ltd/kendall_multipart_kendall.git')
            assert isinstance(nparts, int), 'nparts must be an integer'
            assert nparts > 1, 'nparts must be greater than 1'
            self.kendall_mp_nparts = nparts
            assert isinstance(min_part_size, int), 'min_part_size must be an integer'
            assert min_part_size > 1, 'min_part_size must be greater than 1'
            self.kendall_mp_min_part_size = min_part_size
            assert isinstance(no_trend_alpha, float), 'no_trend_alpha must be a float'
            assert no_trend_alpha > 0 and no_trend_alpha < 1, 'no_trend_alpha must be between 0 and 1'
            self.kendall_mp_no_trend_alpha = no_trend_alpha
            assert len(np.atleast_1d(expect_slope)) == self.kendall_mp_nparts, 'expect_slope must be of length nparts'
            assert set(np.atleast_1d(expect_slope)).issubset([1, 0, -1]), (
                f'expect_slope must be 1 -1, or 0, got:{set(np.atleast_1d(expect_slope))}')

            # mpmk_check_step
            if callable(mpmk_check_step):
                n = mpmk_check_step(100)
                assert isinstance(n, int), 'if mpmk_check_step is a function must return an integer'
                assert n > 0, 'if mpmk_check_step is a function must return an integer greater than 0'
            elif isinstance(mpmk_check_step, int):
                assert mpmk_check_step > 0, 'mpmk_check_step must be greater than 0'
            else:
                raise ValueError('mpmk_check_step must be an integer or function')
            self.mpmk_check_step = mpmk_check_step

            assert isinstance(efficent_mode, bool), 'efficent_mode must be a boolean'
            self.efficent_mode = efficent_mode
            assert isinstance(mpmk_efficent_min, int), 'mpmk_efficent_min must be an integer'
            assert mpmk_efficent_min > 0, 'mpmk_efficent_min must be greater than 0'
            self.mpmpk_efficent_min = mpmk_efficent_min
            assert isinstance(mpmk_window, float), 'mpmk_window must be a float'
            assert mpmk_window > 0 and mpmk_window < 1 / nparts, f'mpmk_window must be between 0 and {1 / nparts}'
            self.mpmk_window = mpmk_window
            self.power_test = self._power_test_mp_kendall
        elif significance_mode == 'pettitt-test':
            assert pyhomogeneity_imported, (
                'cannot run pettitt test, pyhomogeneity not installed'
                'to install run:\n'
                'pip install pyhomogeneity')
            assert isinstance(nsims_pettit, int), 'nsims_pettit must be an integer'
            assert isinstance(efficent_mode, bool), 'efficent_mode must be a boolean'
            if efficent_mode:
                warnings.warn('efficent_mode not implemented for pettitt test, setting efficent_mode=False')
            self.efficent_mode = False  # TODO long term fix this if needed
            self.power_test = self._power_test_pettitt
        else:
            raise NotImplementedError(f'significance_mode {significance_mode} not implemented, shouldnt get here')

        self.expect_slope = expect_slope

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
        self.significance_mode = significance_mode

    def plot_iteration(self, y0, true_conc):
        """
        plot the concentration data itteration and the true concentration data
        if provided as well as the power test results
        and any predictions from the power test (e.g. the slope of the line used)
        :param y0: noisy concentration data
        :param true_conc: true concentration data
        :return:
        """
        istart = 0
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(np.arange(len(y0)), y0, c='k', label=f'noisy data')
        ax.plot(np.arange(len(true_conc)), true_conc, c='r', label=f'True data', marker='.')
        if self.significance_mode in ['linear-regression', 'linear-regression-from-max', 'linear-regression-from-min']:
            if self._power_from_max:
                istart = np.argmax(true_conc)
            elif self._power_from_min:
                istart = np.argmin(true_conc)
            o2 = stats.linregress(np.arange(len(y0))[istart:], y0[istart:])
            ax.plot(np.arange(len(y0))[istart:], o2.intercept + o2.slope * np.arange(len(y0))[istart:], c='b',
                    ls='--', label='regression, p={:.3f}'.format(o2.pvalue))

        elif self.significance_mode in ['mann-kendall', 'mann-kendall-from-max', 'mann-kendall-from-min']:
            if self._power_from_max:
                istart = np.argmax(true_conc)
            elif self._power_from_min:
                istart = np.argmin(true_conc)
            mk = MannKendall(data=y0[istart:], alpha=self.min_p_value)
            mk.plot_data(ax=ax)

        elif self.significance_mode == 'n-section-mann-kendall':
            mpmk = MultiPartKendall(data=y0, nparts=self.kendall_mp_nparts,
                                    expect_part=self.expect_slope, min_size=self.kendall_mp_min_part_size,
                                    alpha=self.min_p_value, no_trend_alpha=self.kendall_mp_no_trend_alpha)
            bp = mpmk.get_maxz_breakpoints(raise_on_none=False)
            if bp is None:
                warnings.warn('no breakpoints found, not plotting multipart mann kendall')
            else:
                mpmk.plot_data_from_breakpoints(bp, ax=ax)
        elif self.significance_mode == 'pettitt-test':
            h, cp, p, U, mu = pettitt_test(y0, alpha=self.min_p_value,
                                           sim=self.nsims_pettitt)
            p = round(p, 3)
            ax.axvline(x=cp, color='r', label=f'change_point, {p=}')
        else:
            raise ValueError(f'unknown significance_mode {self.significance_mode}, should not get here')

        if istart > 0:
            ax.axvline(istart, c='k', ls=':', label='start of power test')
        ax.legend()
        return fig, ax

    def _run_power_calc(self, testnitter, seed, true_conc_ts, idv, error, expect_slope, max_conc_val,
                        max_conc_time, **kwargs):
        if testnitter is not None:
            warnings.warn('testnitter is expected to be None unless you are testing run times')

        if seed is None:
            seed = np.random.randint(21, 54762438)

        nsamples = len(true_conc_ts)
        if nsamples < self.min_samples:
            raise ValueError(f'nsamples must be greater than {self.min_samples}, you can change the '
                             f'minimum number of samples in the DetectionPowerCalculator class init')
        # tile to nsims
        if testnitter is not None:
            rand_shape = (testnitter, nsamples)
            conc_with_noise = np.tile(true_conc_ts, testnitter).reshape(rand_shape)
        else:
            rand_shape = (self.nsims, nsamples)
            conc_with_noise = np.tile(true_conc_ts, self.nsims).reshape(rand_shape)

        # generate noise
        np.random.seed(seed)
        all_seeds = list(np.random.randint(21, 54762438, 2))
        np.random.seed(all_seeds.pop(0))
        noise = np.random.normal(0, error, rand_shape)
        conc_with_noise += noise

        # run slope test
        power, significant = self.power_test(idv, conc_with_noise,
                                             expected_slope=expect_slope,  # just used for sign
                                             imax=np.argmax(true_conc_ts), imin=np.argmin(true_conc_ts),
                                             true_data=true_conc_ts,
                                             return_slope=False)

        out = pd.Series({'idv': idv,
                         'power': power,
                         'max_conc': max_conc_val,
                         'max_conc_time': max_conc_time,
                         'error': error,
                         'seed': seed,
                         'python_error': None
                         })
        for k, v in kwargs.items():
            out[k] = v

        out_data = {}
        out_data['power'] = out
        if self.return_true_conc:
            out_data['true_conc'] = pd.DataFrame(data=true_conc_ts, columns=['true_conc'])

        if self.return_noisy_conc_itters > 0:
            if self.only_significant_noisy:
                conc_with_noise = conc_with_noise[significant]
            outn = min(self.return_noisy_conc_itters, conc_with_noise.shape[0])
            out_data['noisy_conc'] = pd.DataFrame(data=conc_with_noise[:outn].T,
                                                  columns=np.arange(outn))
        if len(out_data) == 1:
            out_data = out_data['power']
        return out_data

    def power_calc(self, idv, error: float, true_conc_ts: np.ndarray,
                   seed: {int, None} = None, testnitter=None, **kwargs):
        """

        :param idv: identifiers for the power calc sites, passed straight through to the output
        :param error: standard deviation of the noise
        :param true_conc_ts: the true concentration timeseries (only used for
                        pass_true_conc model)
        :param seed: int or None for random seed
        :param testnitter: None (usually) or a different nitter then self.niter for testing run times
        :return: pd.DataFrame with the power calc results (len=1) note power is percent 0-100
                Possible other dataframes if self.return_true_conc is True or self.return_noisy_conc_itters > 0
                in which case a dictionary will be returned:
                    [
                    'power': power_df, # always
                    'true_conc': true_conc_ts, if self.return_true_conc is True
                    'noisy_conc' : noisy_conc_ts, if self.return_noisy_conc_itters > 0
                    ]
        """

        assert true_conc_ts is not None
        max_conc_val = np.max(true_conc_ts)
        max_conc_time = np.argmax(true_conc_ts)
        if self.expect_slope == 'auto':
            power, significant, expect_slope = self.power_test('auto_slope_finder',
                                                               np.atleast_1d(true_conc_ts)[np.newaxis, :],
                                                               expected_slope=None, imax=np.argmax(true_conc_ts),
                                                               imin=np.argmin(true_conc_ts), true_data=true_conc_ts,
                                                               return_slope=True)
        else:
            expect_slope = self.expect_slope

        outdata = self._run_power_calc(testnitter=testnitter,
                                       seed=seed,
                                       true_conc_ts=true_conc_ts,
                                       idv=idv,
                                       error=error,
                                       expect_slope=expect_slope,
                                       max_conc_val=max_conc_val,
                                       max_conc_time=max_conc_time,
                                       **kwargs)
        return outdata

    def mulitprocess_power_calcs(self,
                                 outpath: {Path, None, str},
                                 id_vals: np.ndarray,
                                 error_vals: {np.ndarray, float},
                                 true_conc_ts_vals: {np.ndarray, list, None} = None,
                                 seed: {np.ndarray, int, None} = 5585,
                                 run=True, debug_mode=False,
                                 **kwargs
                                 ):
        """
        multiprocessing wrapper for power_calc, see power_calc for details
        note that if a given run raises and exception the traceback for the exception will be included in the
        returned dataset under the column 'python_error' if 'python_error' is None then the run was successful
        to change the number of cores used pass n_cores to the constructor init

        :param outpath: path to save results to
        :param id_vals: id values for each simulation

        All values from here on out should be either a single value or an array of values with the same shape as
            id_vals

        :param error_vals: standard deviation of noise to add for each simulation
        :param true_conc_ts_vals: the true concentration time series for each simulation only used for the
                                    'pass_true_conc' mrt_model, note that this can be a list of arrays of different
                                    lengths for each simulation, Numpy does not support jagged arrays
        :param seed: ndarray (integer seeds), None (no seeds), or int (1 seed for all simulations)
        :param run: if True run the simulations, if False just build  the run_dict and print the number of simulations
        :param debug_mode: if True run as single process to allow for easier debugging
        :return: dataframe with input data and the results of all of the power calcs. note power is percent 0-100
        """

        use_kwargs = dict(error_vals=error_vals,
                          true_conc_ts_vals=true_conc_ts_vals,
                          seed=seed,
                          **kwargs
                          )

        outpath, id_vals, use_kwargs = self._multiprocess_checks(outpath, id_vals, **use_kwargs)

        runs = []
        for i in range(len(id_vals)):
            kwargs = {k.replace('_vals', ''): v[i] for k, v in use_kwargs.items()}
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


class AutoDetectionPowerSlope(DetectionPowerSlope):
    implemented_mrt_models = (
        'piston_flow',
        'binary_exponential_piston_flow',
    )
    _auto_mode = True
    condensed_mode = False

    # precision to round values to (2 = 0.01), set stupidly high to ensure no rounding on default
    previous_slope_per = 9
    max_conc_lim_per = 9
    min_conc_lim_per = 9
    mrt_per = 9
    mrt_p1_per = 9
    frac_p1_per = 9
    f_p1_per = 9
    f_p2_per = 9

    def set_condensed_mode(self, previous_slope_per=2,
                           max_conc_lim_per=1,
                           min_conc_lim_per=1,
                           mrt_per=0,
                           mrt_p1_per=2,
                           frac_p1_per=2,
                           f_p1_per=2,
                           f_p2_per=2, ):
        """
        set calculator to condense the number of runs based by rounding the inputs to a specified precision
        :param previous_slope_per: precision to round previous_slope to (2 = 0.01)
        :param max_conc_lim_per: precision to round max_conc_lim to (2 = 0.01)
        :param min_conc_lim_per: precision to round min_conc_lim to (2 = 0.01)
        :param mrt_per: precision to round mrt to
        :param mrt_p1_per: precision to round mrt_p1 to
        :param frac_p1_per: precision to round frac_p1 to
        :param f_p1_per: precision to round f_p1 to
        :param f_p2_per: precision to round f_p2 to
        :return:
        """

        self.condensed_mode = True
        self.previous_slope_per = previous_slope_per
        self.max_conc_lim_per = max_conc_lim_per
        self.min_conc_lim_per = min_conc_lim_per
        self.mrt_per = mrt_per
        self.mrt_p1_per = mrt_p1_per
        self.frac_p1_per = frac_p1_per
        self.f_p1_per = f_p1_per
        self.f_p2_per = f_p2_per

    def power_calc(self, idv, error: float, mrt_model: str, samp_years: int, samp_per_year: int,
                   implementation_time: int, initial_conc: float, target_conc: float, prev_slope: float,
                   max_conc_lim: float, min_conc_lim: float, mrt: float = 0,
                   # options for binary_exponential_piston_flow model
                   mrt_p1: {float, None} = None, frac_p1: {float, None} = None, f_p1: {float, None} = None,
                   f_p2: {float, None} = None,
                   # options for the model run
                   seed: {int, None} = 5585,
                   testnitter=None,
                   **kwargs):
        """

        calculate the detection power for a given set of parameters

        :param idv: identifiers for the power calc sites, passed straight through to the output
        :param error: standard deviation of the noise
        :param mrt_model: the model to use for the mean residence time options:
                          * 'piston_flow': use the piston flow model (no mixing, default)
                          * 'binary_exponential_piston_flow': use the binary exponential piston flow model
                          for unitary exponential_piston_flow model set frac_1 = 1 and mrt_p1 = mrt
                          for no lag, set mrt=0, mrt_model='piston_flow'
        :param samp_years: number of years to sample
        :param samp_per_year: number of samples to collect each year
        :param implementation_time: number of years over which reductions are implemented
        :param initial_conc: inital median value of the concentration
        :param target_conc: target concentration to reduce to
        :param prev_slope: slope of the previous data (e.g. prior to the initial concentration)
        :param max_conc_lim: maximum concentration limit user specified or None (default)
        :param min_conc_lim: minimum concentration limit for the source, only used for the
                         binary_exponential_piston_flow model)
        :param mrt: the mean residence time of the site

        Options for binary_exponential_piston_flow model:
        :param mrt_p1: the mean residence time of the first piston flow model (only used for
                        binary_exponential_piston_flow model)
        :param frac_p1: the fraction of the first piston flow model (only used for
                        binary_exponential_piston_flow model)
        :param f_p1: the fraction of the first piston flow model (only used for
                        binary_exponential_piston_flow model)
        :param f_p2: the fraction of the first piston flow model (only used for
                        binary_exponential_piston_flow model)

        Model run options:
        :param seed: int or None for random seed
        :param testnitter: None (usually) or a different nitter then self.niter for testing run times
        :param kwargs: kwargs passed to the output series
                      (e.g. region='temp' will yield a 'region' index with a value of 'temp')
        :return: pd.Seris with the power calc results note power is percent 0-100
                Possible other dataframes if self.return_true_conc is True or self.return_noisy_conc_itters > 0
                in which case a dictionary will be returned:
                    [
                    'power': power_df, # always
                    'true_conc': true_conc_ts, if self.return_true_conc is True
                    'noisy_conc' : noisy_conc_ts, if self.return_noisy_conc_itters > 0
                    ]
        """
        if testnitter is not None:
            warnings.warn('testnitter is expected to be None unless you are testing run times')
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
            assert pd.api.types.is_number(max_conc_lim), 'max_conc must be a number unless mrt_model="pass_true_conc"'
            assert max_conc_lim >= initial_conc, 'max_conc must be greater than or equal to initial_conc'
            assert max_conc_lim >= target_conc, 'max_conc must be greater than or equal to target_conc'
            assert pd.api.types.is_integer(implementation_time)

        # mange lag
        if mrt_model == 'piston_flow':
            if self.significance_mode == 'pettitt-test' and mrt != 0:
                warnings.warn('using the Pettitt test with lagged data can cause some weird results, we do'
                              'not recommend using this combination')

            true_conc_ts, max_conc_val, max_conc_time, mrt_p2 = self.truets_from_piston_flow(mrt,
                                                                                             initial_conc, target_conc,
                                                                                             prev_slope, max_conc_lim,
                                                                                             samp_per_year, samp_years,
                                                                                             implementation_time)
            if self.expect_slope == 'auto':
                expect_slope = (target_conc - initial_conc) / implementation_time
            else:
                expect_slope = self.expect_slope

        elif mrt_model == 'binary_exponential_piston_flow':
            if self.significance_mode == 'pettitt-test' and mrt != 0:
                warnings.warn('using the Pettitt test with lagged data can cause some weird results, we do'
                              'not recommend using this combination')
            assert age_tools_imported, (
                'cannot run binary_exponential_piston_flow model, age_tools not installed'
                'to install run:\n'
                'pip install git+https://github.com/Komanawa-Solutions-Ltd/gw_age_tools')
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
                initial_conc, target_conc, prev_slope, max_conc_lim, min_conc_lim,
                samp_per_year, samp_years, implementation_time,
                return_extras=False)
            if self.expect_slope == 'auto':
                expect_slope = (target_conc - initial_conc) / implementation_time
            else:
                expect_slope = self.expect_slope

        else:
            raise NotImplementedError(f'mrt_model {mrt_model} not currently implemented')

        out_data = self._run_power_calc(testnitter=testnitter,
                                        seed=seed,
                                        true_conc_ts=true_conc_ts,
                                        idv=idv,
                                        error=error,
                                        expect_slope=expect_slope,
                                        max_conc_val=max_conc_val,
                                        max_conc_time=max_conc_time,
                                        mrt_model=mrt_model,
                                        samp_years=samp_years,
                                        samp_per_year=samp_per_year,
                                        implementation_time=implementation_time,
                                        initial_conc=initial_conc,
                                        target_conc=target_conc,
                                        prev_slope=prev_slope,
                                        max_conc_lim=max_conc_lim,
                                        min_conc_lim=min_conc_lim,
                                        mrt=mrt,
                                        mrt_p1=mrt_p1,
                                        frac_p1=frac_p1,
                                        f_p1=f_p1,
                                        f_p2=f_p2,
                                        **kwargs)

        return out_data

    def mulitprocess_power_calcs(
            self,
            outpath: {Path, None, str},
            id_vals: np.ndarray,
            error_vals: {np.ndarray, float},
            samp_years_vals: {np.ndarray, int},
            samp_per_year_vals: {np.ndarray, int},
            implementation_time_vals: {np.ndarray, int},
            initial_conc_vals: {np.ndarray, float},
            target_conc_vals: {np.ndarray, float},
            previous_slope_vals: {np.ndarray, float},
            max_conc_lim_vals: {np.ndarray, float},
            min_conc_lim_vals: {np.ndarray, float},
            mrt_model_vals: {np.ndarray, str},
            mrt_vals: {np.ndarray, float},
            mrt_p1_vals: {np.ndarray, float, None} = None,
            frac_p1_vals: {np.ndarray, float, None} = None,
            f_p1_vals: {np.ndarray, float, None} = None,
            f_p2_vals: {np.ndarray, float, None} = None,
            seed: {np.ndarray, int, None} = None,
            run=True, debug_mode=False, **kwargs
    ):

        use_kwargs = dict(error_vals=error_vals,
                          seed=seed,
                          samp_years_vals=samp_years_vals,
                          samp_per_year_vals=samp_per_year_vals,
                          implementation_time_vals=implementation_time_vals,
                          initial_conc_vals=initial_conc_vals,
                          target_conc_vals=target_conc_vals,
                          previous_slope_vals=previous_slope_vals,
                          max_conc_lim_vals=max_conc_lim_vals,
                          min_conc_lim_vals=min_conc_lim_vals,
                          mrt_model_vals=mrt_model_vals,
                          mrt_vals=mrt_vals,
                          mrt_p1_vals=mrt_p1_vals,
                          frac_p1_vals=frac_p1_vals,
                          f_p1_vals=f_p1_vals,
                          f_p2_vals=f_p2_vals,
                          **kwargs)

        outpath, id_vals, use_kwargs = self._multiprocess_checks(outpath, id_vals, **use_kwargs)

        # make runs
        runs = []

        if self.condensed_mode:
            all_use_idv = []
            run_list = []
            print('creating and condensing runs')
            use_kwargs = {k: self._round_kwarg_value(v, k) for k, v in use_kwargs.items()}

            for i in range(len(id_vals)):
                if i % 1000 == 0:
                    print(f'forming/condesing run {i} of {len(id_vals)}')

                use_idv = '_'.join([self._get_id_str(use_kwargs[e][i], e) for e in
                                    ['error', 'samp_years', 'samp_per_year', 'implementation_time', 'initial_conc',
                                     'target_conc', 'previous_slope', 'max_conc_lim', 'min_conc_lim', 'mrt_model',
                                     'mrt',
                                     'mrt_p1', 'frac_p1', 'f_p1', 'f_p2']
                                    ])
                all_use_idv.append(use_idv)
                if use_idv in run_list:
                    continue
                run_list.append(use_idv)
                kwargs = {k.replace('_vals', ''): v[i] for k, v in use_kwargs.items()}
                runs.append(kwargs)

        else:
            all_use_idv = id_vals

        if self.condensed_mode:
            print(f'running {len(runs)} runs, simplified from {len(id_vals)} runs')
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
        outdata.loc[:, 'idv'] = id_vals
        outdata.set_index('idv', inplace=True, drop=True)

        if outpath is not None:
            outpath.parent.mkdir(parents=True, exist_ok=True)
            outdata.to_hdf(outpath, 'data')
        return outdata
