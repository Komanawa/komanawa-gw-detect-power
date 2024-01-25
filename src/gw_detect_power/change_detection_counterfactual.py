"""
created matt_dumont 
on: 25/01/24
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import warnings
import logging
from gw_detect_power.base_detection_calculator import BaseDetectionCalculator, _run_multiprocess

# handle import of optional dependencies
age_tools_imported = True
pyhomogeneity_imported = True
kendal_imported = True

# todo names 'true_conc_base_vals' # 'true_conc_alt_vals'

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


# todo discuss comparing modelled vs measured results, fail to reject null hypothesis, if you run power calcs against a
#  status quo then you can say whether your measured results are correct or simply you dont have enough power.

class DetectionPowerCounterFactual(BaseDetectionCalculator):  # todo pass own concentration
    implemented_mrt_models = ()
    implemented_significance_modes = (  # todo non-paired???
        'paired-t-test',
        'wilcoxon-signed-rank-test',
    )
    _counterfactual = True
    _poss_alternatives = ('alt!=base', 'alt<base', 'alt>base')
    _scipy_alternatives = ('two-sided', 'less', 'greater')

    def __init__(self,
                 significance_mode,
                 nsims=1000,
                 p_value=0.05,
                 min_samples=10,
                 alternative='alt!=base',
                 wx_zero_method='wilcox', wx_correction=False, wx_method='auto',
                 ncores=None,
                 log_level=logging.INFO,
                 return_true_conc=False,
                 return_noisy_conc_itters=0,
                 only_significant_noisy=False,
                 print_freq=None,
                 ):
        """

        :param significance_mode: str, one of:
                                    'paired-t-test': paired t test (parametric), scipy.stats.ttest_rel
                                    'wilcoxon-signed-rank-test': wilcoxon signed rank test (non-parametric),
                                                                 scipy.stats.wilcoxon
        :param nsims: number of noise simulations to run for each change detection (e.g. nsims=1000,
                      power= number of detected changes/1000 noise simulations)
        :param p_value: minimum p value (see also alternative), if
                           p >= p_value the null hypothesis will not be rejected (base and alt are the same)
                           p < p_value the null hypothesis will be rejected (base and alt are different)
        :param min_samples: minimum number of samples required, less than this number of samples will raise an exception
        :param alternative: str, one of:
                                'alt!=base': two sided test (default),
                                'alt<base': one sided test ~
                                'alt>base'
        :param wx_zero_method: str, one of:
                                    “wilcox”: Discards all zero-differences (default); see [4].
                                    “pratt”: Includes zero-differences in the ranking process, but drops the ranks
                                             of the zeros (more conservative); see [3]. In this case, the normal
                                             approximation is adjusted as in [5].
                                    “zsplit”: Includes zero-differences in the ranking process and splits the zero
                                              rank between positive and negative ones.
                                for more info see:
                                  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
        :param wx_correction: bool, If True, apply continuity correction by adjusting the Wilcoxon rank statistic by
                                0.5 towards the mean value when computing the z-statistic. Default is False.
        :param wx_method: str, see scipy.stats.wilcoxon for more info
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
        assert isinstance(only_significant_noisy, bool), 'only_significant_noisy must be a boolean'
        self.only_significant_noisy = only_significant_noisy
        assert isinstance(return_true_conc, bool), 'return_true_conc must be a boolean'
        self.return_true_conc = return_true_conc
        assert isinstance(return_noisy_conc_itters, int), 'return_noisy_conc_itters must be an integer'
        assert return_noisy_conc_itters <= nsims, 'return_noisy_conc_itters must be <= nsims'
        assert return_noisy_conc_itters >= 0, 'return_noisy_conc_itters must be >= 0'
        self.return_noisy_conc_itters = return_noisy_conc_itters

        assert alternative in self._poss_alternatives, (f'alternative {alternative} not implemented, must be one of '
                                                        f'{self._poss_alternatives}')
        alt_dict = dict(zip(self._poss_alternatives, self._scipy_alternatives))
        self.alternative = alt_dict[alternative]
        assert isinstance(wx_zero_method, str), 'wx_zero_method must be a string'
        assert wx_zero_method in ['wilcox', 'pratt', 'zsplit'], 'wx_zero_method must be one of "wilcox", "pratt", ' \
                                                                '"zsplit"'
        self.wx_zero_method = wx_zero_method
        assert isinstance(wx_correction, bool), 'wx_correction must be a boolean'
        self.wx_correction = wx_correction
        assert isinstance(wx_method, str), 'wx_method must be a string'
        assert wx_method in ['auto', 'asymptotic', 'exact'], 'wx_method must be one of "auto", "asymptotic", "exact"'
        self.wx_method = wx_method

        if significance_mode == 'paired-t-test':
            self._power_test = self._power_test_paired_t
        elif significance_mode == 'wilcoxon-signed-rank-test':
            self._power_test = self._power_test_wilcoxon
        else:
            raise NotImplementedError(f'significance_mode {significance_mode} not implemented, must be one of '
                                      f'{self.implemented_significance_modes}')

        assert isinstance(nsims, int), 'nsims must be an integer'
        self.nsims = nsims
        assert isinstance(min_samples, int), 'min_samples must be an integer'
        self.min_samples = min_samples
        self.min_p_value = p_value
        assert self.min_p_value > 0 and self.min_p_value < 1, 'min_p_value must be between 0 and 1'
        assert isinstance(ncores, int) or ncores is None, 'ncores must be an integer or None'
        self.ncores = ncores
        assert log_level in [logging.CRITICAL, logging.FATAL, logging.ERROR, logging.WARNING, logging.WARN,
                             logging.INFO, logging.DEBUG], f'unknown log_level {log_level}'
        self.log_level = log_level
        self.significance_mode = significance_mode

    def plot_iteration(self, y01, y02, true_conc1, true_conc2): # todo
        raise NotImplementedError

    def _power_test_paired_t(self, base_with_noise, alt_with_noise):
        assert base_with_noise.shape == alt_with_noise.shape, ('base_with_noise and alt_with_noise must have the same '
                                                               'shape')
        outdata = stats.ttest_rel(alt_with_noise, base_with_noise, axis=1, alternative=self.alternative)
        p_list = outdata.pvalue < self.min_p_value
        power = p_list.mean() * 100
        return power, p_list

    def _power_test_wilcoxon(self, base_with_noise, alt_with_noise):
        assert base_with_noise.shape == alt_with_noise.shape, ('base_with_noise and alt_with_noise must have the same '
                                                               'shape')
        outdata = stats.wilcoxon(alt_with_noise, base_with_noise, axis=1, alternative=self.alternative,
                                 zero_method=self.wx_zero_method, correction=self.wx_correction,
                                 method=self.wx_method)
        assert hasattr(outdata, 'pvalue'), 'scipy changed'
        p_list = outdata.pvalue < self.min_p_value
        power = p_list.mean() * 100
        return power, p_list

    def _run_power_calc(self, idv, testnitter, seed_base, seed_alt, true_conc_base, true_conc_alt, error_base,
                        error_alt, **kwargs):
        if testnitter is not None:
            warnings.warn('testnitter is expected to be None unless you are testing run times')

        if seed_alt is None:
            seed_alt = np.random.randint(21, 54762438)
        if seed_base is None:
            seed_base = np.random.randint(21, 54762438)

        if seed_base == seed_alt:
            raise ValueError('seed_base and seed_alt must be different otherwise the same noise will be added to both '
                             'concentration time series and the effective noise will be zero')

        nsamples = len(true_conc_base)
        assert nsamples == len(true_conc_alt), 'true_conc_base and true_conc_alt must be the same length'
        assert np.isfinite(true_conc_base).all(), 'true_conc_base must not contain any NaN or inf values'
        assert np.isfinite(true_conc_alt).all(), 'true_conc_alt must not contain any NaN or inf values'

        if nsamples < self.min_samples:
            raise ValueError(f'nsamples must be greater than {self.min_samples}, you can change the '
                             f'minimum number of samples in the DetectionPowerCalculator class init')

        # tile to nsims
        if testnitter is not None:
            rand_shape = (testnitter, nsamples)
        else:
            rand_shape = (self.nsims, nsamples)

        base_with_noise = np.tile(true_conc_base, testnitter).reshape(rand_shape)
        alt_with_noise = np.tile(true_conc_alt, testnitter).reshape(rand_shape)

        # generate noise
        np.random.seed(seed_base)
        base_noise = np.random.normal(0, error_base, rand_shape)
        np.random.seed(seed_alt)
        alt_noise = np.random.normal(0, error_alt, rand_shape)

        # add noise
        base_with_noise += base_noise
        alt_with_noise += alt_noise

        # run test
        power, significant = self._power_test(base_with_noise, alt_with_noise)

        out = pd.Series({'idv': idv,
                         'power': power,
                         'error_base': error_base,
                         'error_alt': error_alt,
                         'seed_base': seed_base,
                         'seed_alt': seed_alt,
                         'python_error': None})
        for k, v in kwargs.items():
            out[k] = v

        out_data = {}
        out_data['power'] = out
        if self.return_true_conc:
            out_data['true_conc'] = pd.DataFrame(data=[true_conc_base, true_conc_alt],
                                                 columns=['true_conc_base', 'true_conc_alt'])

        if self.return_noisy_conc_itters > 0:
            if self.only_significant_noisy:
                out_base_with_noise = base_with_noise[significant]
                out_alt_with_noise = alt_with_noise[significant]
            outn = min(self.return_noisy_conc_itters, out_base_with_noise.shape[0])
            out_data['base_noisy_conc'] = pd.DataFrame(data=out_base_with_noise[:outn].T,
                                                       columns=np.arange(outn))
            out_data['alt_noisy_conc'] = pd.DataFrame(data=out_alt_with_noise[:outn].T,
                                                      columns=np.arange(outn))

        if len(out_data) == 1:
            out_data = out_data['power']
        return out_data

    def power_calc(self, idv, error_base: float,
                   true_conc_base: np.ndarray,
                   true_conc_alt: np.ndarray,
                   error_alt: {float, None} = None,
                   seed: {int, None} = None, testnitter=None,
                   **kwargs
                   ):  # todo start here,
        raise NotImplementedError

    def mulitprocess_power_calcs(self):  # todo
        raise NotImplementedError


class AutoDetectionPowerCounterFactual(DetectionPowerCounterFactual):  # todo don't pass own concentration
    implemented_mrt_models = (
        'piston_flow',
        'binary_exponential_piston_flow',
    )
    _auto_mode = True
    _counterfactual = True

    def set_condensed_mode(self):  # todo
        raise NotImplementedError

    def power_calc(self):  # todo
        # todo a delay parameter (e.g. don't start monitring right away?
        # todo source target 1, source target 2
        raise NotImplementedError

    def mulitprocess_power_calcs(self):  # todo
        raise NotImplementedError
