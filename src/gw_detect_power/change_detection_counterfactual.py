"""
created matt_dumont 
on: 25/01/24
"""

import warnings
import logging
from gw_detect_power.base_detection_calculator import BaseDetectionCalculator, _run_multiprocess

# handle import of optional dependencies
age_tools_imported = True
pyhomogeneity_imported = True
kendal_imported = True

# todo 'true_conc_base_vals' # 'true_conc_alt_vals'

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



class DetectionPowerCounterFactual(BaseDetectionCalculator):  # todo pass own concentration
    implemented_mrt_models = ()
    implemented_significance_modes = (
        'counter-factual',  # todo multiple tests???
    )
    _counterfactual = True

    def __init__(self, p_value=0.05,
                 min_samples=10,
                 efficent_mode=True,
                 ncores=None,
                 log_level=logging.INFO,
                 return_true_conc=False,
                 return_noisy_conc_itters=0,
                 only_significant_noisy=False,
                 print_freq=None
                 ):
        raise NotImplementedError

    def plot_iteration(self, y01, y02, true_conc1, true_conc2):
        raise NotImplementedError


class AutoDetectionPowerCounterFactual(DetectionPowerCounterFactual):  # todo don't pass own concentration
    implemented_mrt_models = (
        'piston_flow',
        'binary_exponential_piston_flow',
    )
    _auto_mode = True
    _counterfactual = True

    def __init__(self):
        raise NotImplementedError

