"""
created matt_dumont 
on: 3/10/23
"""
import pandas as pd

from komanawa.gw_detect_power.change_detection_slope import DetectionPowerSlope
from komanawa.kendall_stats import make_example_data
import numpy as np
import itertools
from pathlib import Path
import timeit
import sys
import os

# constants
nsims = 10
mpmk_check_step = 1
mpmk_efficent_min = 10
mpmk_window = 0.05
nsims_pettit = 2000

# iterables
methods = DetectionPowerSlope.implemented_significance_modes
ndata = [50, 100, 500, 1000, 5000]
efficency_modes = [True, False]

def timeit_test(methods=methods, ndata=ndata, efficency_modes=efficency_modes, n=1):
    """
    run an automated timeit test, must be outside of the function definition, prints results in scientific notation
    units are seconds

    :param py_file_path: path to the python file that holds the functions,
                        if the functions are in the same script as call then  __file__ is sufficient.
                        in this case the function call should be protected by: if __name__ == '__main__':
    :param function_names: the names of the functions to test (iterable), functions must not have arguments
    :param n: number of times to test
    :return:
    """
    py_file_path = __file__
    print(py_file_path)
    d = os.path.dirname(py_file_path)
    fname = os.path.basename(py_file_path).replace('.py', '')
    sys.path.append(d)

    out = pd.DataFrame(index=pd.MultiIndex.from_product([ndata, efficency_modes],
                                                        names=['n data', 'efficency_mode']),
                       columns=methods)

    for nd, emode, method in itertools.product(ndata, efficency_modes, methods):
        print(f'testing: {method=}, {nd=}, {emode=}')
        fn = f'run_model'
        t = timeit.timeit(f'{fn}("{method}", {nd}, {emode})',
                          setup='from {} import {}'.format(fname, fn),
                          number=n) / n
        out.loc[(nd, emode), method] = t
        print('{0:e} seconds'.format(t))
    return out


def run_model(method, ndata, emode):
    if method in ['linear-regression', 'mann-kendall', ]:
        x, data = make_example_data.make_increasing_decreasing_data(slope=0.1, noise=0, step=100 / ndata)
        use_noise = 5
    elif method in ['linear-regression-from-max', 'mann-kendall-from-max', 'n-section-mann-kendall']:
        x, data = make_example_data.make_multipart_sharp_change_data(slope=make_example_data.multipart_sharp_slopes[0],
                                                                     noise=0,
                                                                     unsort=False, na_data=False, step=100 / ndata)
        use_noise = make_example_data.multipart_sharp_noises[1]

    elif method in ['linear-regression-from-min', 'mann-kendall-from-min', ]:
        x, data = make_example_data.make_multipart_sharp_change_data(slope=make_example_data.multipart_sharp_slopes[1],
                                                                     noise=0,
                                                                     unsort=False, na_data=False, step=100 / ndata)
        use_noise = make_example_data.multipart_sharp_noises[1]

    elif method == 'pettitt-test':
        data = np.zeros((ndata)) + 10
        data[len(data) * 3 // 4:] = 8
        use_noise = 5
    else:
        raise ValueError(f'unknown method {method}')

    expect_slope = 'auto'
    if method == 'n-section-mann-kendall':
        expect_slope = (1, -1)

    dpc = DetectionPowerSlope(significance_mode=method,
                                   nsims=nsims, min_p_value=0.05, min_samples=10,
                                   expect_slope=expect_slope, efficent_mode=emode, nparts=2, min_part_size=10,
                                   no_trend_alpha=0.50,
                                   mpmk_check_step=mpmk_check_step, mpmk_efficent_min=mpmk_efficent_min,
                                   mpmk_window=mpmk_window,
                                   nsims_pettit=nsims_pettit, )

    out = dpc.power_calc(idv='test',
                         error=use_noise,
                         mrt_model='pass_true_conc',
                         true_conc_ts=data)


if __name__ == '__main__':
    data = timeit_test()
    data.to_csv(Path(__file__).parent.joinpath('timeit_test_results.txt'))
