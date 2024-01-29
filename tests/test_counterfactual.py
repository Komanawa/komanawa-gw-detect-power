"""
created matt_dumont 
on: 25/01/24
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import itertools
from gw_detect_power import DetectionPowerCounterFactual, AutoDetectionPowerCounterFactual


def make_step_test_data(delta, length):
    x1 = np.full(length, 12)
    x2 = x1 + delta
    return x1, x2


def make_linear_test_data(slope, length, delta=0):
    x1 = np.full(length, 12)
    x = np.arange(length)
    x2 = x1 + slope * x + delta
    return x1, x2


def make_bilinar_test_data(slope1, slope2, length, delta1=0, delta2=0):
    x = np.arange(length)
    x0 = np.full(length, 12)
    x1 = x0 + slope1 * x + delta1
    x2 = x0 + slope2 * x + delta2
    return x1, x2


def test_plot_iteration(plot=False):
    dp = DetectionPowerCounterFactual(significance_mode='paired-t-test',
                                      nsims=1000,
                                      p_value=0.05,
                                      min_samples=10,
                                      alternative='alt!=base',
                                      wx_zero_method='wilcox', wx_correction=False, wx_method='auto',
                                      ncores=None,
                                      return_true_conc=True,
                                      return_noisy_conc_itters=5,
                                      only_significant_noisy=False,
                                      )
    base, alt = make_bilinar_test_data(0.1, -0.1, 100)
    out = dp.power_calc(idv='test',
                        error_base=30,
                        error_alt=30,
                        true_conc_base=base,
                        true_conc_alt=alt,
                        seed_alt=1,
                        seed_base=2,
                        )
    for i in range(5):
        use_base = out['true_conc']['true_conc_base']
        use_alt = out['true_conc']['true_conc_alt']
        noisy_base = out['base_noisy_conc'].iloc[:, i]
        noisy_alt = out['alt_noisy_conc'].iloc[:, i]
        fig, ax = dp.plot_iteration(noisy_base, noisy_alt, use_base, use_alt)
        ax.set_title(f'itter {i}')

    if plot:
        plt.show()
    plt.close('all')


def test_power_calc_functionality():
    dp = DetectionPowerCounterFactual(significance_mode='paired-t-test',
                                      nsims=1000,
                                      p_value=0.05,
                                      min_samples=10,
                                      alternative='alt!=base',
                                      wx_zero_method='wilcox', wx_correction=False, wx_method='auto',
                                      ncores=None,
                                      return_true_conc=False,
                                      return_noisy_conc_itters=0,
                                      only_significant_noisy=False,
                                      )

    # test exception with two identical seeds
    ts1, ts2 = make_step_test_data(0.5, 100)
    test_passed = False
    try:
        dp.power_calc(idv='test',
                      error_base=1,
                      error_alt=1,
                      true_conc_base=ts1,
                      true_conc_alt=ts2,
                      seed_alt=1,
                      seed_base=1,
                      )
    except ValueError as val:
        test_passed = 'seed_base and seed_alt' in str(val)

    assert test_passed, 'seed_base and seed_alt the same should raise an error'

    #  test passing kwargs
    temp = dp.power_calc(idv='test',
                         error_base=1,
                         error_alt=1,
                         true_conc_base=ts1,
                         true_conc_alt=ts2,
                         seed_alt=1,
                         seed_base=2,
                         african_swallow='non-migratory',
                         )
    assert 'african_swallow' in temp.keys(), 'kwargs should be passed through power_calc'
    assert temp['african_swallow'] == 'non-migratory', 'kwargs should be passed through power_calc'

    # test return true conc and noisy conc, +- only signifcant noisy

    # True conc and noisy conc
    dp = DetectionPowerCounterFactual(significance_mode='paired-t-test',
                                      nsims=1000,
                                      p_value=0.05,
                                      min_samples=10,
                                      alternative='alt!=base',
                                      wx_zero_method='wilcox', wx_correction=False, wx_method='auto',
                                      ncores=None,
                                      return_true_conc=True,
                                      return_noisy_conc_itters=5,
                                      only_significant_noisy=False,
                                      )
    out = dp.power_calc(idv='test',
                        error_base=1,
                        error_alt=1,
                        true_conc_base=ts1,
                        true_conc_alt=ts2,
                        seed_alt=1,
                        seed_base=2,
                        )
    assert isinstance(out, dict)
    assert 'power' in out.keys(), 'power should be returned'
    assert 'true_conc' in out.keys(), 'true_conc should be returned'
    assert 'alt_noisy_conc' in out.keys(), 'alt_noisy_conc should be returned'
    assert 'base_noisy_conc' in out.keys(), 'base_noisy_conc should be returned'
    assert 'significant' in out.keys(), 'significant should be returned'
    assert isinstance(out['power'], pd.Series), 'power should be a series'
    assert np.in1d(
        ['power', 'idv', 'error_base',
         'error_alt', 'seed_base', 'seed_alt'], out['power'].index).all(), 'power should have correct index'
    assert isinstance(out['true_conc'], pd.DataFrame), 'true_conc should be a DataFrame'
    assert out['true_conc'].shape == (100, 2), 'true_conc should have correct shape'
    for k in ['alt_noisy_conc', 'base_noisy_conc']:
        assert isinstance(out[k], pd.DataFrame), f'{k} should be a DataFrame'
        assert out[k].shape == (100, 5), f'{k} should have correct shape'
    assert out['significant'].shape == (5,), 'significant should have correct shape'
    assert out['significant'].dtype == bool, 'significant should be a bool'

    # True conc and noisy conc, only significant
    dp = DetectionPowerCounterFactual(significance_mode='paired-t-test',
                                      nsims=1000,
                                      p_value=0.05,
                                      min_samples=10,
                                      alternative='alt!=base',
                                      wx_zero_method='wilcox', wx_correction=False, wx_method='auto',
                                      ncores=None,
                                      return_true_conc=True,
                                      return_noisy_conc_itters=5,
                                      only_significant_noisy=True,
                                      )
    out = dp.power_calc(idv='test',
                        error_base=1,
                        error_alt=1,
                        true_conc_base=ts1,
                        true_conc_alt=ts2,
                        seed_alt=1,
                        seed_base=2,
                        )
    assert out['significant'].shape == (5,), 'significant should have correct shape'
    assert out['significant'].dtype == bool, 'significant should be a bool'
    assert out['significant'].all(), 'all significant should be true'


def test_paired_ttest_power():
    save_path = Path(__file__).parent.joinpath('test_data', 'test_paired_ttest_power_counter.hdf')
    save_data = False
    got = []
    noises = [0.1, 1, 10, 100, 1000]
    data = {}
    base, alt = make_step_test_data(1, 100)
    data['step'] = (base, alt)
    base, alt = make_linear_test_data(0.1, 100)
    data['linear'] = (base, alt)
    base, alt = make_bilinar_test_data(0.1, -0.1, 100)
    data['bilinear'] = (base, alt)

    for alter in ['alt!=base', 'alt<base', 'alt>base']:
        for noise in noises:
            for dname, (base, alt) in data.items():
                dp = DetectionPowerCounterFactual(significance_mode='paired-t-test',
                                                  nsims=1000,
                                                  p_value=0.05,
                                                  min_samples=10,
                                                  alternative=alter,
                                                  wx_zero_method='wilcox', wx_correction=False, wx_method='auto',
                                                  ncores=None,
                                                  return_true_conc=False,
                                                  return_noisy_conc_itters=0,
                                                  only_significant_noisy=False,
                                                  )
                out = dp.power_calc(idv='test',
                                    error_base=noise,
                                    error_alt=noise,
                                    true_conc_base=base,
                                    true_conc_alt=alt,
                                    seed_alt=1,
                                    seed_base=2,
                                    )
                got.append([alter, noise, dname, out['power'], (alt > base).mean()])
    got = pd.DataFrame(got, columns=['alter', 'noise', 'dname', 'power', 'alt>base'])

    if save_data:
        got.to_hdf(save_path, key='data')
    expect = pd.read_hdf(save_path, key='data')
    assert isinstance(expect, pd.DataFrame)
    pd.testing.assert_frame_equal(got, expect, check_dtype=False, check_like=True, check_exact=False)


def test_wilcoxon_power():
    save_path = Path(__file__).parent.joinpath('test_data', 'test_wilcoxon_power_counter.hdf')
    save_data = False
    got = []
    noises = [0.1, 1, 10, 100, 1000]
    data = {}
    base, alt = make_step_test_data(1, 100)
    data['step'] = (base, alt)
    base, alt = make_linear_test_data(0.1, 100)
    data['linear'] = (base, alt)
    base, alt = make_bilinar_test_data(0.1, -0.1, 100)
    data['bilinear'] = (base, alt)

    for alter in ['alt!=base', 'alt<base', 'alt>base']:
        for noise in noises:
            for dname, (base, alt) in data.items():
                dp = DetectionPowerCounterFactual(significance_mode='wilcoxon-signed-rank-test',
                                                  nsims=1000,
                                                  p_value=0.05,
                                                  min_samples=10,
                                                  alternative=alter,
                                                  wx_zero_method='wilcox', wx_correction=False, wx_method='auto',
                                                  ncores=None,
                                                  return_true_conc=False,
                                                  return_noisy_conc_itters=0,
                                                  only_significant_noisy=False,
                                                  )
                out = dp.power_calc(idv='test',
                                    error_base=noise,
                                    error_alt=noise,
                                    true_conc_base=base,
                                    true_conc_alt=alt,
                                    seed_alt=1,
                                    seed_base=2,
                                    )
                got.append([alter, noise, dname, out['power'], (alt > base).mean()])
    got = pd.DataFrame(got, columns=['alter', 'noise', 'dname', 'power', 'alt>base'])

    if save_data:
        got.to_hdf(save_path, key='data')

    expect = pd.read_hdf(save_path, key='data')
    assert isinstance(expect, pd.DataFrame)
    pd.testing.assert_frame_equal(got, expect, check_dtype=False, check_like=True, check_exact=False)


def test_multiprocess_power_calc():
    got = []
    noise_alt = [0.1, 1, 10]
    noises = [0.1, 1, 10, 100, 1000]
    data = {}
    base, alt = make_step_test_data(1, 100)
    data['step'] = (base, alt)
    base, alt = make_linear_test_data(0.1, 100)
    data['linear'] = (base, alt)
    base, alt = make_bilinar_test_data(0.1, -0.1, 100)
    data['bilinear'] = (base, alt)
    dp = DetectionPowerCounterFactual(significance_mode='wilcoxon-signed-rank-test',
                                      nsims=1000,
                                      p_value=0.05,
                                      min_samples=10,
                                      alternative='alt!=base',
                                      wx_zero_method='wilcox', wx_correction=False, wx_method='auto',
                                      ncores=None,
                                      return_true_conc=False,
                                      return_noisy_conc_itters=0,
                                      only_significant_noisy=False,
                                      )
    errors_base = []
    errors_alt = []
    alt_ts = []
    base_ts = []
    idvs = []
    t = ("I'm a fast cook I guess. Sorry, I was all the way over "
         "there... you're a fast cook that's it. Were these magic grits? "
         "Did you get these from the same guy who sold Jack his beanstalk beans? "
         "Objection! The witness may disregard the question")
    unique_kwargs = t.split(' ')

    i = 0
    for n_alt in noise_alt:
        for noise in noises:
            for dname, (base, alt) in data.items():
                print(i)
                errors_base.append(noise)
                errors_alt.append(n_alt)
                alt_ts.append(alt)
                base_ts.append(base)
                idv = f'{dname}_{noise}'
                idvs.append(idv)
                out = dp.power_calc(idv=idv,
                                    error_base=noise,
                                    error_alt=n_alt,
                                    true_conc_base=base,
                                    true_conc_alt=alt,
                                    seed_alt=1,
                                    seed_base=2,
                                    single_kwarg='test_single_kwarg',
                                    mult_kwargs=unique_kwargs[i]
                                    )
                got.append(out)
                i += 1
    got = pd.DataFrame(got)

    assert len(unique_kwargs) == len(idvs), 'bad test design'

    # multiprocess
    out = dp.mulitprocess_power_calcs(None,
                                      idv_vals=np.array(idvs),
                                      true_conc_base_vals=base_ts,
                                      true_conc_alt_vals=alt_ts,
                                      error_base_vals=np.array(errors_base),
                                      error_alt_vals=np.array(errors_alt),
                                      seed_alt_vals_vals=1,
                                      seed_base_vals_vals=2,
                                      run=False, debug_mode=False,
                                      single_kwarg='test_single_kwarg',
                                      mult_kwargs=unique_kwargs,
                                      )
    assert out is None, 'run=False should return None'

    out = dp.mulitprocess_power_calcs(None,
                                      idv_vals=np.array(idvs),
                                      true_conc_base_vals=base_ts,
                                      true_conc_alt_vals=alt_ts,
                                      error_base_vals=np.array(errors_base),
                                      error_alt_vals=np.array(errors_alt),
                                      seed_alt_vals_vals=1,
                                      seed_base_vals_vals=2,
                                      run=True, debug_mode=False,
                                      single_kwarg='test_single_kwarg',
                                      mult_kwargs=unique_kwargs,
                                      )

    out2 = dp.mulitprocess_power_calcs(None,
                                       idv_vals=np.array(idvs),
                                       true_conc_base_vals=base_ts,
                                       true_conc_alt_vals=alt_ts,
                                       error_base_vals=np.array(errors_base),
                                       error_alt_vals=np.array(errors_alt),
                                       seed_alt_vals_vals=1,
                                       seed_base_vals_vals=2,
                                       run=True, debug_mode=True,
                                       single_kwarg='test_single_kwarg',
                                       mult_kwargs=unique_kwargs,
                                       )

    out = out.reset_index()
    out2 = out2.reset_index()
    pd.testing.assert_frame_equal(out, out2, check_dtype=False, check_like=True, check_exact=False)
    pd.testing.assert_frame_equal(out, got, check_dtype=False, check_like=True, check_exact=False)
    assert all(out['single_kwarg'] == 'test_single_kwarg')
    assert set(out['mult_kwargs']) == set(unique_kwargs)


def test_auto_true_conc(plot=False):
    save_path = Path(__file__).parent.joinpath('test_data', 'test_auto_true_conc_counter.hdf')
    save_data = False
    dp_auto = AutoDetectionPowerCounterFactual(significance_mode='wilcoxon-signed-rank-test',
                                               nsims=1000,
                                               p_value=0.05,
                                               min_samples=10,
                                               alternative='alt!=base',
                                               wx_zero_method='wilcox', wx_correction=False, wx_method='auto',
                                               ncores=None,
                                               return_true_conc=True,
                                               return_noisy_conc_itters=0,
                                               only_significant_noisy=False,
                                               )
    delays = [0, 5]
    targ_bases = [None, 12]
    targ_alts = [6, 8]
    imp_bases = [None, 4]
    imp_alts = [4, 6]
    got = {}
    for delay, targ_base, targ_alt, imp_base, imp_alt in itertools.product(delays, targ_bases, targ_alts, imp_bases,
                                                                           imp_alts):
        idv = (f'{delay=}\n'
               f'{targ_base=}_{imp_base=}\n'
               f'{targ_alt=}_{imp_alt=}')
        out = dp_auto.power_calc(
            idv=idv,
            implementation_time_alt=imp_alt,
            target_conc_alt=targ_alt,
            target_conc_base=targ_base,
            implementation_time_base=imp_base,
            delay_years=delay,
            error_base=0,
            mrt_model='piston_flow',
            samp_years=10,
            samp_per_year=5,
            initial_conc=10,
            prev_slope=0,
            max_conc_lim=20,
            min_conc_lim=1,
            mrt=0,
            error_alt=None,
            mrt_p1=0,
            frac_p1=0,
            f_p1=0,
            f_p2=0,
            seed_base=1,
            seed_alt=2,
        )
        got[idv] = t = out['true_conc']

        if plot:
            fig, ax = plt.subplots()
            ax.plot(t.index, t['true_conc_alt'].values, label='alt_conc', marker='o', color='r')
            ax.plot(t.index, t['true_conc_base'].values, label='base_conc', marker='o', color='b')
            ax.legend()
            ax.set_title(idv)
    if plot:
        plt.show()
        plt.close('all')

    if save_data:
        for k, v in got.items():
            v.to_hdf(save_path, key=k)

    for k, v in got.items():
        expect = pd.read_hdf(save_path, key=k)
        assert isinstance(expect, pd.DataFrame)
        pd.testing.assert_frame_equal(v, expect, check_dtype=False, check_like=True, check_exact=False)


def test_power_calc_auto():
    save_path = Path(__file__).parent.joinpath('test_data', 'test_power_calc_auto_counter.hdf')
    save_data = False
    got = []
    noises = [0.1, 1, 10, 100, 1000]
    delay = [0, 1, 5]
    targets = [9, 7, 5]
    dp_auto = AutoDetectionPowerCounterFactual(significance_mode='wilcoxon-signed-rank-test',
                                               nsims=1000,
                                               p_value=0.05,
                                               min_samples=10,
                                               alternative='alt!=base',
                                               wx_zero_method='wilcox', wx_correction=False, wx_method='auto',
                                               ncores=None,
                                               return_true_conc=False,
                                               return_noisy_conc_itters=0,
                                               only_significant_noisy=False,
                                               )

    i = 0
    errors = []
    use_targs = []
    use_delays = []
    idvs = []

    for dl in delay:
        for noise in noises:
            for targ in targets:
                print(i)
                errors.append(noise)
                idv = f'{targ}_{noise}_delay_{dl}'
                idvs.append(idv)
                use_targs.append(targ)
                use_delays.append(dl)

                out = dp_auto.power_calc(
                    idv=idv,
                    error_base=noise,
                    mrt_model='binary_exponential_piston_flow',
                    samp_years=10,
                    samp_per_year=5,
                    implementation_time_alt=5,
                    initial_conc=10,
                    target_conc_alt=targ,
                    prev_slope=0,
                    max_conc_lim=20,
                    min_conc_lim=1,
                    mrt=3,
                    target_conc_base=None,
                    implementation_time_base=None,
                    error_alt=None,
                    delay_years=dl,
                    mrt_p1=3,
                    frac_p1=1,
                    f_p1=0.7,
                    f_p2=0.7,
                    seed_base=1,
                    seed_alt=2,
                )
                got.append(out)
                i += 1
    got = pd.DataFrame(got)

    if save_data:
        got.to_hdf(save_path, key='data')
    expect = pd.read_hdf(save_path, key='data')
    assert isinstance(expect, pd.DataFrame)
    pd.testing.assert_frame_equal(got, expect, check_dtype=False, check_like=True, check_exact=False)


def test_multiprocess_power_calc_auto():
    t = ("I'm a fast cook I guess. Sorry, I was all the way over "
         "there... you're a fast cook that's it. Were these magic grits? "
         "Did you get these from the same guy who sold Jack his beanstalk beans? "
         "Objection! The witness may disregard the question")
    unique_kwargs = t.split(' ')

    got = []
    noises = [0.1, 1, 10, 100, 1000]
    delay = [0, 1, 5]
    targets = [9, 7, 5]
    dp_auto = AutoDetectionPowerCounterFactual(significance_mode='wilcoxon-signed-rank-test',
                                               nsims=1000,
                                               p_value=0.05,
                                               min_samples=10,
                                               alternative='alt!=base',
                                               wx_zero_method='wilcox', wx_correction=False, wx_method='auto',
                                               ncores=None,
                                               return_true_conc=False,
                                               return_noisy_conc_itters=0,
                                               only_significant_noisy=False,
                                               )

    i = 0
    errors = []
    use_targs = []
    use_delays = []
    idvs = []

    for dl in delay:
        for noise in noises:
            for targ in targets:
                print(i)
                errors.append(noise)
                idv = f'{targ}_{noise}_delay_{dl}'
                idvs.append(idv)
                use_targs.append(targ)
                use_delays.append(dl)

                out = dp_auto.power_calc(
                    idv=idv,
                    error_base=noise,
                    mrt_model='binary_exponential_piston_flow',
                    samp_years=10,
                    samp_per_year=5,
                    implementation_time_alt=5,
                    initial_conc=10,
                    target_conc_alt=targ,
                    prev_slope=0,
                    max_conc_lim=20,
                    min_conc_lim=1,
                    mrt=3,
                    target_conc_base=None,
                    implementation_time_base=None,
                    error_alt=None,
                    delay_years=dl,
                    mrt_p1=3,
                    frac_p1=1,
                    f_p1=0.7,
                    f_p2=0.7,
                    seed_base=1,
                    seed_alt=2,
                    multi_kwargs=unique_kwargs[i],
                    single_kwarg='test_single_kwarg'
                )
                got.append(out)
                i += 1
    got = pd.DataFrame(got)

    # multiprocessing
    out = dp_auto.mulitprocess_power_calcs(None,
                                           idv_vals=np.array(idvs),
                                           error_base_vals=np.array(errors),
                                           run=True, debug_mode=False,

                                           mrt_model_vals='binary_exponential_piston_flow',
                                           samp_years_vals=10,
                                           samp_per_year_vals=5,
                                           implementation_time_alt_vals=5,
                                           initial_conc_vals=10,
                                           target_conc_alt_vals=use_targs,
                                           prev_slope_vals=0,
                                           max_conc_lim_vals=20,
                                           min_conc_lim_vals=1,
                                           mrt_vals=3,
                                           target_conc_base_vals=None,
                                           implementation_time_base_vals=None,
                                           error_alt_vals=None,
                                           delay_years_vals=use_delays,
                                           mrt_p1_vals=3,
                                           frac_p1_vals=1,
                                           f_p1_vals=0.7,
                                           f_p2_vals=0.7,
                                           seed_base_vals=1,
                                           seed_alt_vals=2,
                                           multi_kwargs=unique_kwargs,
                                           single_kwarg='test_single_kwarg',
                                           )
    out = out.reset_index()
    pd.testing.assert_frame_equal(out, got, check_dtype=False, check_like=True, check_exact=False)
    assert all(out['single_kwarg'] == 'test_single_kwarg')
    assert set(out['multi_kwargs']) == set(unique_kwargs)


def test_condenced_non_condenced():
    save_path = Path(__file__).parent.joinpath('test_data', 'test_condenced_non_condenced_counter.hdf')
    save_data = False
    kwargs = dict(
        outpath=None,
        idv_vals=np.arange(9),
        error_base_vals=np.array([5.1, 5.101, 5.10001] * 3),
        max_conc_lim_vals=np.array([20.001, 20, 20.1] * 3),
        initial_conc_vals=np.array([10, 10.001, 10.1] * 3),
        target_conc_alt_vals=np.array([6, 6.001, 6.1] * 3),
        mrt_vals=np.array([1.4, 1.1, 1.0] * 3),
        mrt_p1_vals=np.array([1.4, 1.1, 1.0] * 3),

        samp_years_vals=10,
        samp_per_year_vals=5,
        implementation_time_alt_vals=5,
        prev_slope_vals=0,
        delay_years_vals=0,
        implementation_time_base_vals=None,
        min_conc_lim_vals=1,
        target_conc_base_vals=None,
        error_alt_vals=None,

        mrt_model_vals='binary_exponential_piston_flow',
        run=True, debug_mode=False,
        frac_p1_vals=1,
        f_p1_vals=0.7,
        f_p2_vals=0.7,
        seed_base_vals=1,
        seed_alt_vals=2,
    )

    dp_auto = AutoDetectionPowerCounterFactual(significance_mode='wilcoxon-signed-rank-test',
                                               nsims=1000,
                                               p_value=0.05,
                                               min_samples=10,
                                               alternative='alt!=base',
                                               wx_zero_method='wilcox', wx_correction=False, wx_method='auto',
                                               ncores=None,
                                               return_true_conc=False,
                                               return_noisy_conc_itters=0,
                                               only_significant_noisy=False,
                                               )

    out_raw = dp_auto.mulitprocess_power_calcs(**kwargs)

    dp_auto.set_condensed_mode(
        target_conc_per=0,
        initial_conc_per=0,
        error_per=1,
        prev_slope_per=2,
        max_conc_lim_per=0,
        min_conc_lim_per=0,
        mrt_per=0,
        mrt_p1_per=0,
        frac_p1_per=2,
        f_p1_per=2,
        f_p2_per=2)

    out_condensed = dp_auto.mulitprocess_power_calcs(**kwargs)

    # check condensed vs non-condensed
    if save_data:
        out_raw.to_hdf(save_path, key='raw')
        out_condensed.to_hdf(save_path, key='condensed')

    expect_raw = pd.read_hdf(save_path, key='raw')
    assert isinstance(expect_raw, pd.DataFrame)
    expect_condensed = pd.read_hdf(save_path, key='condensed')
    assert isinstance(expect_condensed, pd.DataFrame)
    pd.testing.assert_frame_equal(out_raw, expect_raw, check_dtype=False, check_like=True, check_exact=False)
    pd.testing.assert_frame_equal(out_condensed, expect_condensed, check_dtype=False, check_like=True,
                                  check_exact=False)


if __name__ == '__main__':
    plt_flag = False
    test_power_calc_auto()
    test_paired_ttest_power()
    test_wilcoxon_power()
    test_condenced_non_condenced()
    test_multiprocess_power_calc_auto()
    test_multiprocess_power_calc()
    test_power_calc_functionality()
    test_auto_true_conc(plot=plt_flag)
    test_plot_iteration(plt_flag)

    pass
