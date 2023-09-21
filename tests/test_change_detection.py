"""
created matt_dumont 
on: 17/07/23
"""
import itertools
import time
import pandas as pd
import numpy as np
from pathlib import Path
from gw_detect_power import DetectionPowerCalculator

def test_unitary_epfm(plot=False):
    example = DetectionPowerCalculator()
    (out_conc, conc_max, max_conc_time,
     frac_p2, total_source_conc,
     age_fractions, out_years, ages, past_conc) = example.truets_from_binary_exp_piston_flow(
        mrt=10,
        mrt_p1=10,
        frac_p1=1,
        f_p1=0.8,
        f_p2=0.8,
        initial_conc=10,
        target_conc=5,
        prev_slope=0,
        max_conc=200,
        min_conc=1,
        samp_per_year=10,
        samp_years=20,
        implementation_time=5,
        return_extras=True,
        low_mem=True
    )
    (out_conc2, conc_max2, max_conc_time2,
     frac_p22) = example.truets_from_binary_exp_piston_flow(
        mrt=10,
        mrt_p1=10,
        frac_p1=1,
        f_p1=0.8,
        f_p2=0.8,
        initial_conc=10,
        target_conc=5,
        prev_slope=0,
        max_conc=200,
        min_conc=1,
        samp_per_year=10,
        samp_years=20,
        implementation_time=5,
        return_extras=False,
        low_mem=False
    )
    if plot:
        import matplotlib.pyplot as plt
        fig, (ax, ax2) = plt.subplots(nrows=2, figsize=(10, 10))
        ax.plot(out_years, out_conc, marker='o', label='out_conc', color='r')
        ax.plot(total_source_conc.index, total_source_conc, marker='o', label='source_conc', color='b')
        ax.plot(past_conc.index, past_conc, marker='o', label='past_conc', color='pink')
        ax.set_ylabel('Concentration')
        ax.set_xlabel('Years')
        ax.legend()

        ax2.plot(ages, age_fractions, marker='o', label='age_fractions', color='g')
        ax2.set_ylabel('Fraction')
        ax2.set_xlabel('Years')
        ax2.legend()
        fig.tight_layout()
        plt.show()
    out_conc = pd.Series(index=out_years, data=out_conc)
    age_fractions = pd.Series(index=ages, data=age_fractions)
    test_data_path = Path(__file__).parent.joinpath('test_data', 'test_unitary_epfm.hdf')
    write_test_data = False
    if write_test_data:
        test_data_path.unlink(missing_ok=True)
        out_conc.to_hdf(test_data_path, 'out_conc')
        age_fractions.to_hdf(test_data_path, 'age_fractions')
        total_source_conc.to_hdf(test_data_path, 'total_source_conc')
        past_conc.to_hdf(test_data_path, 'past_conc')
    true_out_conc = pd.read_hdf(test_data_path, 'out_conc')
    true_age_fractions = pd.read_hdf(test_data_path, 'age_fractions')
    true_total_source_conc = pd.read_hdf(test_data_path, 'total_source_conc')
    true_past_conc = pd.read_hdf(test_data_path, 'past_conc')
    assert np.allclose(out_conc, true_out_conc)
    assert np.allclose(age_fractions, true_age_fractions)
    assert np.allclose(past_conc, true_past_conc)
    assert np.allclose(total_source_conc, true_total_source_conc)
    assert np.allclose(out_conc2, out_conc)


def test_unitary_epfm_slope(plot=False):
    example = DetectionPowerCalculator()
    (out_conc, conc_max, max_conc_time,
     frac_p2, total_source_conc,
     age_fractions, out_years, ages, past_conc) = example.truets_from_binary_exp_piston_flow(
        mrt=10,
        mrt_p1=10,
        frac_p1=1,
        f_p1=0.8,
        f_p2=0.8,
        initial_conc=10,
        target_conc=5,
        prev_slope=0.5,
        max_conc=20,
        min_conc=1.,
        samp_per_year=10,
        samp_years=20,
        implementation_time=5,
        return_extras=True,
        low_mem=True
    )
    (out_conc2, conc_max2, max_conc_time2,
     frac_p22) = example.truets_from_binary_exp_piston_flow(
        mrt=10,
        mrt_p1=10,
        frac_p1=1,
        f_p1=0.8,
        f_p2=0.8,
        initial_conc=10,
        target_conc=5,
        prev_slope=0.5,
        max_conc=20,
        min_conc=1.,
        samp_per_year=10,
        samp_years=20,
        implementation_time=5,
        return_extras=False,
        low_mem=False
    )
    if plot:
        import matplotlib.pyplot as plt
        fig, (ax, ax2) = plt.subplots(nrows=2, figsize=(10, 10))
        ax.plot(out_years, out_conc, marker='o', label='out_conc', color='r')
        ax.plot(total_source_conc.index, total_source_conc, marker='o', label='source_conc', color='b')
        ax.plot(past_conc.index, past_conc, marker='o', label='past_conc', color='pink')
        ax.axvline(0, color='k', linestyle='--', label='present')
        ax.set_ylabel('Concentration')
        ax.set_xlabel('Years')
        ax.legend()

        ax2.plot(ages, age_fractions, marker='o', label='age_fractions', color='g')
        ax2.set_ylabel('Fraction')
        ax2.set_xlabel('Years')
        ax2.legend()
        fig.tight_layout()
        plt.show()
    out_conc = pd.Series(index=out_years, data=out_conc)
    age_fractions = pd.Series(index=ages, data=age_fractions)
    test_data_path = Path(__file__).parent.joinpath('test_data', 'test_unitary_epfm_slope.hdf')
    write_test_data = False
    if write_test_data:
        test_data_path.unlink(missing_ok=True)
        out_conc.to_hdf(test_data_path, 'out_conc')
        past_conc.to_hdf(test_data_path, 'past_conc')
        total_source_conc.to_hdf(test_data_path, 'total_source_conc')
        age_fractions.to_hdf(test_data_path, 'age_fractions')
    true_out_conc = pd.read_hdf(test_data_path, 'out_conc')
    true_past_conc = pd.read_hdf(test_data_path, 'past_conc')
    true_total_source_conc = pd.read_hdf(test_data_path, 'total_source_conc')
    true_age_fractions = pd.read_hdf(test_data_path, 'age_fractions')
    assert np.allclose(out_conc, true_out_conc)
    assert np.allclose(past_conc, true_past_conc)
    assert np.allclose(age_fractions, true_age_fractions)
    assert np.allclose(total_source_conc, true_total_source_conc)
    assert np.allclose(out_conc2, out_conc)


def test_piston_flow(plot=False):
    example = DetectionPowerCalculator()
    true_conc_ts, max_conc, max_conc_time, frac_p2 = example.truets_from_piston_flow(mrt=10, initial_conc=5,
                                                                                     target_conc=2.5,
                                                                                     prev_slope=1,
                                                                                     max_conc=10, samp_per_year=4,
                                                                                     samp_years=20,
                                                                                     implementation_time=7)
    true_conc_ts_org, max_conc_org, max_conc_time_org, frac_p2_org = np.array(
        [5.0, 5.256410256410256, 5.512820512820513, 5.769230769230769, 6.0256410256410255, 6.282051282051282,
         6.538461538461538, 6.794871794871795, 7.051282051282051, 7.3076923076923075, 7.564102564102564,
         7.82051282051282, 8.076923076923077, 8.333333333333332, 8.58974358974359, 8.846153846153847, 9.102564102564102,
         9.358974358974358, 9.615384615384615, 9.871794871794872, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.722222222222221, 9.444444444444445,
         9.166666666666666, 8.88888888888889, 8.61111111111111, 8.333333333333334, 8.055555555555555, 7.777777777777778,
         7.5, 7.222222222222222, 6.944444444444445, 6.666666666666666, 6.388888888888889, 6.111111111111111,
         5.833333333333333, 5.555555555555555, 5.277777777777778, 5.0, 4.722222222222222, 4.444444444444445,
         4.166666666666666, 3.8888888888888884, 3.6111111111111107, 3.333333333333333, 3.0555555555555554,
         2.7777777777777777, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]), 10.0, 5.0, None
    assert np.allclose(true_conc_ts, true_conc_ts_org)
    assert np.allclose(max_conc, max_conc_org)
    assert np.allclose(max_conc_time, max_conc_time_org)
    assert frac_p2_org is None and frac_p2 is None

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(true_conc_ts)) / 4, true_conc_ts, marker='o')
        plt.axhline(max_conc, color='k', linestyle='--')
        plt.axvline(max_conc_time, color='k', linestyle='--')
        plt.axvline(10, color='k', linestyle=':')
    true_conc_ts, max_conc, max_conc_time, frac_p2 = example.truets_from_piston_flow(mrt=10, initial_conc=5,
                                                                                     target_conc=2.5,
                                                                                     prev_slope=1,
                                                                                     max_conc=10, samp_per_year=4,
                                                                                     samp_years=15,
                                                                                     implementation_time=7)
    true_conc_ts_org, max_conc_org, max_conc_time_org, frac_p2_org = np.array(
        [5.0, 5.256410256410256, 5.512820512820513, 5.769230769230769, 6.0256410256410255, 6.282051282051282,
         6.538461538461538, 6.794871794871795, 7.051282051282051, 7.3076923076923075, 7.564102564102564,
         7.82051282051282,
         8.076923076923077, 8.333333333333332, 8.58974358974359, 8.846153846153847, 9.102564102564102,
         9.358974358974358,
         9.615384615384615, 9.871794871794872, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.722222222222221, 9.444444444444445, 9.166666666666666,
         8.88888888888889, 8.61111111111111, 8.333333333333334, 8.055555555555555, 7.777777777777778, 7.5,
         7.222222222222222, 6.944444444444445, 6.666666666666666, 6.388888888888889, 6.111111111111111,
         5.833333333333333,
         5.555555555555555, 5.277777777777778, 5.0, 4.722222222222222]), 10.0, 5.0, None
    assert np.allclose(true_conc_ts, true_conc_ts_org)
    assert np.allclose(max_conc, max_conc_org)
    assert np.allclose(max_conc_time, max_conc_time_org)
    assert frac_p2_org is None and frac_p2 is None
    if plot:
        plt.plot(np.arange(len(true_conc_ts)) / 4, true_conc_ts, marker='o')


def test_bepfm_slope(plot=False):
    example = DetectionPowerCalculator()
    (out_conc, conc_max, max_conc_time,
     frac_p2, total_source_conc,
     age_fractions, out_years, ages, past_conc) = example.truets_from_binary_exp_piston_flow(
        mrt=20,
        mrt_p1=5,
        frac_p1=.25,
        f_p1=0.8,
        f_p2=0.8,
        initial_conc=10,
        target_conc=5,
        prev_slope=0.5,
        max_conc=20,
        min_conc=1.,
        samp_per_year=10,
        samp_years=20,
        implementation_time=5,
        return_extras=True,
        low_mem=True
    )
    (out_conc2, conc_max2, max_conc_time2,
     frac_p22) = example.truets_from_binary_exp_piston_flow(
        mrt=20,
        mrt_p1=5,
        frac_p1=.25,
        f_p1=0.8,
        f_p2=0.8,
        initial_conc=10,
        target_conc=5,
        prev_slope=0.5,
        max_conc=20,
        min_conc=1.,
        samp_per_year=10,
        samp_years=20,
        implementation_time=5,
        return_extras=False,
        low_mem=False
    )
    if plot:
        import matplotlib.pyplot as plt
        fig, (ax, ax2) = plt.subplots(nrows=2, figsize=(10, 10))
        ax.plot(out_years, out_conc, marker='o', label='out_conc', color='r')
        ax.plot(total_source_conc.index, total_source_conc, marker='o', label='source_conc', color='b')
        ax.plot(past_conc.index, past_conc, marker='o', label='past_conc', color='pink')
        ax.axvline(0, color='k', linestyle='--', label='present')
        ax.set_ylabel('Concentration')
        ax.set_xlabel('Years')
        ax.legend()

        ax2.plot(ages, age_fractions, marker='o', label='age_fractions', color='g')
        ax2.set_ylabel('Fraction')
        ax2.set_xlabel('Years')
        ax2.legend()
        fig.tight_layout()
        plt.show()
    out_conc = pd.Series(index=out_years, data=out_conc)
    age_fractions = pd.Series(index=ages, data=age_fractions)
    test_data_path = Path(__file__).parent.joinpath('test_data', 'test_bepfm_slope.hdf')
    write_test_data = False
    if write_test_data:
        test_data_path.unlink(missing_ok=True)
        out_conc.to_hdf(test_data_path, 'out_conc')
        past_conc.to_hdf(test_data_path, 'past_conc')
        total_source_conc.to_hdf(test_data_path, 'total_source_conc')
        age_fractions.to_hdf(test_data_path, 'age_fractions')
    true_out_conc = pd.read_hdf(test_data_path, 'out_conc')
    true_past_conc = pd.read_hdf(test_data_path, 'past_conc')
    true_total_source_conc = pd.read_hdf(test_data_path, 'total_source_conc')
    true_age_fractions = pd.read_hdf(test_data_path, 'age_fractions')
    assert np.allclose(out_conc, true_out_conc)
    assert np.allclose(past_conc, true_past_conc)
    assert np.allclose(age_fractions, true_age_fractions)
    assert np.allclose(total_source_conc, true_total_source_conc)
    assert np.allclose(out_conc, out_conc2)


def test_bpefm(plot=False):
    example = DetectionPowerCalculator()
    (out_conc, conc_max, max_conc_time,
     frac_p2, total_source_conc,
     age_fractions, out_years, ages, past_conc) = example.truets_from_binary_exp_piston_flow(
        mrt=10,
        mrt_p1=2.5,
        frac_p1=.75,
        f_p1=0.8,
        f_p2=0.8,
        initial_conc=10,
        target_conc=5,
        prev_slope=0,
        max_conc=200,
        min_conc=1,
        samp_per_year=2,
        samp_years=50,
        implementation_time=5,
        return_extras=True,
        low_mem=True
    )
    (out_conc2, conc_max2, max_conc_time2,
     frac_p22) = example.truets_from_binary_exp_piston_flow(
        mrt=10,
        mrt_p1=2.5,
        frac_p1=.75,
        f_p1=0.8,
        f_p2=0.8,
        initial_conc=10,
        target_conc=5,
        prev_slope=0,
        max_conc=200,
        min_conc=1,
        samp_per_year=2,
        samp_years=50,
        implementation_time=5,
        return_extras=False,
        low_mem=False
    )
    if plot:
        import matplotlib.pyplot as plt
        fig, (ax, ax2) = plt.subplots(nrows=2, figsize=(10, 10))
        ax.plot(out_years, out_conc, marker='o', label='out_conc', color='r')
        ax.plot(total_source_conc.index, total_source_conc, marker='o', label='source_conc', color='b')
        ax.plot(past_conc.index, past_conc, marker='o', label='past_conc', color='pink')
        ax.set_ylabel('Concentration')
        ax.set_xlabel('Years')
        ax.set_xlim(-100, 50)
        ax.legend()

        ax2.plot(ages, age_fractions, marker='o', label='age_fractions', color='g')
        ax2.set_ylabel('Fraction')
        ax2.set_xlabel('Years')
        ax2.legend()
        fig.tight_layout()
        plt.show()
    out_conc = pd.Series(index=out_years, data=out_conc)
    age_fractions = pd.Series(index=ages, data=age_fractions)
    test_data_path = Path(__file__).parent.joinpath('test_data', 'test_bpefm.hdf')
    write_test_data = False
    if write_test_data:
        test_data_path.unlink(missing_ok=True)
        out_conc.to_hdf(test_data_path, 'out_conc')
        age_fractions.to_hdf(test_data_path, 'age_fractions')
        total_source_conc.to_hdf(test_data_path, 'total_source_conc')
        past_conc.to_hdf(test_data_path, 'past_conc')
    true_out_conc = pd.read_hdf(test_data_path, 'out_conc')
    true_age_fractions = pd.read_hdf(test_data_path, 'age_fractions')
    true_total_source_conc = pd.read_hdf(test_data_path, 'total_source_conc')
    true_past_conc = pd.read_hdf(test_data_path, 'past_conc')
    assert np.allclose(out_conc, true_out_conc)
    assert np.allclose(age_fractions, true_age_fractions)
    assert np.allclose(past_conc, true_past_conc)
    assert np.allclose(total_source_conc, true_total_source_conc)
    assert np.allclose(out_conc, out_conc2)


def make_test_power_calc_runs(plot=False):
    runs = []
    errors = [0.5, 1.5, 2, 4]
    samp_per_yr = [4, 12, 52]
    samp_yrs = [5, 10, 20]
    implementation_times = [5, 20, 100]
    initial_cons = [7, 15]
    target_cons = [2.4]
    max_cons = [25]
    min_cons = [1]
    prev_slopes = [0, 0.5]
    mrt_models = ['piston_flow', 'binary_exponential_piston_flow']
    mrt = [0, 5]
    mrt_p1 = [2.5]
    frac_p1 = [0.75]
    f_p1 = [0.8]
    f_p2 = [0.75]

    arg_vals = [
        errors, samp_per_yr, samp_yrs, implementation_times, initial_cons, target_cons, max_cons, min_cons, prev_slopes,
        mrt_models, mrt, mrt_p1, frac_p1, f_p1, f_p2
    ]
    seed_val = 5548
    for i, vals in enumerate(itertools.product(*arg_vals)):
        e, spy, sy, it, ic, tc, mc, mic, ps, mrtm, mrt, mrt_p1, frac_p1, f_p1, f_p2 = vals
        if mrtm == 'piston_flow':
            mrt_p1, frac_p1, f_p1, f_p2 = None, None, None, None
        else:
            if mrt == 0:
                continue

        temp = dict(
            idv=str(i),
            error=e,
            samp_per_year=spy,
            samp_years=sy,
            implementation_time=it,
            initial_conc=ic,
            target_conc=tc,
            max_conc=mc,
            min_conc=mic,
            prev_slope=ps,
            mrt_model=mrtm,
            mrt=mrt,
            mrt_p1=mrt_p1,
            frac_p1=frac_p1,
            f_p1=f_p1,
            f_p2=f_p2,
            seed=seed_val,
        )
        runs.append(temp)

    # test pass conc ts
    errors = [0.5, 1.5, 2, 4, 7]
    data = pd.Series(index=np.arange(0, 15, .25))
    data.loc[0] = 10
    data.loc[5] = 9
    data.loc[10] = 3
    data.loc[15] = 2.4
    data = data.interpolate()
    data2 = data.loc[np.arange(0, 6, 0.5)] - 0.5
    assert len(data2) > 10
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(data.index, data, marker='o', c='r', label='data')
        ax.plot(data2.index, data2, marker='o', c='b', label='data2')
        ax.legend()
        ax.set_ylabel('Concentration')
        ax.set_xlabel('Years')
        plt.show()
    true_tss = [data, data2]

    for i, (e, tts) in enumerate(itertools.product(errors, true_tss)):
        temp = dict(
            idv=f'tts_{i}',
            error=e,
            true_conc_ts=tts.copy(),
            mrt_model='pass_true_conc',
            seed=seed_val,
        )
        runs.append(temp)

    print(len(runs))

    return runs


def test_power_calc_and_mp():
    save_path = Path(__file__).parent.joinpath('test_data', 'test_power_calc_and_mp.hdf')
    write_test_data = False
    ex = DetectionPowerCalculator()
    runs = make_test_power_calc_runs()

    t = time.time()
    mp_data = ex.mulitprocess_power_calcs(
        outpath=None,
        id_vals=np.array([r.get('idv') for r in runs]),
        error_vals=np.array([r.get('error') for r in runs]),
        samp_years_vals=np.array([r.get('samp_years') for r in runs]),
        samp_per_year_vals=np.array([r.get('samp_per_year') for r in runs]),
        implementation_time_vals=np.array([r.get('implementation_time') for r in runs]),
        initial_conc_vals=np.array([r.get('initial_conc') for r in runs]),
        target_conc_vals=np.array([r.get('target_conc') for r in runs]),
        previous_slope_vals=np.array([r.get('prev_slope') for r in runs]),
        max_conc_vals=np.array([r.get('max_conc') for r in runs]),
        min_conc_vals=np.array([r.get('min_conc') for r in runs]),
        mrt_model_vals=np.array([r.get('mrt_model') for r in runs]),
        mrt_vals=np.array([r.get('mrt') for r in runs]),
        mrt_p1_vals=np.array([r.get('mrt_p1') for r in runs]),
        frac_p1_vals=np.array([r.get('frac_p1') for r in runs]),
        f_p1_vals=np.array([r.get('f_p1') for r in runs]),
        f_p2_vals=np.array([r.get('f_p2') for r in runs]),
        true_conc_ts_vals=[r.get('true_conc_ts') for r in runs],
        seed=np.array([r.get('seed') for r in runs]),
    )
    print(f'elapsed time for mp: {time.time() - t}')

    print('running non-mp this takes c. 8-10 mins')
    data = []
    t = time.time()
    for i, run in enumerate(runs):
        if i % 50 == 0:
            print(f'starting run {i + 1} of {len(runs)}')
        out = ex.power_calc(**run)
        data.append(out)
    data = pd.DataFrame(data)
    data.set_index('idv', inplace=True)
    print(f'elapsed time for non-mp: {time.time() - t}')

    if write_test_data:
        save_path.unlink(missing_ok=True)
        data.to_hdf(save_path, key='data', mode='w')
        mp_data.to_hdf(save_path, key='mp_data', mode='a')
    true_data = pd.read_hdf(save_path, key='data')
    true_mp_data = pd.read_hdf(save_path, key='mp_data')

    assert data.shape == mp_data.shape == true_data.shape == true_mp_data.shape, 'data shapes do not match'
    assert set(data.columns) == set(mp_data.columns) == set(true_data.columns) == set(
        true_mp_data.columns), 'data columns do not match'
    bad_cols = []
    for col in data.columns:
        if col in ['mrt_model', 'python_error']:
            col_same = (data[col].equals(mp_data[col])
                        and data[col].equals(true_data[col])
                        and data[col].equals(true_mp_data[col]))
        else:
            col_same = (np.allclose(data[col], mp_data[col], equal_nan=True)
                        and np.allclose(data[col], true_data[col], equal_nan=True)
                        and np.allclose(data[col], true_mp_data[col], equal_nan=True))
        if not col_same:
            bad_cols.append(col)
    if len(bad_cols) == 0:
        save_path = Path.home().joinpath('Downloads', 'test_power_calc_and_mp.hdf')
        save_path.unlink(missing_ok=True)
        data.to_hdf(save_path, key='data', mode='w')
        mp_data.to_hdf(save_path, key='mp_data', mode='a')
        true_data.to_hdf(save_path, key='true_data', mode='a')
        true_mp_data.to_hdf(save_path, key='true_mp_data', mode='a')
        assert False, f'columns {bad_cols} do not match, data saved to {save_path}'



if __name__ == '__main__':
    plot_flag = False
    test_unitary_epfm_slope(plot=plot_flag)
    test_piston_flow(plot=plot_flag)
    test_unitary_epfm(plot=plot_flag)
    test_bepfm_slope(plot=plot_flag)
    test_bpefm(plot=plot_flag)
    print('passed all unique tests, now for longer tests')
    make_test_power_calc_runs(plot_flag)
    test_power_calc_and_mp()
    print('passed all tests')
