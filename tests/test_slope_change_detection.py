"""
created matt_dumont 
on: 17/07/23
"""
import itertools
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from komanawa.gw_detect_power import DetectionPowerCalculator, AutoDetectionPowerSlope, DetectionPowerSlope


def print_myself():
    import traceback
    print(traceback.extract_stack(None, 2)[0][2])


def test_unitary_epfm(plot=False):
    print_myself()

    example = AutoDetectionPowerSlope(efficent_mode=False)
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
        plt.close('all')
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
    print_myself()
    example = AutoDetectionPowerSlope(efficent_mode=False)
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
        plt.close('all')
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
    print_myself()
    example = AutoDetectionPowerSlope(efficent_mode=False)
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
        plt.show()
        plt.close('all')


def test_bepfm_slope(plot=False):
    print_myself()
    example = AutoDetectionPowerSlope(efficent_mode=False)
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
        plt.close('all')
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
    print_myself()
    example = AutoDetectionPowerSlope(efficent_mode=False)
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
        plt.close('all')
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
    plt.close('all')


def make_power_calc_kwargs(error_val, samp_years=20):
    out = dict(idv='true',
               error=error_val,
               mrt_model='binary_exponential_piston_flow',
               samp_years=samp_years,
               samp_per_year=10,
               implementation_time=5,
               initial_conc=10,
               target_conc=5,
               prev_slope=1,
               max_conc_lim=25,
               min_conc_lim=1,
               mrt=5,
               #
               mrt_p1=3,
               frac_p1=0.7,
               f_p1=0.7,
               f_p2=0.7,
               #
               seed=558)
    return out


def test_return_true_noisy_conc(show=False):
    print_myself()
    write_test_data = False
    save_path = Path(__file__).parent.joinpath('test_data', 'test_return_true_noisy_conc.hdf')
    both_dp = AutoDetectionPowerSlope(significance_mode='linear-regression', return_true_conc=True,
                                      return_noisy_conc_itters=3, efficent_mode=False)
    true_dp = AutoDetectionPowerSlope(significance_mode='linear-regression', return_true_conc=True,
                                      return_noisy_conc_itters=0, efficent_mode=False)
    noise_dp = AutoDetectionPowerSlope(significance_mode='linear-regression', return_true_conc=False,
                                       return_noisy_conc_itters=3, efficent_mode=False)
    all_out = {}
    fig_true, axs_true = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(10, 10))
    fig_noise, axs_noise = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(10, 10))
    for i, dp_name in enumerate(['both_dp', 'true_dp', 'noise_dp']):
        dp = eval(dp_name)
        error_val = 1.5
        output = dp.power_calc(**make_power_calc_kwargs(error_val))
        all_out[dp_name] = output
        if 'noisy_conc' in output:
            temp = output['noisy_conc']
            for n, c in enumerate(['r', 'b', 'orange']):
                axs_noise[i].scatter(temp.index, temp.iloc[:, i], marker='o', label=f'noisy_conc_{n}', color=c)
                axs_noise[i].set_title(f'Noisy Conc {dp_name}')
                axs_noise[i].legend()
        if 'true_conc' in output:
            temp = output['true_conc']
            axs_true[i].plot(temp.index, temp, marker='o', label='true_conc')
            axs_true[i].set_title(f'True Conc {dp_name}')
            axs_true[i].legend()
    fig_true.tight_layout()
    fig_noise.tight_layout()
    if show:
        plt.show()
    plt.close('all')

    # save and check_data
    for k in ['true_dp', 'noise_dp']:
        temp = all_out[k]
        pd.testing.assert_series_equal(temp['power'], all_out['both_dp']['power'])
        if 'noisy_conc' in temp:
            pd.testing.assert_frame_equal(temp['noisy_conc'], all_out['both_dp']['noisy_conc'])
        if 'true_conc' in temp:
            pd.testing.assert_frame_equal(temp['true_conc'], all_out['both_dp']['true_conc'])
    if write_test_data:
        save_path.unlink(missing_ok=True)
        for k, v in all_out['both_dp'].items():
            v.to_hdf(save_path, k)
    else:
        for k, v in all_out['both_dp'].items():
            if isinstance(v, pd.DataFrame):
                true = pd.read_hdf(save_path, k)
                assert isinstance(true, pd.DataFrame)
                pd.testing.assert_frame_equal(v, true)
            elif isinstance(v, pd.Series):
                true = pd.read_hdf(save_path, k)
                assert isinstance(true, pd.Series)
                true = true.rename({'previous_slope': 'prev_slope'})
                pd.testing.assert_series_equal(v.sort_index(), true.sort_index())
    plt.close('all')


def test_linear_from_max_vs_from_start(show=False):
    print_myself()
    save_path = Path(__file__).parent.joinpath('test_data', 'test_linear_from_max_vs_from_start.hdf')
    write_test_data = False
    from kendall_stats import make_example_data
    # increasing
    x_inc, y_inc = make_example_data.make_multipart_sharp_change_data(make_example_data.multipart_sharp_slopes[0],
                                                                      noise=0,
                                                                      na_data=False, unsort=False)

    x_dec, y_dec = make_example_data.make_multipart_sharp_change_data(make_example_data.multipart_sharp_slopes[1],
                                                                      noise=0,
                                                                      na_data=False, unsort=False)
    idx = np.arange(0, len(x_inc), 5)
    x_inc = x_inc[idx]
    y_inc = y_inc[idx]
    x_dec = x_dec[idx]
    y_dec = y_dec[idx]

    fig, ax = plt.subplots()
    ax.plot(x_inc, y_inc, marker='o', label='increasing')
    ax.plot(x_dec, y_dec, marker='o', label='decreasing')
    ax.set_title('True Conc for increasing and decreasing slopes test')
    ax.legend()

    norm_dp = DetectionPowerSlope(significance_mode='linear-regression', return_true_conc=True,
                                  return_noisy_conc_itters=3, efficent_mode=False)
    max_dp = DetectionPowerSlope(significance_mode='linear-regression-from-max', return_true_conc=True,
                                 return_noisy_conc_itters=3, efficent_mode=False)
    min_dp = DetectionPowerSlope(significance_mode='linear-regression-from-min', return_true_conc=True,
                                 return_noisy_conc_itters=3, efficent_mode=False)
    error_val = 0.5
    norm_inc_power = norm_dp.power_calc(idv='norm_inc', error=error_val, true_conc_ts=y_inc, mrt_model='pass_true_conc')
    norm_dec_power = norm_dp.power_calc(idv='norm_dec', error=error_val, true_conc_ts=y_dec, mrt_model='pass_true_conc')
    max_inc_power = max_dp.power_calc(idv='max_inc', error=error_val, true_conc_ts=y_inc, mrt_model='pass_true_conc')
    min_dec_power = min_dp.power_calc(idv='min_dec', error=error_val, true_conc_ts=y_dec, mrt_model='pass_true_conc')

    assert norm_inc_power['power']['power'] == 0
    assert norm_dec_power['power']['power'] == 0
    assert max_inc_power['power']['power'] == 100
    assert min_dec_power['power']['power'] == 100
    pass

    # test with piston flow and binary exponential piston flow lags...
    norm_dp_auto = AutoDetectionPowerSlope(significance_mode='linear-regression', return_true_conc=True,
                                           return_noisy_conc_itters=3, efficent_mode=False)
    max_dp_auto = AutoDetectionPowerSlope(significance_mode='linear-regression-from-max', return_true_conc=True,
                                          return_noisy_conc_itters=3, efficent_mode=False)
    all_outdata = {}
    for error in [0, 2, 5]:
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10, 10))
        for i, dp_name in enumerate(['norm_dp_auto', 'max_dp_auto']):
            use_dp = eval(dp_name)
            temp_out = use_dp.power_calc(**make_power_calc_kwargs(error, samp_years=5))
            all_outdata[f'{dp_name.replace("_auto", "")}_{error}'] = temp_out
            axs[i, 0].plot(temp_out['true_conc'].index, temp_out['true_conc']['true_conc'], marker='o',
                           label='true_conc')
            for n, c in enumerate(['r', 'b', 'orange']):
                t = temp_out['noisy_conc']
                axs[i, 1].scatter(t.index, t.iloc[:, n], marker='o', label=f'noisy_conc_{n}', color=c)

            axs[i, 0].set_title(f'True Conc {dp_name}, power:{temp_out["power"]["power"]}')
            axs[i, 0].legend()
            axs[i, 1].set_title(f'Noisy Conc {dp_name}, power:{temp_out["power"]["power"]}')
            axs[i, 1].legend()
        fig.suptitle(f'Error: {error}')
        fig.tight_layout()
    if show:
        plt.show()
    plt.close('all')

    if write_test_data:
        save_path.unlink(missing_ok=True)
        for k, v in all_outdata.items():
            for k2, v2 in v.items():
                if k2 == 'significant':
                    continue
                use_k = f'{k}_{k2}'
                v2.to_hdf(save_path, use_k)

    # test data
    for k, v in all_outdata.items():
        for k2, v2 in v.items():
            if k2 == 'significant':
                continue
            use_k = f'{k}_{k2}'
            true_data = pd.read_hdf(save_path, use_k)
            if isinstance(v2, pd.DataFrame):
                assert isinstance(true_data, pd.DataFrame)
                pd.testing.assert_frame_equal(v2, true_data)
            elif isinstance(v2, pd.Series):
                assert isinstance(true_data, pd.Series)
                true_data = true_data.rename({'previous_slope': 'prev_slope'})
                pd.testing.assert_series_equal(v2.sort_index(), true_data.sort_index())
            else:
                raise ValueError(f'Unknown type: {type(v2)}')
    plt.close('all')


def test_mann_kendall_power(show=False):
    print_myself()
    save_path = Path(__file__).parent.joinpath('test_data', 'test_kendall_from_max_vs_from_start.hdf')
    write_test_data = False
    from kendall_stats import make_example_data, MannKendall
    # increasing
    x_inc, y_inc = make_example_data.make_multipart_sharp_change_data(make_example_data.multipart_sharp_slopes[0],
                                                                      noise=0,
                                                                      na_data=False, unsort=False)

    x_dec, y_dec = make_example_data.make_multipart_sharp_change_data(make_example_data.multipart_sharp_slopes[1],
                                                                      noise=0,
                                                                      na_data=False, unsort=False)
    idx = np.arange(0, len(x_inc), 5)
    x_inc = x_inc[idx]
    y_inc = y_inc[idx]
    x_dec = x_dec[idx]
    y_dec = y_dec[idx]

    fig, ax = plt.subplots()
    ax.plot(x_inc, y_inc, marker='o', label='increasing')
    ax.plot(x_dec, y_dec, marker='o', label='decreasing')
    ax.set_title('True Conc for increasing and decreasing slopes test')
    ax.legend()

    norm_dp = DetectionPowerSlope(significance_mode='mann-kendall', return_true_conc=True,
                                  return_noisy_conc_itters=3, efficent_mode=False)
    max_dp = DetectionPowerSlope(significance_mode='mann-kendall-from-max', return_true_conc=True,
                                 return_noisy_conc_itters=3, efficent_mode=False)
    min_dp = DetectionPowerSlope(significance_mode='mann-kendall-from-min', return_true_conc=True,
                                 return_noisy_conc_itters=3, efficent_mode=False)
    error_val = 0.5
    norm_inc_power = norm_dp.power_calc(idv='norm_inc', error=error_val, true_conc_ts=y_inc, mrt_model='pass_true_conc')
    norm_dec_power = norm_dp.power_calc(idv='norm_dec', error=error_val, true_conc_ts=y_dec, mrt_model='pass_true_conc')
    max_inc_power = max_dp.power_calc(idv='max_inc', error=error_val, true_conc_ts=y_inc, mrt_model='pass_true_conc')
    min_dec_power = min_dp.power_calc(idv='min_dec', error=error_val, true_conc_ts=y_dec, mrt_model='pass_true_conc')

    assert norm_inc_power['power']['power'] == 0
    assert norm_dec_power['power']['power'] == 0
    assert max_inc_power['power']['power'] == 100
    assert min_dec_power['power']['power'] == 100
    pass

    # test with piston flow and binary exponential piston flow lags...
    norm_dp_auto = AutoDetectionPowerSlope(significance_mode='mann-kendall', return_true_conc=True,
                                           return_noisy_conc_itters=3, efficent_mode=False)
    max_dp_auto = AutoDetectionPowerSlope(significance_mode='mann-kendall-from-max', return_true_conc=True,
                                          return_noisy_conc_itters=3, efficent_mode=False)
    all_outdata = {}
    for error in [0, 2, 5]:
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10, 10))
        for i, dp_name in enumerate(['norm_dp_auto', 'max_dp_auto']):
            use_dp = eval(dp_name)
            temp_out = use_dp.power_calc(**make_power_calc_kwargs(error, samp_years=5))
            all_outdata[f'{dp_name.replace("_auto", "")}_{error}'] = temp_out
            axs[i, 0].plot(temp_out['true_conc'].index, temp_out['true_conc']['true_conc'], marker='o',
                           label='true_conc')
            for n, c in enumerate(['r', 'b', 'orange']):
                t = temp_out['noisy_conc']
                true_conc = temp_out['true_conc']['true_conc']
                if 'max' in dp_name:
                    mk = MannKendall(data=t.iloc[np.argmax(true_conc):, n], alpha=use_dp.min_p_value)
                else:
                    mk = MannKendall(data=t.iloc[:, n], alpha=use_dp.min_p_value)
                fig, ax0 = mk.plot_data()
                ax0.set_title(f'Noisy Conc {dp_name} {n}, power:{temp_out["power"]["power"]}')
                axs[i, 1].scatter(t.index, t.iloc[:, n], marker='o', label=f'noisy_conc_{n}', color=c)

            axs[i, 0].set_title(f'True Conc {dp_name}, power:{temp_out["power"]["power"]}')
            axs[i, 0].legend()
            axs[i, 1].set_title(f'Noisy Conc {dp_name}, power:{temp_out["power"]["power"]}')
            axs[i, 1].legend()
        fig.suptitle(f'Error: {error}')
        fig.tight_layout()
    if show:
        plt.show()
    plt.close('all')

    if write_test_data:
        save_path.unlink(missing_ok=True)
        for k, v in all_outdata.items():
            for k2, v2 in v.items():
                if k2 == 'significant':
                    continue
                use_k = f'{k}_{k2}'
                v2.to_hdf(save_path, use_k)

    # test data
    for k, v in all_outdata.items():
        for k2, v2 in v.items():
            if k2 == 'significant':
                continue
            use_k = f'{k}_{k2}'
            true_data = pd.read_hdf(save_path, use_k)
            if isinstance(v2, pd.DataFrame):
                assert isinstance(true_data, pd.DataFrame)
                pd.testing.assert_frame_equal(v2, true_data)
            elif isinstance(v2, pd.Series):
                assert isinstance(true_data, pd.Series)
                true_data = true_data.rename({'previous_slope': 'prev_slope'})
                pd.testing.assert_series_equal(v2.sort_index(), true_data.sort_index())
            else:
                raise ValueError(f'Unknown type: {type(v2)}')
    plt.close('all')


def test_multpart_mann_kendall_power(show=False):
    print_myself()
    save_path = Path(__file__).parent.joinpath('test_data', 'test_mp_kendall_from_max_vs_from_start.hdf')
    write_test_data = False
    from kendall_stats import MultiPartKendall

    dp_3part = DetectionPowerSlope(
        significance_mode='n-section-mann-kendall',
        expect_slope=[1, 0, -1], nparts=3, min_part_size=10, no_trend_alpha=0.50,
        return_true_conc=True, return_noisy_conc_itters=3, efficent_mode=False)
    dp_2part = DetectionPowerSlope(
        significance_mode='n-section-mann-kendall',
        expect_slope=[1, -1], nparts=2, min_part_size=10, no_trend_alpha=0.50,
        return_true_conc=True, return_noisy_conc_itters=3, efficent_mode=False)

    # test with piston flow and binary exponential piston flow lags...
    dp_3part = AutoDetectionPowerSlope(
        significance_mode='n-section-mann-kendall',
        expect_slope=[1, 0, -1], nparts=3, min_part_size=10, no_trend_alpha=0.50,
        return_true_conc=True, return_noisy_conc_itters=3, efficent_mode=False)
    dp_2part = AutoDetectionPowerSlope(
        significance_mode='n-section-mann-kendall',
        expect_slope=[1, -1], nparts=2, min_part_size=10, no_trend_alpha=0.50,
        return_true_conc=True, return_noisy_conc_itters=3, efficent_mode=False)
    all_outdata = {}
    for error in [0, 2, 5]:
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10, 10))
        for i, dp_name in enumerate(['dp_3part', 'dp_2part']):
            use_dp = eval(dp_name)
            temp_out = use_dp.power_calc(**make_power_calc_kwargs(error, samp_years=5))
            all_outdata[f'{dp_name}_{error}'] = temp_out
            axs[i, 0].plot(temp_out['true_conc'].index, temp_out['true_conc']['true_conc'], marker='o',
                           label='true_conc')
            for n, c in enumerate(['r', 'b', 'orange']):
                t = temp_out['noisy_conc']
                mpmk = MultiPartKendall(data=t.iloc[:, n], nparts=use_dp.kendall_mp_nparts,
                                        expect_part=use_dp.expect_slope, min_size=use_dp.kendall_mp_min_part_size,
                                        alpha=use_dp.min_p_value, no_trend_alpha=use_dp.kendall_mp_no_trend_alpha)
                bp = mpmk.get_maxz_breakpoints()
                if bp is not None:
                    fig, ax0 = mpmk.plot_data_from_breakpoints(bp[0])
                ax0.set_title(f'Noisy Conc {dp_name} {n}, power:{temp_out["power"]["power"]}, error:{error}')
                axs[i, 1].scatter(t.index, t.iloc[:, n], marker='o', label=f'noisy_conc_{n}', color=c)

            axs[i, 0].set_title(f'True Conc {dp_name}, power:{temp_out["power"]["power"]}')
            axs[i, 0].legend()
            axs[i, 1].set_title(f'Noisy Conc {dp_name}, power:{temp_out["power"]["power"]}')
            axs[i, 1].legend()
        fig.suptitle(f'Error: {error}')
        fig.tight_layout()
    if show:
        plt.show()
    plt.close('all')

    if write_test_data:
        save_path.unlink(missing_ok=True)
        for k, v in all_outdata.items():
            for k2, v2 in v.items():
                if k2 == 'significant':
                    continue
                use_k = f'{k}_{k2}'
                v2.to_hdf(save_path, use_k)

    # test data
    for k, v in all_outdata.items():
        for k2, v2 in v.items():
            use_k = f'{k}_{k2}'
            if k2 == 'significant':
                continue
            true_data = pd.read_hdf(save_path, use_k)
            if isinstance(v2, pd.DataFrame):
                assert isinstance(true_data, pd.DataFrame)
                pd.testing.assert_frame_equal(v2, true_data)
            elif isinstance(v2, pd.Series):
                assert isinstance(true_data, pd.Series)
                true_data = true_data.rename({'previous_slope': 'prev_slope'})
                pd.testing.assert_series_equal(v2.sort_index(), true_data.sort_index())
            else:
                raise ValueError(f'Unknown type: {type(v2)}')
    plt.close('all')


def test_pettitt_power(show=False):
    print_myself()
    from pyhomogeneity import pettitt_test
    from kendall_stats import make_example_data
    power_data = []
    pd = DetectionPowerSlope(significance_mode='pettitt-test', nsims_pettit=1000, nsims=1000, efficent_mode=False)
    noises = [0, 0.25, 1, 2]
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(10, 10))
    for noise, ax in zip(noises, axs[:, 0]):
        print(f'pettitt Noise: {noise}, v change')
        # increasing
        x_inc, y_inc = make_example_data.make_multipart_sharp_change_data(make_example_data.multipart_sharp_slopes[0],
                                                                          noise=noise,
                                                                          na_data=False, unsort=False)

        idx = np.arange(0, len(x_inc), 3)
        x_inc = x_inc[idx]
        y_inc = y_inc[idx]

        y0_org = deepcopy(y_inc)
        h, cp, p, U, mu = pettitt_test(y_inc, alpha=0.05,
                                       sim=1000)
        output = pd.power_calc(idv='pettitt', error=noise, true_conc_ts=y_inc, mrt_model='pass_true_conc')
        power = output['power']
        power_data.append(power)
        assert np.allclose(y_inc, y0_org)
        ax.scatter(x_inc, y_inc, marker='o', label='raw_data')
        if len(np.atleast_1d(cp)) > 1:
            raise ValueError('More than one change point found')
        ax.axvline(x=x_inc[cp], color='r', label='change_point')
        ax.set_title(f'Pettitt Test, noise: {noise}, pval: {round(p, 2)}\n{power=}')
        ax.legend()

    true_conc = np.zeros((25)) + 10
    true_conc[12:] = 8
    for noise, ax in zip(noises, axs[:, 1]):
        print(f'pettitt Noise: {noise}, instaneous change')
        y0 = true_conc + np.random.normal(0, noise, size=true_conc.shape)
        h, cp, p, U, mu = pettitt_test(y0, alpha=0.05,
                                       sim=1000)
        output = pd.power_calc(idv='pettitt', error=noise, true_conc_ts=true_conc, mrt_model='pass_true_conc',
                               seed=5585)
        power = output['power']
        power_data.append(power)
        ax.scatter(np.arange(len(y0)), y0, marker='o', label='raw_data')
        if len(np.atleast_1d(cp)) > 1:
            raise ValueError('More than one change point found')
        ax.axvline(x=cp, color='r', label='change_point')
        ax.set_title(f'Pettitt Test, noise: {noise}, pval: {round(p, 2)}\n{power=}')
        ax.legend()
    fig.tight_layout()

    expect = [100.0, 89.2, 38.3, 15.2, 100.0, 100.0, 98.9, 62.1]
    assert np.allclose(
        np.array(power_data),
        expect,
        atol=0.1
    ), (f'bad values for pettitt test got: {power_data} should be {expect}, note if they are similar the problem may be'
        f'the inability to pass a seed to the pettitt test')
    if show:
        plt.show()
    plt.close('all')


def test_iteration_plotting(show=False):
    print_myself()
    from kendall_stats import make_example_data
    x_inc, y_inc = make_example_data.make_multipart_sharp_change_data(make_example_data.multipart_sharp_slopes[0],
                                                                      noise=0,
                                                                      na_data=False, unsort=False)

    x_dec, y_dec = make_example_data.make_multipart_sharp_change_data(make_example_data.multipart_sharp_slopes[1],
                                                                      noise=0,
                                                                      na_data=False, unsort=False)
    idx = np.arange(0, len(x_inc), 2)
    y_inc = y_inc[idx]
    y_dec = y_dec[idx]
    linear = y_dec[len(y_dec) // 2:]
    pettitt = np.zeros((25)) + 10
    pettitt[12:] = 8

    implemented_significance_modes_data = {
        'linear-regression': linear,
        'linear-regression-from-max': y_inc,
        'linear-regression-from-min': y_dec,
        'mann-kendall': linear,
        'mann-kendall-from-max': y_inc,
        'mann-kendall-from-min': y_dec,
        'n-section-mann-kendall': y_inc,
        'pettitt-test': pettitt,
    }

    for mode, data in implemented_significance_modes_data.items():

        if mode == 'n-section-mann-kendall':
            expect_slope = [1, -1]
        else:
            expect_slope = 'auto'
        dp = DetectionPowerSlope(significance_mode=mode, expect_slope=expect_slope, nsims=100,
                                 nparts=2, return_true_conc=True,
                                 return_noisy_conc_itters=1, efficent_mode=False)
        output = dp.power_calc(idv=mode, error=0.5, true_conc_ts=data, mrt_model='pass_true_conc')
        fig, ax = dp.plot_iteration(output['noisy_conc'].iloc[:, 0], output['true_conc'])
        ax.set_title(f'{mode} power: {output["power"]["power"]}')
        fig.tight_layout()
    if show:
        plt.show()
    plt.close('all')


def make_test_power_calc_runs(plot=False):
    print_myself()
    auto_runs = []
    pass_ts_runs = []
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
            max_conc_lim=mc,
            min_conc_lim=mic,
            prev_slope=ps,
            mrt_model=mrtm,
            mrt=mrt,
            mrt_p1=mrt_p1,
            frac_p1=frac_p1,
            f_p1=f_p1,
            f_p2=f_p2,
            seed=seed_val,
        )
        auto_runs.append(temp)

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
        fig, ax = plt.subplots()
        ax.plot(data.index, data, marker='o', c='r', label='data')
        ax.plot(data2.index, data2, marker='o', c='b', label='data2')
        ax.legend()
        ax.set_ylabel('Concentration')
        ax.set_xlabel('Years')
        plt.show()
        plt.close('all')
    true_tss = [data, data2]

    for i, (e, tts) in enumerate(itertools.product(errors, true_tss)):
        temp = dict(
            idv=f'tts_{i}',
            error=e,
            true_conc_ts=tts.copy(),
            mrt_model='pass_true_conc',
            seed=seed_val,
        )
        pass_ts_runs.append(temp)

    print(f'auto runs: {len(auto_runs)}, pass ts runs: {len(pass_ts_runs)}')
    plt.close('all')

    return auto_runs, pass_ts_runs


def _mutli_process_runs(ex_auto, ex, auto_runs, ts_runs):
    t = time.time()
    mp_data_auto = ex_auto.mulitprocess_power_calcs(
        outpath=None,
        idv_vals=np.array([r.get('idv') for r in auto_runs]),
        error_vals=np.array([r.get('error') for r in auto_runs]),
        samp_years_vals=np.array([r.get('samp_years') for r in auto_runs]),
        samp_per_year_vals=np.array([r.get('samp_per_year') for r in auto_runs]),
        implementation_time_vals=np.array([r.get('implementation_time') for r in auto_runs]),
        initial_conc_vals=np.array([r.get('initial_conc') for r in auto_runs]),
        target_conc_vals=np.array([r.get('target_conc') for r in auto_runs]),
        prev_slope_vals=np.array([r.get('prev_slope') for r in auto_runs]),
        max_conc_lim_vals=np.array([r.get('max_conc_lim') for r in auto_runs]),
        min_conc_lim_vals=np.array([r.get('min_conc_lim') for r in auto_runs]),
        mrt_model_vals=np.array([r.get('mrt_model') for r in auto_runs]),
        mrt_vals=np.array([r.get('mrt') for r in auto_runs]),
        mrt_p1_vals=np.array([r.get('mrt_p1') for r in auto_runs]),
        frac_p1_vals=np.array([r.get('frac_p1') for r in auto_runs]),
        f_p1_vals=np.array([r.get('f_p1') for r in auto_runs]),
        f_p2_vals=np.array([r.get('f_p2') for r in auto_runs]),
        seed_vals=np.array([r.get('seed') for r in auto_runs]),
    )
    print(f'elapsed time for mp (auto): {time.time() - t}')

    # run non auto

    t = time.time()
    mp_data = ex.mulitprocess_power_calcs(
        outpath=None,
        idv_vals=np.array([r.get('idv') for r in ts_runs]),
        error_vals=np.array([r.get('error') for r in ts_runs]),
        samp_years_vals=np.array([r.get('samp_years') for r in ts_runs]),
        samp_per_year_vals=np.array([r.get('samp_per_year') for r in ts_runs]),
        implementation_time_vals=np.array([r.get('implementation_time') for r in ts_runs]),
        initial_conc_vals=np.array([r.get('initial_conc') for r in ts_runs]),
        target_conc_vals=np.array([r.get('target_conc') for r in ts_runs]),
        prev_slope_vals=np.array([r.get('prev_slope') for r in ts_runs]),
        max_conc_lim_vals=np.array([r.get('max_conc_lim') for r in ts_runs]),
        min_conc_lim_vals=np.array([r.get('min_conc_lim') for r in ts_runs]),
        mrt_model_vals=np.array([r.get('mrt_model') for r in ts_runs]),
        mrt_vals=np.array([r.get('mrt') for r in ts_runs]),
        mrt_p1_vals=np.array([r.get('mrt_p1') for r in ts_runs]),
        frac_p1_vals=np.array([r.get('frac_p1') for r in ts_runs]),
        f_p1_vals=np.array([r.get('f_p1') for r in ts_runs]),
        f_p2_vals=np.array([r.get('f_p2') for r in ts_runs]),
        true_conc_ts_vals=[r.get('true_conc_ts') for r in ts_runs],
        seed_vals=np.array([r.get('seed') for r in ts_runs]),
    )
    print(f'elapsed time for mp (non-auto): {time.time() - t}')
    out = pd.concat([mp_data_auto, mp_data])
    assert 'min_conc' not in out.columns
    assert 'mrt_p2' in out.columns
    assert 'true_conc_ts_none' not in out.columns

    return out


def _single_process_runs(ex_auto, ex, auto_runs, ts_runs):
    print('running non-mp this takes c. 8-10 mins')
    data = []
    t = time.time()
    for i, run in enumerate(auto_runs):
        if i % 50 == 0:
            print(f'starting run {i + 1} of {len(auto_runs)}')
        out = ex_auto.power_calc(**run)
        data.append(out)

    # non - auto
    for i, run in enumerate(ts_runs):
        if i % 50 == 0:
            print(f'starting run {i + 1} of {len(ts_runs)}')
        out = ex.power_calc(**run)
        data.append(out)

    data = pd.DataFrame(data)
    data.set_index('idv', inplace=True)
    print(f'elapsed time for non-mp: {time.time() - t}')
    return data


def test_power_calc_and_mp():
    print_myself()
    save_path = Path(__file__).parent.joinpath('test_data', 'test_power_calc_and_mp.hdf')
    write_test_data = False
    auto_runs, ts_runs = make_test_power_calc_runs()

    ex_auto = AutoDetectionPowerSlope(efficent_mode=False)
    ex = DetectionPowerSlope(efficent_mode=False)

    data = _single_process_runs(ex_auto, ex, auto_runs, ts_runs)
    mp_data = _mutli_process_runs(ex_auto, ex, auto_runs, ts_runs)

    if write_test_data:
        save_path.unlink(missing_ok=True)
        data.to_hdf(save_path, key='data', mode='w')
        mp_data.to_hdf(save_path, key='mp_data', mode='a')
    true_data = pd.read_hdf(save_path, key='data')
    true_mp_data = pd.read_hdf(save_path, key='mp_data')
    true_mp_data = true_mp_data.rename({'previous_slope': 'prev_slope'}, axis=1)
    true_data = true_data.rename({'previous_slope': 'prev_slope'}, axis=1)
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
            temp_data = data[col].astype(float)
            temp_mp_data = mp_data[col].astype(float)
            temp_true_data = true_data[col].astype(float)
            temp_true_mp_data = true_mp_data[col].astype(float)
            col_same = (np.allclose(temp_data, temp_mp_data, equal_nan=True)
                        and np.allclose(temp_data, temp_true_data, equal_nan=True)
                        and np.allclose(temp_data, temp_true_mp_data, equal_nan=True))
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


def test_efficient_mode_lr():
    print_myself()
    from kendall_stats import make_example_data
    # increasing
    x, y = make_example_data.make_increasing_decreasing_data(slope=0.1, noise=0)

    x_inc, y_inc = make_example_data.make_multipart_sharp_change_data(make_example_data.multipart_sharp_slopes[0],
                                                                      noise=0,
                                                                      na_data=False, unsort=False)

    x_dec, y_dec = make_example_data.make_multipart_sharp_change_data(make_example_data.multipart_sharp_slopes[1],
                                                                      noise=0,
                                                                      na_data=False, unsort=False)
    idx = np.arange(0, len(x_inc), 5)
    y_inc = y_inc[idx]
    y_dec = y_dec[idx]
    y = y[idx]

    # test normal
    norm_dp = DetectionPowerSlope(significance_mode='linear-regression', return_true_conc=False,
                                  return_noisy_conc_itters=0, efficent_mode=False)
    norm_dpeff = DetectionPowerSlope(significance_mode='linear-regression', return_true_conc=False,
                                     return_noisy_conc_itters=0, efficent_mode=True)
    for error_val in [0.5, 1, 5, 10, 30, 35, 40, 50]:
        eff = norm_dpeff.power_calc(idv='norm_inc', error=error_val, true_conc_ts=y, mrt_model='pass_true_conc',
                                    seed=5585)
        non_eff = norm_dp.power_calc(idv='norm_inc', error=error_val, true_conc_ts=y, mrt_model='pass_true_conc',
                                     seed=5585)
        eff = pd.Series(eff)
        non_eff = pd.Series(non_eff)
        pd.testing.assert_series_equal(eff, non_eff), f'error: {error_val}'

        eff = norm_dpeff.power_calc(idv='norm_inc', error=error_val, true_conc_ts=np.zeros_like(y),
                                    mrt_model='pass_true_conc', seed=5585)
        non_eff = norm_dp.power_calc(idv='norm_inc', error=error_val, true_conc_ts=np.zeros_like(y),
                                     mrt_model='pass_true_conc', seed=5585)
        eff = pd.Series(eff)
        non_eff = pd.Series(non_eff)
        pd.testing.assert_series_equal(eff, non_eff), f'error: {error_val}'

    # test from max
    max_dp = DetectionPowerSlope(significance_mode='linear-regression-from-max', return_true_conc=False,
                                 return_noisy_conc_itters=0, efficent_mode=False)
    max_dp_eff = DetectionPowerSlope(significance_mode='linear-regression-from-max', return_true_conc=False,
                                     return_noisy_conc_itters=0, efficent_mode=True)

    for error_val in [0.5, 1, 5, 10, 30]:
        eff = max_dp_eff.power_calc(idv='norm_inc', error=error_val, true_conc_ts=y_inc, mrt_model='pass_true_conc',
                                    seed=5585)
        non_eff = max_dp.power_calc(idv='norm_inc', error=error_val, true_conc_ts=y_inc, mrt_model='pass_true_conc',
                                    seed=5585)
        eff = pd.Series(eff)
        non_eff = pd.Series(non_eff)
        pd.testing.assert_series_equal(eff, non_eff), f'error: {error_val}'

    # test from min
    min_dp = DetectionPowerSlope(significance_mode='linear-regression-from-min', return_true_conc=False,
                                 return_noisy_conc_itters=0, efficent_mode=False)
    min_dp_eff = DetectionPowerSlope(significance_mode='linear-regression-from-min', return_true_conc=False,
                                     return_noisy_conc_itters=0, efficent_mode=True)
    for error_val in [0.5, 1, 5, 10, 30]:
        eff = min_dp_eff.power_calc(idv='norm_inc', error=error_val, true_conc_ts=y_dec, mrt_model='pass_true_conc',
                                    seed=5585)
        non_eff = min_dp.power_calc(idv='norm_inc', error=error_val, true_conc_ts=y_dec, mrt_model='pass_true_conc',
                                    seed=5585)
        eff = pd.Series(eff)
        non_eff = pd.Series(non_eff)
        pd.testing.assert_series_equal(eff, non_eff), f'error: {error_val}'


def test_efficent_mode_mann_kendall():
    print_myself()
    from kendall_stats import make_example_data
    # increasing
    x, y = make_example_data.make_increasing_decreasing_data(slope=0.1, noise=0)

    x_inc, y_inc = make_example_data.make_multipart_sharp_change_data(make_example_data.multipart_sharp_slopes[0],
                                                                      noise=0,
                                                                      na_data=False, unsort=False)

    x_dec, y_dec = make_example_data.make_multipart_sharp_change_data(make_example_data.multipart_sharp_slopes[1],
                                                                      noise=0,
                                                                      na_data=False, unsort=False)
    idx = np.arange(0, len(x_inc), 5)
    y_inc = y_inc[idx]
    y_dec = y_dec[idx]
    y = y[idx]

    # test normal
    norm_dp = DetectionPowerSlope(significance_mode='mann-kendall', return_true_conc=False,
                                  return_noisy_conc_itters=0, efficent_mode=False, print_freq=100)
    norm_dpeff = DetectionPowerSlope(significance_mode='mann-kendall', return_true_conc=False,
                                     return_noisy_conc_itters=0, efficent_mode=True, print_freq=100)
    for error_val in [0.5, 1, 5, 10, 30, 35, 40, 50]:
        eff = norm_dpeff.power_calc(idv='norm_inc', error=error_val, true_conc_ts=y, mrt_model='pass_true_conc',
                                    seed=5585)
        non_eff = norm_dp.power_calc(idv='norm_inc', error=error_val, true_conc_ts=y, mrt_model='pass_true_conc',
                                     seed=5585)
        eff = pd.Series(eff)
        non_eff = pd.Series(non_eff)
        pd.testing.assert_series_equal(eff, non_eff), f'error: {error_val}'

        eff = norm_dpeff.power_calc(idv='norm_inc', error=error_val, true_conc_ts=np.zeros_like(y),
                                    mrt_model='pass_true_conc', seed=5585)
        non_eff = norm_dp.power_calc(idv='norm_inc', error=error_val, true_conc_ts=np.zeros_like(y),
                                     mrt_model='pass_true_conc', seed=5585)
        eff = pd.Series(eff)
        non_eff = pd.Series(non_eff)
        pd.testing.assert_series_equal(eff, non_eff), f'error: {error_val}'

    # test from max
    max_dp = DetectionPowerSlope(significance_mode='mann-kendall-from-max', return_true_conc=False,
                                 return_noisy_conc_itters=0, efficent_mode=False)
    max_dp_eff = DetectionPowerSlope(significance_mode='mann-kendall-from-max', return_true_conc=False,
                                     return_noisy_conc_itters=0, efficent_mode=True)

    for error_val in [0.5, 1, 5, 10, 30]:
        eff = max_dp_eff.power_calc(idv='norm_inc', error=error_val, true_conc_ts=y_inc, mrt_model='pass_true_conc',
                                    seed=5585)
        non_eff = max_dp.power_calc(idv='norm_inc', error=error_val, true_conc_ts=y_inc, mrt_model='pass_true_conc',
                                    seed=5585)
        eff = pd.Series(eff)
        non_eff = pd.Series(non_eff)
        pd.testing.assert_series_equal(eff, non_eff), f'error: {error_val}'

    # test from min
    min_dp = DetectionPowerSlope(significance_mode='mann-kendall-from-min', return_true_conc=False,
                                 return_noisy_conc_itters=0, efficent_mode=False)
    min_dp_eff = DetectionPowerSlope(significance_mode='mann-kendall-from-min', return_true_conc=False,
                                     return_noisy_conc_itters=0, efficent_mode=True)
    for error_val in [0.5, 1, 5, 10, 30]:
        eff = min_dp_eff.power_calc(idv='norm_inc', error=error_val, true_conc_ts=y_dec, mrt_model='pass_true_conc',
                                    seed=5585)
        non_eff = min_dp.power_calc(idv='norm_inc', error=error_val, true_conc_ts=y_dec, mrt_model='pass_true_conc',
                                    seed=5585)
        eff = pd.Series(eff)
        non_eff = pd.Series(non_eff)
        pd.testing.assert_series_equal(eff, non_eff), f'error: {error_val}'


def test_efficient_mode_mpmk():
    print_myself()
    from kendall_stats import make_example_data

    dp_2part = DetectionPowerSlope(
        significance_mode='n-section-mann-kendall', nsims=100,
        expect_slope=[1, -1], nparts=2, min_part_size=10, no_trend_alpha=0.50,
        return_true_conc=False, return_noisy_conc_itters=0, efficent_mode=False,
        mpmk_check_step=2, mpmk_efficent_min=10, mpmk_window=0.05, )
    dp_2part_eff = DetectionPowerSlope(
        significance_mode='n-section-mann-kendall', nsims=100,
        expect_slope=[1, -1], nparts=2, min_part_size=10, no_trend_alpha=0.50,
        return_true_conc=False, return_noisy_conc_itters=0, efficent_mode=True,
        mpmk_check_step=2, mpmk_efficent_min=80, mpmk_window=0.1, )
    dp_2part_eff_2 = DetectionPowerSlope(
        significance_mode='n-section-mann-kendall', nsims=100,
        expect_slope=[1, -1], nparts=2, min_part_size=10, no_trend_alpha=0.50,
        return_true_conc=False, return_noisy_conc_itters=0, efficent_mode=True,
        mpmk_check_step=2, mpmk_efficent_min=5, mpmk_window=0.1, )

    # parabolic
    dp_3part = DetectionPowerSlope(
        significance_mode='n-section-mann-kendall', nsims=100,
        expect_slope=[1, 0, -1], nparts=3, min_part_size=10, no_trend_alpha=0.50,
        return_true_conc=False, return_noisy_conc_itters=0, efficent_mode=False,
        mpmk_check_step=2, mpmk_efficent_min=2, mpmk_window=0.05)
    dp_3part_eff = DetectionPowerSlope(
        significance_mode='n-section-mann-kendall', nsims=100,
        expect_slope=[1, 0, -1], nparts=3, min_part_size=10, no_trend_alpha=0.50,
        return_true_conc=False, return_noisy_conc_itters=0, efficent_mode=True,
        mpmk_check_step=2, mpmk_efficent_min=2, mpmk_window=0.05)

    x, y_para = make_example_data.make_multipart_parabolic_data(make_example_data.multipart_parabolic_slopes[0],
                                                                noise=0,
                                                                na_data=False, unsort=False, step=2)
    noises = [1, 7.5, 10, 50]
    diffs = [0.1, 3.5, 16, 5]
    for noise, allow_dif in zip(noises, diffs):
        print(f'testing efficiency on parabolic data {noise=}')
        out = dp_3part.power_calc(idv='sharp', error=noise, true_conc_ts=y_para, mrt_model='pass_true_conc', seed=987)
        out_eff = dp_3part_eff.power_calc(idv='sharp', error=noise, true_conc_ts=y_para, mrt_model='pass_true_conc',
                                          seed=987)
        out = pd.Series(out)
        out_eff = pd.Series(out_eff)
        print(f'{out["power"]=} {out_eff["power"]=}')
        assert np.isclose(out_eff['power'], out['power'], rtol=allow_dif)

    # test sharp
    x_inc, y_inc = make_example_data.make_multipart_sharp_change_data(make_example_data.multipart_sharp_slopes[0],
                                                                      noise=0,
                                                                      na_data=False, unsort=False)
    noises = [1, 2.5, 5]
    diffs = [0.1, 0.5, 6]
    for noise, allow_dif in zip(noises, diffs):
        print(f'testing efficiency on sharp data {noise=}')
        out = dp_2part.power_calc(idv='sharp', error=noise, true_conc_ts=y_inc, mrt_model='pass_true_conc', seed=688765)
        out_eff = dp_2part_eff.power_calc(idv='sharp', error=noise, true_conc_ts=y_inc, mrt_model='pass_true_conc',
                                          seed=688765)
        out_eff_2 = dp_2part_eff_2.power_calc(idv='sharp', error=noise, true_conc_ts=y_inc, mrt_model='pass_true_conc',
                                              seed=688765)
        out = pd.Series(out)
        out_eff = pd.Series(out_eff)
        print(f"{out['power']=} {out_eff['power']=} {out_eff_2['power']=}")
        assert np.isclose(out_eff_2['power'], out_eff['power'], rtol=allow_dif)
        pd.testing.assert_series_equal(out, out_eff, obj=f'sharp noise: {noise}')

    # flat
    yflat = np.zeros_like(y_inc) + 10
    for noise in noises:
        print(f'testing efficiency on flat data {noise=}')
        out = dp_2part.power_calc(idv='sharp', error=noise, true_conc_ts=yflat, mrt_model='pass_true_conc', seed=5652)
        out_eff = dp_2part_eff.power_calc(idv='sharp', error=noise, true_conc_ts=yflat, mrt_model='pass_true_conc',
                                          seed=5652)
        out = out['power']
        out_eff = out_eff['power']
        print(f"{out=} {out_eff=}")
        assert np.isclose(out_eff, out, rtol=2)


def check_function_mpmk_check_step():
    dp_2part = DetectionPowerSlope(
        significance_mode='n-section-mann-kendall', nsims=100,
        expect_slope=[1, -1], nparts=2, min_part_size=10, no_trend_alpha=0.50,
        return_true_conc=False, return_noisy_conc_itters=0, efficent_mode=False,
        mpmk_check_step=4, mpmk_efficent_min=10, mpmk_window=0.05, )

    def check_step_func(n):
        if n > 130:
            return 5
        else:
            return 4

    dp_2part_func = DetectionPowerSlope(
        significance_mode='n-section-mann-kendall', nsims=100,
        expect_slope=[1, -1], nparts=2, min_part_size=10, no_trend_alpha=0.50,
        return_true_conc=False, return_noisy_conc_itters=0, efficent_mode=False,
        mpmk_check_step=check_step_func, mpmk_efficent_min=10, mpmk_window=0.05, )

    from kendall_stats import make_example_data
    x_inc, y_inc = make_example_data.make_multipart_sharp_change_data(make_example_data.multipart_sharp_slopes[0],
                                                                      noise=0,
                                                                      na_data=False, unsort=False)
    assert len(x_inc) <= 130
    noises = [1, 2.5, 5]
    diffs = [0.1, 0.5, 6]
    for noise, allow_dif in zip(noises, diffs):
        print(f'testing efficiency on sharp data {noise=}')
        out = dp_2part.power_calc(idv='sharp', error=noise, true_conc_ts=y_inc, mrt_model='pass_true_conc', seed=688765)
        out2 = dp_2part_func.power_calc(idv='sharp', error=noise, true_conc_ts=y_inc, mrt_model='pass_true_conc',
                                        seed=688765)
        out = pd.Series(out)
        out2 = pd.Series(out2)
        pd.testing.assert_series_equal(out, out2)


def test_kwarg_passing():
    t = ('Now Mrs. Riley, and only Mrs. Riley, how many fingers am I holding up?  Four! What do you think now dear?'
         "I'm thinking of getting stronger glasses.")
    unique_kwargs = t.split(' ')

    dcp = DetectionPowerSlope(significance_mode='linear-regression', return_true_conc=False,
                              return_noisy_conc_itters=0, efficent_mode=False)
    got_single = []
    for kwarg in unique_kwargs:
        out = dcp.power_calc(idv='norm_inc', error=0.1,
                             true_conc_ts=np.arange(10),
                             seed=5585, mult_kwarg=kwarg, single_kwarg='single')
        got_single.append(out['mult_kwarg'])
        assert out['single_kwarg'] == 'single'

    out = dcp.mulitprocess_power_calcs(None,
                                       idv_vals=['norm_inc'] * len(unique_kwargs),
                                       error_vals=[0.1] * len(unique_kwargs),
                                       true_conc_ts_vals=[np.zeros(10)] * len(unique_kwargs),
                                       seed_vals=np.array([5585] * len(unique_kwargs)),
                                       mult_kwarg_vals=unique_kwargs,
                                       single_kwarg='single')
    assert set(out['mult_kwarg'].unique()) == set(unique_kwargs) == set(got_single)
    assert np.all(out['single_kwarg'] == 'single')


def test_condenced_non_condenced():
    save_path = Path(__file__).parent.joinpath('test_data', 'test_condenced_non_condenced_slope.hdf')
    save_data = False
    kwargs = dict(
        outpath=None,
        idv_vals=np.arange(9),
        error_vals=np.array([5.1, 5.101, 5.10001] * 3),
        max_conc_lim_vals=np.array([20.001, 20, 20.1] * 3),
        initial_conc_vals=np.array([10, 10.001, 10.1] * 3),
        target_conc_vals=np.array([6, 6.001, 6.1] * 3),
        mrt_vals=np.array([1.4, 1.1, 1.0] * 3),
        mrt_p1_vals=np.array([1.4, 1.1, 1.0] * 3),

        samp_years_vals=10,
        samp_per_year_vals=5,
        implementation_time_vals=5,
        prev_slope_vals=0,
        min_conc_lim_vals=1,
        mrt_model_vals='binary_exponential_piston_flow',
        frac_p1_vals=1,
        f_p1_vals=0.7,
        f_p2_vals=0.7,
        seed_vals=5585,
    )

    dp_auto = AutoDetectionPowerSlope()
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

    out_cond = dp_auto.mulitprocess_power_calcs(**kwargs)
    if save_data:
        out_raw.to_hdf(save_path, key='out_raw')
        out_cond.to_hdf(save_path, key='out_cond')
    expect_raw = pd.read_hdf(save_path, key='out_raw')
    assert isinstance(expect_raw, pd.DataFrame)
    expect_cond = pd.read_hdf(save_path, key='out_cond')
    assert isinstance(expect_cond, pd.DataFrame)
    pd.testing.assert_frame_equal(out_raw, expect_raw)
    pd.testing.assert_frame_equal(out_cond, expect_cond)


if __name__ == '__main__':
    plot_flag = False
    make_test_power_calc_runs(plot_flag)
    test_power_calc_and_mp()

    test_condenced_non_condenced()
    test_kwarg_passing()
    test_pettitt_power(show=plot_flag)
    test_unitary_epfm_slope(plot=plot_flag)
    test_piston_flow(plot=plot_flag)
    test_unitary_epfm(plot=plot_flag)
    test_bepfm_slope(plot=plot_flag)
    test_bpefm(plot=plot_flag)
    test_iteration_plotting(show=plot_flag)

    test_efficient_mode_lr()
    test_efficent_mode_mann_kendall()
    test_efficient_mode_mpmk()
    check_function_mpmk_check_step()
    test_return_true_noisy_conc(show=plot_flag)

    test_multpart_mann_kendall_power(show=plot_flag)
    test_linear_from_max_vs_from_start(show=plot_flag)
    test_mann_kendall_power(show=plot_flag)

    print('passed all tests')
