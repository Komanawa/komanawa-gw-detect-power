"""
created matt_dumont 
on: 17/07/23
"""
from exponential_piston_flow import *
from pathlib import Path


def test_binary_piston_flow():
    """

    :return:
    """

    tm1 = 1
    tm2 = 15
    f_p1 = 0.75
    f_p2 = 0.8
    frac_p1 = 0.5
    step = 0.5
    t = np.arange(0.1, 6., step)
    out = binary_exp_piston_flow(t, tm1, tm2, frac_p1, f_p1, f_p2)
    out2 = binary_exp_piston_flow_cdf(t, tm1, tm2, frac_p1, f_p1, f_p2)
    expect_1 = [0.0, 0.41805939018203736, 0.2146388476917839, 0.11019925881439102, 0.05657818598001161,
                0.029048209245947743, 0.05623473509739689, 0.04729158417881503, 0.04194829909552831,
                0.03848392479037335, 0.036013641488773174, 0.03408196667833277]
    expect_2 = [0.0, 0.18645545736347197, 0.33902086423116207, 0.4173505558892067, 0.4575663605149913,
                0.4782138430655392, 0.4929639677524792, 0.5186425191311791, 0.5408469466440985, 0.5608995631058383,
                0.5794942903978542, 0.5970018106077799]
    assert np.allclose(out, expect_1)
    assert np.allclose(out2, expect_2)
    step = 0.00001
    t = np.arange(0.1, 6., step)
    out = binary_exp_piston_flow(t, tm1, tm2, frac_p1, f_p1, f_p2)
    out2 = binary_exp_piston_flow_cdf(t, tm1, tm2, frac_p1, f_p1, f_p2)

    # test approximate integration of pdf vs cdf
    assert np.allclose((out * step).cumsum(), out2)


def test_get_source_initial_conc_bepm(plot=False):
    mrt = 20
    mrt_p1 = 10
    frac_p1 = 0.7
    f_p1 = 0.8
    f_p2 = 0.75
    init_conc = 10
    prev_slope = 0.5
    max_conc = 20
    min_conc = 1.
    mrt_p2 = (mrt - (mrt_p1 * frac_p1)) / (1 - frac_p1)

    # make cdf of age
    age_step = 0.01
    ages = np.arange(0, np.nanmax([mrt_p1, mrt_p2]) * 5, age_step).round(2)  # approximately monthly steps
    age_cdf = binary_exp_piston_flow_cdf(ages, mrt_p1, mrt_p2, frac_p1, f_p1, f_p2)
    age_fractions = np.diff(age_cdf, prepend=0)

    source_conc = get_source_initial_conc_bepm(init_conc, mrt_p1, mrt_p2, age_step, ages, age_fractions, prev_slope,
                                               max_conc,
                                               min_conc,
                                               make_past_conc=False, start_age=np.nan)
    test_data_path = Path(__file__).parent.joinpath('test_data', 'test_get_source_initial_conc_bepm.hdf')
    write_test_data = False
    if write_test_data:
        test_data_path.unlink(missing_ok=True)
        source_conc.to_hdf(test_data_path, 'source_conc')
    true_source_conc = pd.read_hdf(test_data_path, 'source_conc')
    assert np.allclose(source_conc, true_source_conc)

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(source_conc.index, source_conc.values)
        ax.axvline(0, color='k', linestyle='--')
        plt.show()


def test_predict_source_future_past_conc_bepm(plot=False):
    mrt = 20
    mrt_p1 = 10
    frac_p1 = 0.7
    f_p1 = 0.8
    f_p2 = 0.75
    initial_conc = 10
    prev_slope = 0.5
    max_conc = 20
    min_conc = 1.
    age_range = (-20, 50)
    fut_slope = -0.1
    total_source_conc, receptor_conc = predict_source_future_past_conc_bepm(initial_conc, mrt, mrt_p1, frac_p1, f_p1,
                                                                            f_p2,
                                                                            prev_slope, fut_slope, age_range,
                                                                            max_conc, min_conc, max_fut_conc=20,
                                                                            min_fut_conc=1)
    test_data_path = Path(__file__).parent.joinpath('test_data', 'test_predict_source_future_past_conc_bepm.hdf')

    write_test_data = False
    if write_test_data:
        test_data_path.unlink(missing_ok=True)
        total_source_conc.to_hdf(test_data_path, 'total_source_conc')
        receptor_conc.to_hdf(test_data_path, 'receptor_conc')
    true_total_source_conc = pd.read_hdf(test_data_path, 'total_source_conc')
    true_receptor_conc = pd.read_hdf(test_data_path, 'receptor_conc')
    assert np.allclose(total_source_conc.values, true_total_source_conc.values)
    assert np.allclose(receptor_conc.values, true_receptor_conc.values)

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(total_source_conc.index, total_source_conc.values, label='source_conc', color='b')
        ax.plot(receptor_conc.index, receptor_conc.values, label='receptor_conc', color='r')
        ax.axvline(0, color='k', ls='--', label='initial time')
        ax.set_xlabel('time (years)')
        ax.set_ylabel('concentration')
        ax.set_xlim(age_range)
        ax.legend()
        plt.show()


def test_predict_future_conc_bepm(plot=False):
    input_series = pd.Series(index=[-35, 0., 20, 27, 35, 40, 50,100, 200],
                             data=[1, 1, 3, 5, 15, 21, 18, 2.4, 2.4])
    mrt = 20
    mrt_p1 = 5
    frac_p1 = 0.2
    f_p1 = 0.8
    f_p2 = 0.75
    data = predict_future_conc_bepm(once_and_future_source_conc=input_series,
                                    predict_start=20,
                                    predict_stop=200,
                                    mrt_p1=mrt_p1, frac_p1=frac_p1, f_p1=f_p1, f_p2=f_p2, mrt=mrt, mrt_p2=None,
                                    fill_value=1,
                                    fill_threshold=0.05,
                                    pred_step=0.5)
    write_test_data = False
    test_path = Path(__file__).parent.joinpath('test_data', 'test_predict_future_conc_bepm.hdf')
    if write_test_data:
        test_path.unlink(missing_ok=True)
        data.to_hdf(test_path, 'data')
    true_data = pd.read_hdf(test_path, 'data')
    assert np.allclose(data.values, true_data.values)


    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(input_series.index, input_series.values, label='input')
        ax.plot(data.index, data.values, label='output')
        ax.legend()
        plt.show()

    # test with missing source concentration
    try:
        data = predict_future_conc_bepm(once_and_future_source_conc=input_series,
                                        predict_start=0,
                                        predict_stop=200,
                                        mrt_p1=mrt_p1, frac_p1=frac_p1, f_p1=f_p1, f_p2=f_p2, mrt=mrt, mrt_p2=None,
                                        fill_value=1,
                                        fill_threshold=0.05,
                                        pred_step=0.5)
        assert False, 'should have raised an error'
    except ValueError as val:
        assert 'the source concentration is missing' in str(val)


if __name__ == '__main__':
    plot_tests = False
    test_predict_future_conc_bepm(plot_tests)
    test_get_source_initial_conc_bepm(plot_tests)
    test_predict_source_future_past_conc_bepm(plot_tests)
    test_binary_piston_flow()
