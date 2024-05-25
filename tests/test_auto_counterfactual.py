"""
created matt_dumont 
on: 24/05/24
"""
import unittest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import itertools
from komanawa.gw_detect_power import DetectionPowerCounterFactual, AutoDetectionPowerCounterFactual

class TestAutoCounterFactual(unittest.TestCase):
    def test_auto_true_conc(self, plot=False):
        figs = []
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
                figs.append(fig)
                ax.plot(t.index, t['true_conc_alt'].values, label='alt_conc', marker='o', color='r')
                ax.plot(t.index, t['true_conc_base'].values, label='base_conc', marker='o', color='b')
                ax.legend()
                ax.set_title(idv)
        if plot:
            plt.show()
            for fig in figs:
                plt.close(fig)

        if save_data:
            for k, v in got.items():
                v.to_hdf(save_path, key=k)

        for k, v in got.items():
            expect = pd.read_hdf(save_path, key=k)
            self.assertIsInstance(expect, pd.DataFrame)
            pd.testing.assert_frame_equal(v, expect, check_dtype=False, check_like=True, check_exact=False)


    def test_power_calc_auto(self):
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
        self.assertIsInstance(expect, pd.DataFrame)
        pd.testing.assert_frame_equal(got, expect, check_dtype=False, check_like=True, check_exact=False)


    def test_multiprocess_power_calc_auto(self):
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
        self.assertTrue(all(out['single_kwarg'] == 'test_single_kwarg'))
        self.assertEqual(set(out['multi_kwargs']), set(unique_kwargs))


    def test_condenced_non_condenced(self):
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
        self.assertIsInstance(expect_raw, pd.DataFrame)
        expect_condensed = pd.read_hdf(save_path, key='condensed')
        self.assertIsInstance(expect_condensed, pd.DataFrame)
        pd.testing.assert_frame_equal(out_raw, expect_raw, check_dtype=False, check_like=True, check_exact=False)
        pd.testing.assert_frame_equal(out_condensed, expect_condensed, check_dtype=False, check_like=True,
                                      check_exact=False)


    def test_compare_auto_manual(self):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        from komanawa.kendall_stats import MannKendall
        import datetime
        from komanawa.gw_age_tools import check_age_inputs, predict_historical_source_conc, predict_future_conc_bepm
        from scipy.interpolate import interp1d

        raw_n_vals = np.array([
            2.8, 4.1, 2.4, 2.1, 1.7, 3.4, 1.5, 2., 1.4, 1.73, 1.3,
            1.4, 1.4, 1.3, 7.5, 1., 1.7, 1.1, 1.4, 1.7, 2.4, 2.7, 4.6,
            3.4, 5.9, 4.8, 5.2, 2.8, 3.1, 3.6, 4.7, 4.4, 3.8, 4.6,
            3.9, 3.5, 3.6, 3.7, 3.6, 3.9, 4.1, 3.8, 3.6, 3.8, 4.7,
            6.3, 5.1, 6.7, 7., 7.4, 6.4, 5., 5.3, 5.8, 4.8, 5.8,
            5.7, 5., 4.4, 4.4, 6.3, 6.5, 6.7, 7.3, 7.5, 7.4, 7.,
            7.6])

        raw_n_dates = np.array([
            '1986-11-06', '1987-11-18', '1988-11-03', '1989-10-03', '1991-09-23', '1992-10-12', '1993-09-28',
            '1994-09-28',
            '1995-10-02', '1996-10-14', '1997-10-06', '1998-09-30', '1998-09-30', '1999-04-07', '1999-10-12',
            '2000-05-12',
            '2000-11-29', '2001-10-11', '2002-10-22', '2003-10-16', '2004-10-19', '2005-10-26', '2006-10-25',
            '2007-12-13',
            '2008-11-12', '2009-11-10', '2010-12-13', '2011-11-14', '2012-10-30', '2013-11-19', '2014-02-07',
            '2014-05-07',
            '2014-08-04', '2014-11-05', '2015-02-05', '2015-05-04', '2015-08-03', '2015-11-05', '2016-02-02',
            '2016-05-04',
            '2016-08-04', '2016-11-08', '2017-02-23', '2017-05-23', '2017-08-24', '2017-11-28', '2018-02-27',
            '2018-05-17',
            '2018-08-29', '2018-11-22', '2019-02-26', '2019-05-30', '2019-08-20', '2019-11-22', '2020-01-28',
            '2020-07-21',
            '2020-10-28', '2021-01-14', '2021-04-15', '2021-07-28', '2021-10-26', '2022-01-19', '2022-04-20',
            '2022-07-28',
            '2022-10-14', '2023-01-26', '2023-04-04', '2023-07-12'])

        ndata = pd.DataFrame({'n': raw_n_vals, 'date': pd.to_datetime(raw_n_dates)})
        ndata = ndata.loc[ndata.date > '2001-01-01'].set_index('date')

        mk = MannKendall(ndata['n'], alpha=0.05)
        senslope, senintercept, lo_slope, up_slope = mk.calc_senslope()

        ndata['predicted'] = senslope * ndata.index.values.astype(float) + senintercept
        ndata['residual'] = ndata['n'] - ndata['predicted']

        input_error = ndata['residual'].std()

        mrt = 17.5
        mrt_p1 = 17.5
        mrt_p2 = 17.5  # dummy value (only one EPM)
        frac_p1 = 1
        precision = 2  # calculate the historical source  at 0.01-year intervals (roughly monthly)
        f_p1 = 0.625
        f_p2 = 0.625  # dummy value (only one EPM)
        max_conc = 20  # maximum concentration the source area could have
        min_conc = 1  # minimum concentration the source area could have
        p0 = None  # no guess for scipy curve fit

        # calculate the historical slope in units years from 2001-10-11 instead of datetime
        ndata_yr = ndata.copy()
        ndata_yr['yr'] = (ndata_yr.index - ndata_yr.index.min()).days / 365.25
        ndata_yr = ndata_yr.set_index('yr')
        mk = MannKendall(ndata_yr['n'], alpha=0.05)
        senslope, senintercept, lo_slope, up_slope = mk.calc_senslope()
        prev_slope = senslope

        # use the last time step sen fit as the initial concentration
        init_conc = ndata_yr.index[-1] * senslope + senintercept

        # check the age inputs
        mrt, mrt_p2 = check_age_inputs(mrt, mrt_p1, mrt_p2, frac_p1, precision, f_p1, f_p2)
        start_age = max(mrt, mrt_p1, mrt_p2, 30)

        # predict the historical source concentration
        hist = predict_historical_source_conc(init_conc, mrt, mrt_p1, mrt_p2, frac_p1, f_p1, f_p2, prev_slope, max_conc,
                                              min_conc, start_age=start_age, precision=precision)

        hist.name = 'source_conc'
        hist.index.name = 'yr_from_present'

        hist = pd.DataFrame(hist).reset_index()
        hist['date'] = [ndata.index.max().date() + datetime.timedelta(days=e) for e in hist['yr_from_present'] * 365.25]

        base_scenario_source_conc = hist.set_index('yr_from_present')['source_conc']
        alt_scenario_source_conc = base_scenario_source_conc.copy()

        base_scenario_source_conc.loc[10] = base_scenario_source_conc.loc[0]
        base_scenario_source_conc.loc[55] = base_scenario_source_conc.loc[0]
        alt_scenario_source_conc.loc[10] = base_scenario_source_conc.loc[0] * 0.85
        alt_scenario_source_conc.loc[55] = base_scenario_source_conc.loc[0] * 0.85

        predict_start = (ndata.index.min() - ndata.index.max()).days / 365.25
        predict_stop = 55

        base_receptor = predict_future_conc_bepm(
            once_and_future_source_conc=base_scenario_source_conc,
            predict_start=predict_start, predict_stop=predict_stop,
            mrt_p1=mrt_p1, frac_p1=frac_p1, f_p1=f_p1, f_p2=f_p2, mrt=mrt, mrt_p2=mrt_p2,
            fill_value=min_conc,
            fill_threshold=0.05, precision=2, pred_step=0.01)

        alt_receptor = predict_future_conc_bepm(
            once_and_future_source_conc=alt_scenario_source_conc,
            predict_start=predict_start, predict_stop=predict_stop,
            mrt_p1=mrt_p1, frac_p1=frac_p1, f_p1=f_p1, f_p2=f_p2, mrt=mrt, mrt_p2=mrt_p2,
            fill_value=min_conc, fill_threshold=0.05, precision=2, pred_step=0.01)

        dpc = DetectionPowerCounterFactual(significance_mode='wilcoxon-signed-rank-test',
                                           nsims=1000,
                                           p_value=0.05,
                                           alternative='alt<base')

        nsamples_per_year = [12, 52]
        sampling_durations = [20, 30, 50]
        sampling_delays = [0, 10]

        out_data = []

        base_sampler = interp1d(base_receptor.index, base_receptor.values)
        alt_sampler = interp1d(alt_receptor.index, alt_receptor.values)

        for nsamp in nsamples_per_year:
            for dur in sampling_durations:
                for delay in sampling_delays:
                    idv = f'{nsamp}_{dur}_{delay}'
                    in_base = base_sampler(np.arange(delay, dur + 1 / nsamp, 1 / nsamp))
                    in_alt = alt_sampler(np.arange(delay, dur + 1 / nsamp, 1 / nsamp))
                    out = dpc.power_calc(
                        idv=idv,
                        error_base=input_error,
                        true_conc_base=in_base,
                        true_conc_alt=in_alt,
                        error_alt=input_error,
                        seed_base=1,  # setting the seeds ensures that the process is reproducible
                        seed_alt=2,

                        # the following data is passed right to the output pd.Series
                        nsamples_per_year=nsamp,
                        sampling_duration=dur,
                        sampling_delay=delay,
                    )
                    out_data.append(out)

        out_data = pd.DataFrame(out_data).set_index('idv')

        print('an example of the output data')
        print(out_data.head())

        auto_dpc = AutoDetectionPowerCounterFactual(significance_mode='wilcoxon-signed-rank-test',
                                                    nsims=1000,
                                                    p_value=0.05,
                                                    alternative='alt<base')

        auto_outdata = []
        for nsamp in nsamples_per_year:
            for dur in sampling_durations:
                for delay in sampling_delays:
                    idv = f'{nsamp}_{dur}_{delay}'
                    out = auto_dpc.power_calc(
                        idv=idv,
                        error_base=input_error,
                        mrt_model='binary_exponential_piston_flow',
                        samp_years=dur,
                        samp_per_year=nsamp,
                        implementation_time_alt=10,
                        initial_conc=init_conc,
                        target_conc_alt=10.77669 * 0.85,
                        # 15% reduction from source concentration at time 0 to make it directly comparable to the other method.
                        prev_slope=prev_slope,
                        max_conc_lim=max_conc,
                        min_conc_lim=min_conc,
                        mrt=mrt,
                        target_conc_base=10.77669,  # source concentration at time 0
                        implementation_time_base=10,
                        error_alt=None,
                        delay_years=delay,

                        #
                        mrt_p1=mrt_p1,
                        frac_p1=frac_p1,
                        f_p1=f_p1,
                        f_p2=f_p2,
                        #
                        seed_base=1,
                        seed_alt=2,
                    )
                    auto_outdata.append(out)

        auto_outdata = pd.DataFrame(auto_outdata).set_index('idv')
        if np.allclose(auto_outdata.power, out_data.loc[auto_outdata.index].power, atol=2):
            print('The power values are the same')
        else:
            for k in out_data.index:
                print('idv', 'manual', 'auto')
                print(k, out_data.loc[k].power, auto_outdata.loc[k].power)
            raise ValueError('The power values are different')

if __name__ == '__main__':
    unittest.main()