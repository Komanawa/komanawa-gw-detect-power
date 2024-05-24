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
from komanawa.gw_detect_power import DetectionPowerCounterFactual
from base_testers import BaseTesterCounter


class TestDetectionPowerCounterFactual(unittest.TestCase):

    def test_plot_iteration(self, plot=False):
        figs = []
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
        base, alt = BaseTesterCounter.make_bilinar_test_data(0.1, -0.1, 100)
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
            figs.append(fig)
            ax.set_title(f'itter {i}')

        if plot:
            plt.show()
        for fig in figs:
            plt.close(fig)

    def test_power_calc_functionality(self):
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
        ts1, ts2 = BaseTesterCounter.make_step_test_data(0.5, 100)
        with self.assertRaises(ValueError):
            dp.power_calc(idv='test',
                          error_base=1,
                          error_alt=1,
                          true_conc_base=ts1,
                          true_conc_alt=ts2,
                          seed_alt=1,
                          seed_base=1,
                          )

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
        self.assertIn('african_swallow', temp.keys(), 'kwargs should be passed through power_calc')
        self.assertEqual(temp['african_swallow'], 'non-migratory', 'kwargs should be passed through power_calc')

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
        self.assertIsInstance(out, dict)
        self.assertIn('power', out.keys(), 'power should be returned')
        self.assertIn('true_conc', out.keys(), 'true_conc should be returned')
        self.assertIn('alt_noisy_conc', out.keys(), 'alt_noisy_conc should be returned')
        self.assertIn('base_noisy_conc', out.keys(), 'base_noisy_conc should be returned')
        self.assertIn('significant', out.keys(), 'significant should be returned')
        self.assertIsInstance(out['power'], pd.Series, 'power should be a series')
        self.assertTrue(np.in1d(
            ['power', 'idv', 'error_base',
             'error_alt', 'seed_base', 'seed_alt'], out['power'].index).all(), 'power should have correct index')
        self.assertIsInstance(out['true_conc'], pd.DataFrame, 'true_conc should be a DataFrame')
        self.assertEqual(out['true_conc'].shape, (100, 2), 'true_conc should have correct shape')
        for k in ['alt_noisy_conc', 'base_noisy_conc']:
            self.assertIsInstance(out[k], pd.DataFrame, f'{k} should be a DataFrame')
            self.assertEqual(out[k].shape, (100, 5), f'{k} should have correct shape')
        self.assertEqual(out['significant'].shape, (5,), 'significant should have correct shape')
        self.assertEqual(out['significant'].dtype, bool, 'significant should be a bool')

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
        self.assertEqual(out['significant'].shape, (5,), 'significant should have correct shape')
        self.assertEqual(out['significant'].dtype, bool, 'significant should be a bool')
        self.assertTrue(out['significant'].all(), 'all significant should be true')

    def test_paired_ttest_power(self):
        save_path = Path(__file__).parent.joinpath('test_data', 'test_paired_ttest_power_counter.hdf')
        save_data = False
        got = []
        noises = [0.1, 1, 10, 100, 1000]
        data = {}
        base, alt = BaseTesterCounter.make_step_test_data(1, 100)
        data['step'] = (base, alt)
        base, alt = BaseTesterCounter.make_linear_test_data(0.1, 100)
        data['linear'] = (base, alt)
        base, alt = BaseTesterCounter.make_bilinar_test_data(0.1, -0.1, 100)
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
        self.assertIsInstance(expect, pd.DataFrame)
        pd.testing.assert_frame_equal(got, expect, check_dtype=False, check_like=True, check_exact=False)

    def test_wilcoxon_power(self):
        save_path = Path(__file__).parent.joinpath('test_data', 'test_wilcoxon_power_counter.hdf')
        save_data = False
        got = []
        noises = [0.1, 1, 10, 100, 1000]
        data = {}
        base, alt = BaseTesterCounter.make_step_test_data(1, 100)
        data['step'] = (base, alt)
        base, alt = BaseTesterCounter.make_linear_test_data(0.1, 100)
        data['linear'] = (base, alt)
        base, alt = BaseTesterCounter.make_bilinar_test_data(0.1, -0.1, 100)
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
        self.assertIsInstance(expect, pd.DataFrame)
        pd.testing.assert_frame_equal(got, expect, check_dtype=False, check_like=True, check_exact=False)

    def test_multiprocess_power_calc(self):
        got = []
        noise_alt = [0.1, 1, 10]
        noises = [0.1, 1, 10, 100, 1000]
        data = {}
        base, alt = BaseTesterCounter.make_step_test_data(1, 100)
        data['step'] = (base, alt)
        base, alt = BaseTesterCounter.make_linear_test_data(0.1, 100)
        data['linear'] = (base, alt)
        base, alt = BaseTesterCounter.make_bilinar_test_data(0.1, -0.1, 100)
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

        self.assertEqual(len(unique_kwargs), len(idvs), 'bad test design')

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
        self.assertIsNone(out, 'run=False should return None')

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
        self.assertTrue(all(out['single_kwarg'] == 'test_single_kwarg'))
        self.assertEqual(set(out['mult_kwargs']), set(unique_kwargs))
