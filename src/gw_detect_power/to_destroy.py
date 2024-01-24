"""
created matt_dumont 
on: 24/01/24
"""


class DetectionPowerCalculator:
    implemented_mrt_models = (
        'piston_flow',
        'binary_exponential_piston_flow',
        'pass_true_conc',
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
        raise


    def mulitprocess_power_calcs(  # todo propogate over
            self,
            outpath: {Path, None, str},
            id_vals: np.ndarray,
            error_vals: {np.ndarray, float},
            samp_years_vals: {np.ndarray, int, None} = None,
            samp_per_year_vals: {np.ndarray, int, None} = None,
            implementation_time_vals: {np.ndarray, int, None} = None,
            initial_conc_vals: {np.ndarray, float, None} = None,
            target_conc_vals: {np.ndarray, float, None} = None,
            previous_slope_vals: {np.ndarray, float, None} = None,
            max_conc_vals: {np.ndarray, float, None} = None,
            min_conc_vals: {np.ndarray, float, None} = None,
            mrt_model_vals: {np.ndarray, str} = 'piston_flow',
            mrt_vals: {np.ndarray, float} = 0.0,
            mrt_p1_vals: {np.ndarray, float, None} = None,
            frac_p1_vals: {np.ndarray, float, None} = None,
            f_p1_vals: {np.ndarray, float, None} = None,
            f_p2_vals: {np.ndarray, float, None} = None,
            true_conc_ts_vals: {np.ndarray, list, None} = None,
            seed: {np.ndarray, int, None} = 5585,
            run=True,
    ):
        """
        multiprocessing wrapper for power_calc, see power_calc for details
        note that if a given run raises and exception the traceback for the exception will be included in the
        returned dataset under the column 'python_error' if 'python_error' is None then the run was successful
        to change the number of cores used pass n_cores to the constructor init

        :param outpath: path to save results to
        :param id_vals: id values for each simulation
        All values from here on out should be either a single value or an array of values with the same shape as id_vals
        :param error_vals: standard deviation of noise to add for each simulation
        :param samp_years_vals: number of years to sample for each simulation
        :param samp_per_year_vals: number of samples per year for each simulation
        :param implementation_time_vals: the length of implementation of reduction for each simulation
        :param initial_conc_vals: the initial concentration for each simulation
        :param target_conc_vals: the target concentration for each simulation
        :param previous_slope_vals: the previous slope of the source concentration
               for each simulation units conc/yr
        :param max_conc_vals: the maximum concentration for each simulation if None then no maximum
                              (based solely on the previous slope and lag time)
        :param min_conc_vals: the minimum concentration for each simulation only used for exponential piston flow
        :param mrt_model_vals: the mrt model to use for each simulation
        :param mrt_vals: the mean residence time for each simulation
        :param mrt_p1_vals: the mean residence time for the first piston for each simulation (only used for
                            bianary exponential piston flow)
        :param frac_p1_vals:  the fraction of the total mass in the first piston for each simulation (only used for
                              bianary exponential piston flow)
                              set to 1 for an exponential piston flow model, otherwise set to None
        :param f_p1_vals: the fraction of the total mass in the first piston for each simulation (only used for
                            bianary exponential piston flow) otherwise set to None
        :param f_p2_vals: the fraction of the total mass in the first piston for each simulation (only used for
                            bianary exponential piston flow) otherwise set to None
        :param true_conc_ts_vals: the true concentration time series for each simulation only used for the
                                    'pass_true_conc' mrt_model, note that this can be a list of arrays of different
                                    lengths for each simulation, Numpy does not support jagged arrays
        :param seed: ndarray (integer seeds), None (no seeds), or int (1 seed for all simulations)
        :param run: if True run the simulations, if False just build  the run_dict and print the number of simulations
        :return: dataframe with input data and the results of all of the power calcs. note power is percent 0-100
        """
        if self.return_true_conc or self.return_noisy_conc_itters > 0:
            warnings.warn('return_true_conc and return_noisy_conc_itters are not supported for mulitprocess_power_calcs'
                          'only power results will be returned')

        if isinstance(outpath, str):
            outpath = Path(outpath)
        id_vals = np.atleast_1d(id_vals)
        expect_shape = id_vals.shape

        if isinstance(mrt_model_vals, str):
            mrt_model_vals = np.array([mrt_model_vals] * len(id_vals))
        mrt_model_vals = np.atleast_1d(mrt_model_vals)
        assert mrt_model_vals.shape == id_vals.shape, f'mrt_model_vals and mrt_vals must have the same shape'
        assert np.in1d(mrt_model_vals, self.implemented_mrt_models).all(), (
            f'mrt_model_vals must be one of {self.implemented_mrt_models} '
            f'got {np.unique(mrt_model_vals)}')
        if any(mrt_model_vals == 'binary_exponential_piston_flow'):
            assert age_tools_imported, (
                'cannot run binary_exponential_piston_flow model, age_tools not installed'
                'to install run:\n'
                'pip install git+https://github.com/Komanawa-Solutions-Ltd/gw_age_tools')

        # manage multiple size options
        error_vals = self._adjust_shape(error_vals, expect_shape, none_allowed=False, is_int=False,
                                        idv='error_vals')
        samp_years_vals = self._adjust_shape(samp_years_vals, expect_shape, none_allowed=True, is_int=True,
                                             idv='samp_years_vals')
        samp_per_year_vals = self._adjust_shape(samp_per_year_vals, expect_shape, none_allowed=True, is_int=True,
                                                idv='samp_per_year_vals')
        implementation_time_vals = self._adjust_shape(implementation_time_vals, expect_shape, none_allowed=True,
                                                      is_int=True,
                                                      idv='implementation_time_vals')
        initial_conc_vals = self._adjust_shape(initial_conc_vals, expect_shape, none_allowed=True, is_int=False,
                                               idv='initial_conc_vals')
        target_conc_vals = self._adjust_shape(target_conc_vals, expect_shape, none_allowed=True, is_int=False,
                                              idv='target_conc_vals')
        previous_slope_vals = self._adjust_shape(previous_slope_vals, expect_shape, none_allowed=True, is_int=False,
                                                 idv='previous_slope_vals')
        max_conc_vals = self._adjust_shape(max_conc_vals, expect_shape, none_allowed=True, is_int=False,
                                           idv='max_conc_vals')
        min_conc_vals = self._adjust_shape(min_conc_vals, expect_shape, none_allowed=True, is_int=False,
                                           idv='min_conc_vals')
        mrt_vals = self._adjust_shape(mrt_vals, expect_shape, none_allowed=True, is_int=False, idv='mrt_vals')
        mrt_p1_vals = self._adjust_shape(mrt_p1_vals, expect_shape, none_allowed=True, is_int=False, idv='mrt_p1_vals')
        frac_p1_vals = self._adjust_shape(frac_p1_vals, expect_shape, none_allowed=True, is_int=False,
                                          idv='frac_p1_vals')
        f_p1_vals = self._adjust_shape(f_p1_vals, expect_shape, none_allowed=True, is_int=False, idv='f_p1_vals')
        f_p2_vals = self._adjust_shape(f_p2_vals, expect_shape, none_allowed=True, is_int=False, idv='f_p2_vals')
        true_conc_ts_vals = self._check_propogate_truets(true_conc_ts_vals, expect_shape)
        use_seeds = self._adjust_shape(seed, expect_shape, none_allowed=True, is_int=True, idv='seed')
        not_na_idx = pd.notna(max_conc_vals) & pd.notna(initial_conc_vals)
        assert (max_conc_vals[not_na_idx] >= initial_conc_vals[
            not_na_idx]).all(), 'max_conc must be greater than or equal to initial_conc'
        not_na_idx = pd.notna(max_conc_vals) & pd.notna(target_conc_vals)
        assert (max_conc_vals[not_na_idx] >= target_conc_vals[
            not_na_idx]).all(), 'max_conc must be greater than or equal to target_conc'

        run_vals = [
            id_vals,
            error_vals,
            samp_years_vals,
            samp_per_year_vals,
            implementation_time_vals,
            initial_conc_vals,
            target_conc_vals,
            previous_slope_vals,
            max_conc_vals,
            min_conc_vals,
            mrt_model_vals,
            mrt_vals,
            mrt_p1_vals,
            frac_p1_vals,
            f_p1_vals,
            f_p2_vals,
            true_conc_ts_vals,
            use_seeds

        ]
        runs = []
        run_list = []
        all_use_idv = []
        # make runs
        print('creating and condensing runs')
        for i, (
                idv, error, samp_years, samp_per_year, implementation_time,
                inital_conc, target_conc, previous_slope, max_conc, min_conc,
                mrt_model, mrt, mrt_p1, frac_p1, f_p1, f_p2, true_conc_ts, seed_val
        ) in enumerate(zip(*run_vals)):
            # create unique value so that if multiple sites are passed with the same values (within rounding error)
            # they are only run once
            if true_conc_ts is None:
                ptis = None
            else:
                ptis = i
            if i % 1000 == 0:
                print(f'forming/condesing run {i} of {len(id_vals)}')

            use_idv = (
                self._get_id_str(error, 'error'),
                self._get_id_str(samp_years, 'samp_years'),
                self._get_id_str(samp_per_year, 'samp_per_year'),
                self._get_id_str(implementation_time, 'implementation_time'),
                self._get_id_str(inital_conc, 'inital_conc'),
                self._get_id_str(target_conc, 'target_conc'),
                self._get_id_str(previous_slope, 'previous_slope', 2),
                self._get_id_str(max_conc, 'max_conc'),
                self._get_id_str(min_conc, 'min_conc'),
                self._get_id_str(mrt_model, 'mrt_model'),
                self._get_id_str(mrt, 'mrt', 0),
                self._get_id_str(mrt_p1, 'mrt_p1'),
                self._get_id_str(frac_p1, 'frac_p1', 2),
                self._get_id_str(f_p1, 'f_p1', 2),
                self._get_id_str(f_p2, 'f_p2', 2),
                self._get_id_str(ptis, 'ptis'),
            )
            use_idv = '_'.join(use_idv)
            all_use_idv.append(use_idv)
            if use_idv in run_list:
                continue
            run_list.append(use_idv)

            runs.append(dict(
                idv=use_idv,
                error=error,
                samp_years=samp_years,
                samp_per_year=samp_per_year,
                implementation_time=implementation_time,
                initial_conc=inital_conc,
                target_conc=target_conc,
                prev_slope=previous_slope,
                max_conc=max_conc,
                min_conc=min_conc,
                mrt_model=mrt_model,
                mrt=mrt,
                mrt_p1=mrt_p1,
                frac_p1=frac_p1,
                f_p1=f_p1,
                f_p2=f_p2,
                true_conc_ts=true_conc_ts,
                seed=seed_val,

            ))

        # run
        print(f'running {len(runs)} runs, simplified from {len(id_vals)} runs')
        if not run:
            print(f'stopping as {run=}')
            return
        result_data = _run_multiprocess(self._power_calc_mp, runs, num_cores=self.ncores,
                                        logging_level=self.log_level)
        result_data = pd.DataFrame(result_data)
        result_data.set_index('idv', inplace=True)

        outdata = result_data.loc[all_use_idv]
        outdata.loc[:, 'idv'] = id_vals
        outdata.set_index('idv', inplace=True) # todo drop the old index

        if outpath is not None:
            outpath.parent.mkdir(parents=True, exist_ok=True)
            outdata.to_hdf(outpath, 'data')
        return outdata



    def power_calc(self,  # todo check options, most nones are gone
                   idv,
                   error: float,
                   mrt_model: str,
                   samp_years: int,
                   samp_per_year: int,
                   implementation_time: int,
                   initial_conc: float,
                   target_conc: float,
                   prev_slope: float,
                   max_conc_lim: float,
                   min_conc_lim: float,
                   mrt: float=0,
                   # options for binary_exponential_piston_flow model
                   mrt_p1: {float, None} = None,
                   frac_p1: {float, None} = None,
                   f_p1: {float, None} = None,
                   f_p2: {float, None} = None,
                   # options for the pass_true_conc_ts model
                   seed: {int, None} = 5585,
                   testnitter=None,
                   **kwargs
                   ):




