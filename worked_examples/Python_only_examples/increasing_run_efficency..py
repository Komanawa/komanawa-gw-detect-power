"""
created matt_dumont 
on: 16/02/24
"""
#%% md
# Increasing Runtime Efficiency

# The detection power calculations can be slow, especially if you are running a large number of detection power simulations and using a computationally expensive detection method.  This notebook provides some tips on how to increase the runtime efficiency of the detection power calculations.

# Before we get too far let's import the required packages and generate some data to work with.
#%%
import numpy as np
from komanawa.gw_detect_power import AutoDetectionPowerSlope, AutoDetectionPowerCounterFactual
#%% md
## Testing run time (Slope)

# The classes have a method to test the run time of the detection power calculations. This can be useful to determine how long it will take to run a large number of simulations.  An example follows:
#%%
dpc = AutoDetectionPowerSlope(
    significance_mode='n-section-mann-kendall', nsims=1000,
    expect_slope=[1, 0, -1], nparts=3, min_part_size=10, no_trend_alpha=0.50,
    return_true_conc=True, return_noisy_conc_itters=3)

dpc.time_test_power_calc_itter(
    testnitter=5,  # only run 5 change detection iterations instead of 1000 as per dpc.nsims
    # all the following are kwargs for the DetectionPowerCalculator.power_calc function
    idv='true',
    error=0.5,
    mrt_model='binary_exponential_piston_flow',
    samp_years=10,
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
    seed=558
)
#%% md
## Testing run time (Counterfactual)

# The same method is available for the counterfactual detection power calculations.  An example follows:
#%%
dpc_counter = AutoDetectionPowerCounterFactual(significance_mode='paired-t-test', nsims=1000,
                                           p_value=0.05,
                                           min_samples=10,
                                           alternative='alt!=base')

dpc_counter.time_test_power_calc_itter(
    testnitter=5,  # only run 5 change detection iterations instead of 1000 as per dpc.nsims
    # all the following are kwargs for the DetectionPowerCalculator.power_calc function
    idv='true',
    error_base=0.5,
    mrt_model='binary_exponential_piston_flow',
    samp_years=10,
    samp_per_year=10,
    implementation_time_alt=5,
    initial_conc=10,
    target_conc_alt=5,
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
    seed=558
)
#%% md
## Running in muliprocessing mode

# The classes are set up to quickly run multiple detection power estimates in parallel. Essentially you are expected to pre-process the inputs and then pass the runs. Most kwargs can be passed as either an array or as a single value  An example follows:
#%%
dpc = AutoDetectionPowerSlope(
    significance_mode='linear-regression', nsims=1000,
    ncores=3  # set the number of cores to use
)

outdata = dpc.mulitprocess_power_calcs(outpath=None,  # can be saved to a .hdf file if desired
                                       idv_vals=np.array([f'run_{i}' for i in range(6)]),
                                       error_vals=0.5,
                                       samp_years_vals=np.array([5, 5, 5, 10, 10, 10]),
                                       samp_per_year_vals=np.array([1, 4, 12, 1, 4, 12]),
                                       implementation_time_vals=5,
                                       initial_conc_vals=10,
                                       target_conc_vals=7,
                                       prev_slope_vals=0,
                                       max_conc_lim_vals=20,
                                       min_conc_lim_vals=1,
                                       mrt_model_vals='piston_flow',
                                       mrt_vals=0,
                                       seed_vals=535,
                                       )
#%% md
# When running in multiprocessing mode, any errors will be passed by the traceback will be saved to a "python_error" column. This column will be None if there is no error.  This can be used to identify any errors that have occurred.  An example follows:
#%%
print(outdata['python_error'].notna().sum(), 'errors')
print('Errors:\n')
for idv, error in zip(outdata.index[~outdata['python_error'].isna()],
                      outdata['python_error'][~outdata['python_error'].isna()]):
    print(idv, error)
#%% md
## Efficient mode (Slope only)

# Efficient mode is only available in the slope detection power calculations. If implemented the detection power test will be run on the True timeseries data. If the true data does not meet the test (e.g. no reduction) then a power of zero is returned.  Efficient mode can be set by:
#%%
dpc = AutoDetectionPowerSlope(
    significance_mode='linear-regression', nsims=1000,
    ncores=3, efficent_mode=True
)
#%% md

# In addition, there are a number of options to speed up multipart Mann Kendall tests including only evaluating a window, and checking for breakpoints at fewer points (e.g. every other point).  See the documentation for more details.

### Condensed mode

# Condensed mode is available in multiprocessing for either of the autodetection classes.  This takes advantage of the fact that the detection power calculations are often run on similar data and for all intents and purposes an error term of 0.5 mg/l is essentially the same as an error term of 0.52 mg/l. When setting condensed mode the user sets the precision for all of the important float values.  The detection power calculations are run once for each set of rounded inputs and then the results are propagated to each input value.  To set condensed mode, run the following BEFORE running the multiprocessing power calculations:
#%%
dpc = AutoDetectionPowerSlope(
    significance_mode='linear-regression', nsims=1000,
    ncores=3)

dpc.set_condensed_mode(
    target_conc_per=1,  # round the target_conc to 1 decimal place
    initial_conc_per=1,
    error_per=2,  # round the error to 2 decimal places, etc.
    prev_slope_per=2,
    max_conc_lim_per=1,
    min_conc_lim_per=1,
    mrt_per=0,
    mrt_p1_per=2,
    frac_p1_per=2,
    f_p1_per=2,
    f_p2_per=2)
#%% md
## Preprocessing the input data

# If you are working through a large number of simulations then it can be worth preprocessing the input data, saving it to an external file (e.g. an .hdf file) and then reading from that file, so that you can quickly re-run the scenarios if you have made a mistake in your code.