Groundwater Detection Power Calculator
#######################################

This package is designed to calculate the statistical power of detecting a change in groundwater/surface concentration
depending on sampling duration, sampling frequency, 'true' receptor concentration and the noise in the receptor.
there is also support for understanding statisitical power in the context of groundwater travel times (e.g. lag)
and groundwater temporal dispersion (e.g. mixing of different aged waters via a binary piston flow lag model).

In this repo we have a couple key definitions:

* **Receptor**: The receptor is the location where the concentration is measured.  This is typically a groundwater well, stream or lake.
* **Source**: The source is the location where the concentration is changed.  This is typically a point source (e.g. a wastewater treatment plant) or a non-point source (e.g. a catchment/groundwater source area).
* **Noise**: here by noise we include the variation in the concentration at the receptor. This includes true sampling noise, but also includes any other variation in the concentration at the receptor that cannot be identified or corrected for (e.g. from weather events etc.). Typically the noise will be estimated as the standard deviation of the receptor concentration time series (assuming no trend), or the standard deviation of the residuals from a model (e.g. linear regression) of the receptor concentration time series.
* **True Receptor Concentration**: The true receptor concentration is the concentration at the receptor if there was no noise.

Look up tables for statistical power
=====================================

We have included a number of lookup table to support less computationally savvy users. These look up tables are here to give estimates of the detection power.

These tables have been run for a no lag scenario with:

* 5, 10, 20, 30, 50, 75, & 100 year implementation times
* 5, 10, 15, 20, 25, 30 & 50 year monitoring durations
* 1, 4, 12, 26 & 52 samples/year sampling frequencies
* 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2, 2.5, 3, 4, 5, & 7.5 mg/l N-NO3 Noise levels
* 4, 5.6, 6, 7, 8, 9, 10, 11.3, 15 & 20 mg/l starting N-NO3 concentrations
* 5, 10, 15, 20, 25, 30, 40, 50 & 75% reductions in N-NO3 concentrations over the implementation period

The piston flow lag includes mean residence times of 1, 3, 5, 7, 10, 12, 15 years.

To use these tables:

`README.rst <README.rst>`_

1. Locate and download the right table (decision tree):
    1. `if you are not interested in the effect of lag <lookup_tables/no_lag_table.xlsx>`_
    2. if you are interested in the effect of lag, then download the table for the appropriate implementation time:
        * `5 year implementation time <lookup_tables/piston_flow_lag_table_imp_5.xlsx>`_
        * `10 year implementation time <lookup_tables/piston_flow_lag_table_imp_10.xlsx>`_
        * `20 year implementation time <lookup_tables/piston_flow_lag_table_imp_20.xlsx>`_
        * `30 year implementation time <lookup_tables/piston_flow_lag_table_imp_30.xlsx>`_
        * `50 year implementation time <lookup_tables/piston_flow_lag_table_imp_50.xlsx>`_
        * `75 year implementation time <lookup_tables/piston_flow_lag_table_imp_75.xlsx>`_
        * `100 year implementation time <lookup_tables/piston_flow_lag_table_imp_100.xlsx>`_
2. open the table in a spreadsheet program (e.g. excel)
3. Locate the row that corresponds to the closest:
    * sampling duration (samp_years)
    * sampling frequency (samp_per_year)
    * implementation_time
    * initial_conc
    * target_conc
    * percent_reduction
    * mean residence time (mrt (if applicable))
4. The provided power is the percent chance of detecting the change in concentration


Dependencies
==================

* pandas>=2.0.3
* numpy>=1.25.2
* scipy>=1.11.2
* tables>=3.8.0
* psutil>=5.9.5

Optional Dependencies
----------------------

* pyhomogeneity (for the Pettitt test)
* kendall_stats (for the Mann Kendall / MultiPart Mann Kendall / Multipart Seasonal Mann Kendall)
* gw_age_tools (for the binary piston flow lag)


Installation
==================

This package is currently held as a simple github repo,
but the intention is to make it available on PyPI in the future, It also sources other repos that are only hosted on
github.  Therefore, the easiest way to install is to use pip and install directly from github.  This will ensure that
all dependencies are installed.

Install from Github
----------------------

.. code-block:: bash

    conda create -c conda-forge --name gw_detect  python=3.11 pandas=2.0.3 numpy=1.25.2 matplotlib=3.7.2 scipy=1.11.2 pytables=3.8.0 psutil=5.9.5
    conda activate gw_detect

    pip install pyhomogeneity
    pip install git+https://github.com/Komanawa-Solutions-Ltd/kendall_multipart_kendall.git
    pip install git+https://github.com/Komanawa-Solutions-Ltd/gw_age_tools
    pip install git+https://github.com/Komanawa-Solutions-Ltd/gw_detect_power


Methodology
================

The statistical power calculation is fairly straight forward.  the steps are:

1. Create a 'True' receptor time series (e.g. the concentration at the receptor/well if there was no lag)
2. Generate noise based on the user passed standard deviation ('error' kwarg).  A normal distribution is used.
3. Add the noise to the true receptor time series
4. Assess the significance of the noisy receptor time series.
5. If the change is statistically significant (p< minimum p value) and in the expected direction,
then the detection power is 1.0, otherwise it is 0.0
6. Repeat steps 2-5 for the number of iterations specified by the user ('n_iterations' kwarg) the statistical power
is then reported as the mean of the detection power over the number of iterations (as a percentage).


Options to create the 'True' receptor time series
-------------------------------------------------------

We have implemented four different options to create the 'True' receptor time series.  These are:

Simple linear reductions between initial and target concentration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# todo + plots

Simple linear reductions concentration with a Piston Flow lag with a positive, negative, or no previous slope
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# todo + plots

Simple linear reductions with an single or binary exponential piston flow lag with a positive or no previous slope
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# todo + plots

Pass your own True receptor time series
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

the user is able to pass a bespoke receptor time series to the function. This is done by passing a numpy array to the 'true_conc_ts' kwarg, mrt_model='pass_true_conc'. All other kwargs except 'idv', and 'error' must be set as None. The sampling rate will be assumed to be the same as the passed receptor concentration.  That is a true_conc_ts with 20 values will have the randomly generated error added to each value and then be assessed for statistical power.

# todo plot of the data to explain this.

Options to assess the significance of the noisy receptor time series
--------------------------------------------------------------------------

* Linear regression from the first point to the last point (detection is a significant slope in the expected direction)
* Linear regression from the [max|min] point to the last point (detection is a significant slope in the expected direction)
* Mann-Kendall test from the first point to the last point (requires kendall_stats optional dependency) (detection is a significant slope in the expected direction)
* Mann-Kendall test from the [max|min] point to the last point (requires kendall_stats optional dependency) (detection is a significant slope in the expected direction)
* MultiPart Mann Kendall/Multipart Seasonal Mann Kendall (requires kendall_stats optional dependency) here if the process identifies any significant breakpoints (within the alpha, no_trend_alpha, and expected slopes) the test records detection. See `kendall_stats <https://github.com/Komanawa-Solutions-Ltd/kendall_multipart_kendall#multipartkendall>`_ for more details
* Pettitt test (requires pyhomogeneity optional dependency)  # todo more details


Python Package Usage
======================

Detailed documentation is available in the docstrings of the functions and classes.
The following is a brief overview of the package.

# todo

Detection power class initialisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# todo signifiacne mode, nsims, other kwargs

truets_from_piston_flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^
# todo + plot of example

truets_from_binary_exp_piston_flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# todo + plot of example

power_calc function
^^^^^^^^^^^^^^^^^^^^^^

# todo example and output?

mulitprocess_power_calcs
^^^^^^^^^^^^^^^^^^^^^^^^^^

# todo example
