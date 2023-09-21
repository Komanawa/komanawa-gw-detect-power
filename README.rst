Groundwater Detection Power Calculator
#######################################

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

# todo make gw lag calcs it's own repo???

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

# todo plots?

* Simple linear reductions between initial and target concentration
* Simple linear reductions between initial and target concentration with a Piston Flow lag +- previous slope  # todo require dependency??
* Simple linear reductions between initial and target concentration with an single or binary exponential piston flow lag # todo requires dependency???
* Pass your own True receptor time series:
    In addition to the aforementioned options you are able to pass a bespoke receptor time series to the function.
    This is done by passing a numpy array to the 'true_conc_ts' kwarg, mrt_model='pass_true_conc'. All other kwargs except
    'idv', and 'error' must be set as None. The sampling rate will be assumed to be the same
    as the passed receptor concentration.  That is a true_conc_ts with 20 values will have the randomly generated error
    added to each value and then be assessed for statistical power.

Options to assess the significance of the noisy receptor time series
--------------------------------------------------------------------------
#todo plots of this

* Linear regression from the first point to the last point
* Linear regression from the max/min point to the last point
* Mann-Kendall test from the first point to the last point (requires kendall_stats optional dependency)
* Mann-Kendall test from the max/min point to the last point (requires kendall_stats optional dependency)
* Pettitt test (requires pyhomogeneity optional dependency)
* MultiPart Mann Kendall/Multipart Seasonal Mann Kendall (requires kendall_stats optional dependency)


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

Python Package Usage
======================

Detailed documentation is available in the docstrings of the functions and classes.
The following is a brief overview of the package.

# todo