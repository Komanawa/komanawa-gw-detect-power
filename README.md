# komanawa.gw_detect_power


## Installation

### Install the package from PyPI
todo

### Install from the GitHub repository / bleeding edge
pip install git+https://github.com/Komanawa-Solutions-Ltd/kslcore.git

## Methodology

The statistical power calculation is fairly straight forward.  the steps are:

#. Create a 'True' receptor time series (e.g. the concentration at the receptor/well if there was no lag)
#. Generate noise based on the user passed standard deviation ('error' kwarg).  A normal distribution is used.
#. Add the noise to the true receptor time series
#. Assess the significance of the noisy receptor time series.
#. If the change is statistically significant (p< minimum p value) and in the expected direction, 
then the detection power is 1.0, otherwise it is 0.0
#. Repeat steps 2-5 for the number of iterations specified by the user ('n_iterations' kwarg) the statistical power
is then reported as the mean of the detection power over the number of iterations (as a percentage).



### Options to create the 'True' receptor time series

#### Simple linear reductions between initial and target concentration
#todo plots of this

#### Simple linear reductions between initial and target concentration with a Piston Flow lag +- previous slope
#todo plots of this

#### Simple linear reductions between initial and target concentration with an single or binary exponential piston flow lag
#todo plots of this

#### Pass your own True receptor time series
In addition to the aforementioned options you are able to pass a bespoke receptor time series to the function.  
This is done by passing a numpy array to the 'true_conc_ts' kwarg, mrt_model='pass_true_conc'. All other kwargs except
'idv', and 'error' must be set as None. The sampling rate will be assumed to be the same
as the passed receptor concentration.  That is a true_conc_ts with 20 values will have the randomly generated error
added to each value and then be assessed for statistical power.
#todo plots of this

### Options to assess the significance of the noisy receptor time series
#todo plots of this

#### Linear regression
Fit a Linear regression to the noisy receptor. 

#### Mann-Kendall
Not yet implemented

## Development
Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.