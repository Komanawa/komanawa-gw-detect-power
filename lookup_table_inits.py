"""
created matt_dumont 
on: 28/08/23
"""
from pathlib import Path
import numpy as np

implementation_times = [5, 10, 20, 30, 50, 75, 100]
sampling_times = [5, 10, 15, 20, 25, 30, 50]
nsamps_per_year = [1, 4, 12, 26, 52]
n_noises = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2, 2.5, 3, 4,
            5, 7.5]
start_concs = [4, 5.6, 6, 7, 8, 9, 10, 11.3, 15, 20]
per_reductions = (np.array([5, 10, 15, 20, 25, 30, 40, 50, 75]) / 100).round(2)

# lag options
pf_mrts = [1, 3, 5, 7, 10, 12, 15]
epfm_mrts = [1, 2, 5, 7, 10, 15, 20, 30, 40, 50, 75, 100]
epfm_fracs = [0.1, 0.25, 0.5, 0.75, 0.9]

lookup_dir = Path(__file__).parent.joinpath('lookup_tables')
lookup_dir.mkdir(exist_ok=True)

base_vars = [implementation_times, per_reductions, sampling_times, nsamps_per_year, n_noises, start_concs, ]

base_outkeys = [
    'power',
    'error',
    'samp_years',
    'samp_per_year',
    'implementation_time',
    'initial_conc',
    'target_conc',
    'percent_reduction',
]

other_outkeys = [
    'mrt',
    'frac_p1',
]