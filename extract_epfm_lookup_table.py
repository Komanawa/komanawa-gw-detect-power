"""
created matt_dumont 
on: 28/08/23

This script extracts the data from the epfm.npz file and constructs the epfm tables This was done as the epfm tables
take up 600mb of data, all of which can be reconstructed from the data file and this script with minimal computational
effort.  unmodified this script will save the tables to the user's Downloads/EPFM_lag_tables directory
"""
from pathlib import Path
from lookup_table_inits import implementation_times, per_reductions, base_vars, base_outkeys, other_outkeys, \
    epfm_mrts, epfm_fracs, lookup_dir
import itertools
import pandas as pd
import numpy as np


def construct_epfm_table_from_data(use_outdir, datafile=lookup_dir.joinpath('epfm.npz')):
    """
    construct the epfm excel tables from the data file
    :param use_outdir: directory to save the tables to
    :param datafile: the data file to use (should not need to change)
    :return:
    """
    use_outdir.mkdir(exist_ok=True)
    for imp_time, per_red in itertools.product(implementation_times, per_reductions):
        print(f'making table for:  {imp_time=}, {per_red=}')
        indata = pd.DataFrame(columns=['samp_t', 'nsamp', 'n_noise', 'start', 'mrt', 'f1'],
                              data=itertools.product(*(base_vars[2:] + [epfm_mrts, epfm_fracs]))
                              )
        indata['imp_t'] = imp_time
        indata['red'] = per_red
        indata['target'] = indata.start - indata.start * indata.red
        print(len(indata), 'rows', np.array([1.3]).nbytes * len(indata) * 1e-6 * 7, 'mb')
        outdata = indata.rename(columns={
            'imp_t': 'implementation_time',
            'red': 'percent_reduction',
            'samp_t': 'samp_years',
            'nsamp': 'samp_per_year',
            'n_noise': 'error',
            'start': 'initial_conc',
            'target': 'target_conc',
        })
        outdata['power'] = np.load(datafile)[f'imp_{imp_time}_perred_{int(per_red * 100)}']
        outdata['percent_reduction'] = (outdata.initial_conc - outdata.target_conc) / outdata.initial_conc * 100
        outdata = outdata[base_outkeys + other_outkeys[:1]]
        outdata.to_excel(use_outdir.joinpath(f'imp_{imp_time}_perred_{int(per_red * 100)}.xlsx'))


if __name__ == '__main__':
    construct_epfm_table_from_data(use_outdir=Path().home().joinpath('Downloads', 'EPFM_lag_tables'))
