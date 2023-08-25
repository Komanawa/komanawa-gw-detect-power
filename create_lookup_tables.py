"""
created matt_dumont 
on: 25/08/23
"""
import itertools
import tempfile
import zipfile
import numpy as np
from pathlib import Path
import pandas as pd
import py7zr

from change_detection_v2 import DetectionPowerCalculator

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

outdir = Path(__file__).parent.joinpath('lookup_tables')
outdir = Path(__file__).home().joinpath('unbacked', 'lookup_tables2') # todo DADB
outdir.mkdir(exist_ok=True)

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


def _save_compressed_file(outdata, outpath, ziplib=None):
    # save and compress
    if ziplib is None:
        outdata.to_excel(outpath)
    else:
        with tempfile.TemporaryDirectory() as tdir:
            tdir = Path(tdir)
            tpath = tdir.joinpath(outpath.name)
            # todo write a excel function in a front sheet
            outdata.to_excel(tpath)

            if ziplib == '7z':
                with py7zr.SevenZipFile(outpath.with_suffix('.7z'), 'w') as archive:
                    archive.write(tpath, arcname=outpath.name)
            else:
                with zipfile.ZipFile(outpath.with_suffix('.zip'), mode="w",compresslevel=9) as zf:
                    zf.write(tpath, arcname=outpath.name)


def no_lag_table(test_size=False):
    indata = pd.DataFrame(columns=['imp_t', 'red', 'samp_t', 'nsamp', 'n_noise', 'start'],
                          data=itertools.product(*base_vars))
    indata['target'] = indata.start - indata.start * indata.red
    print(len(indata), 'rows', np.array([1.3]).nbytes * len(indata) * 1e-6 * 7, 'mb')
    if test_size:
        print('saving dummy file to test size')
        outdata = indata.rename(columns={
            'imp_t': 'implementation_time',
            'red': 'percent_reduction',
            'samp_t': 'samp_years',
            'nsamp': 'samp_per_year',
            'n_noise': 'error',
            'start': 'initial_conc',
            'target': 'target_conc',
        })
        outdata['power'] = np.random.random(len(outdata))

    else:
        dpc = DetectionPowerCalculator()
        outdata = dpc.mulitprocess_power_calcs(
            outpath=None,
            id_vals=indata.index.values,
            error_vals=indata.n_noise.values,
            samp_years_vals=indata.samp_t.values,
            samp_per_year_vals=indata.nsamp.values,
            implementation_time_vals=indata.imp_t.values,
            initial_conc_vals=indata.start.values,
            target_conc_vals=indata.target.values,
            previous_slope_vals=0,
            max_conc_vals=None,
            min_conc_vals=None,
            mrt_model_vals='piston_flow',
            mrt_vals=0.0,
            mrt_p1_vals=None,
            frac_p1_vals=None,
            f_p1_vals=None,
            f_p2_vals=None,
            seed=5585,
            run=False,
        )
        # todo add percent reduction
    outdata = outdata[base_outkeys]
    _save_compressed_file(outdata, outdir.joinpath('no_lag_table.xlsx'))



def piston_flow_lag_table(test_size=False):
    for imp_time in implementation_times:
        indata = pd.DataFrame(columns=['red', 'samp_t', 'nsamp', 'n_noise', 'start', 'mrt'],
                              data=itertools.product(*(base_vars[1:] + [pf_mrts]))
                              )
        indata['target'] = indata.start - indata.start * indata.red
        indata['imp_t'] = imp_time
        print(len(indata), 'rows', np.array([1.3]).nbytes * len(indata) * 1e-6 * 7, 'mb')
        if test_size:
            outdata = indata.rename(columns={
                'imp_t': 'implementation_time',
                'red': 'percent_reduction',
                'samp_t': 'samp_years',
                'nsamp': 'samp_per_year',
                'n_noise': 'error',
                'start': 'initial_conc',
                'target': 'target_conc',
            })
            outdata['power'] = np.random.random(len(outdata))

        else:
            raise NotImplementedError
            # todo add percent reduction
        outdata = outdata[base_outkeys + other_outkeys[:1]]
        _save_compressed_file(outdata, outdir.joinpath(f'piston_flow_lag_table_imp_{imp_time}.xlsx'))


def epfm_lag_table(test_size=False): # todo tooo big, make this an option for a local user to run if they want.
    for imp_time, per_red in itertools.product(implementation_times, per_reductions):
        indata = pd.DataFrame(columns=['samp_t', 'nsamp', 'n_noise', 'start', 'mrt', 'f1'],
                              data=itertools.product(*(base_vars[2:] + [epfm_mrts, epfm_fracs]))
                              )
        indata['imp_t'] = imp_time
        indata['red'] = per_red
        indata['target'] = indata.start - indata.start * indata.red
        print(len(indata), 'rows', np.array([1.3]).nbytes * len(indata) * 1e-6 * 7, 'mb')
        if test_size:
            outdata = indata.rename(columns={
                'imp_t': 'implementation_time',
                'red': 'percent_reduction',
                'samp_t': 'samp_years',
                'nsamp': 'samp_per_year',
                'n_noise': 'error',
                'start': 'initial_conc',
                'target': 'target_conc',
            })
            outdata['power'] = np.random.random(len(outdata))

        else:
            raise NotImplementedError
            # todo add percent reduction
        outdata = outdata[base_outkeys + other_outkeys[:1]]
        _save_compressed_file(outdata, outdir.joinpath(
            f'EPFM_lag_table_imp_{imp_time}_perred_{int(per_red * 100)}.xlsx'))


if __name__ == '__main__':
    test_size = True
    no_lag_table(test_size)
    epfm_lag_table(test_size)
    piston_flow_lag_table(test_size)
