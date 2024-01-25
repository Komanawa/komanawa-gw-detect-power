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
from gw_detect_power import AutoDetectionPowerSlope
from gw_detect_power.lookup_table_inits import implementation_times, base_vars, base_outkeys, \
    other_outkeys, pf_mrts, lookup_dir


def _save_compressed_file(outdata, outpath, ziplib=None):
    # save and compress
    if ziplib is None:
        outdata.to_excel(outpath)
    else:
        with tempfile.TemporaryDirectory() as tdir:
            tdir = Path(tdir)
            tpath = tdir.joinpath(outpath.name)
            outdata.to_excel(tpath)

            if ziplib == '7z':
                with py7zr.SevenZipFile(outpath.with_suffix('.7z'), 'w') as archive:
                    archive.write(tpath, arcname=outpath.name)
            else:
                with zipfile.ZipFile(outpath.with_suffix('.zip'), mode="w", compresslevel=9) as zf:
                    zf.write(tpath, arcname=outpath.name)


def no_lag_table(test_size=False):
    """
    generate the no lag table.  This table is hosted in the github repo
    :param test_size: bool if true just write dummy data to assess the table size
    :return:
    """
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
        outdata['power'] = np.random.random(len(outdata)) * 100

    else:
        dpc = AutoDetectionPowerSlope(min_samples=5)
        outdata = dpc.mulitprocess_power_calcs(
            outpath=None,
            idv_vals=indata.index.values,
            error_vals=indata.n_noise.values,
            samp_years_vals=indata.samp_t.values,
            samp_per_year_vals=indata.nsamp.values,
            implementation_time_vals=indata.imp_t.values,
            initial_conc_vals=indata.start.values,
            target_conc_vals=indata.target.values,
            previous_slope_vals=0,
            max_conc_vals=25,
            min_conc_vals=25,
            mrt_model_vals='piston_flow',
            mrt_vals=0.0,
            mrt_p1_vals=None,
            frac_p1_vals=None,
            f_p1_vals=None,
            f_p2_vals=None,
            seed=5585,
            run=run_model,
        )
        if outdata is None:
            return  # for testing the multiprocess setup
        # add percent reduction
        outdata['percent_reduction'] = (outdata.initial_conc - outdata.target_conc) / outdata.initial_conc * 100
    outdata = outdata[base_outkeys]
    _save_compressed_file(outdata, lookup_dir.joinpath('no_lag_table.xlsx'))


def piston_flow_lag_table(test_size=False):
    """
    generate the piston flow lag table.  This table is hosted in the github repo
    :param test_size: bool if true just write dummy data to assess the table size
    :return:
    """
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
            outdata['power'] = np.random.random(len(outdata)) * 100

        else:
            dpc = AutoDetectionPowerSlope(min_samples=5)
            outdata = dpc.mulitprocess_power_calcs(
                outpath=None,
                idv_vals=indata.index.values,
                error_vals=indata.n_noise.values,
                samp_years_vals=indata.samp_t.values,
                samp_per_year_vals=indata.nsamp.values,
                implementation_time_vals=indata.imp_t.values,
                initial_conc_vals=indata.start.values,
                target_conc_vals=indata.target.values,
                previous_slope_vals=0,
                max_conc_vals=indata.start.values,
                min_conc_vals=0,
                mrt_model_vals='piston_flow',
                mrt_vals=indata.mrt.values,
                mrt_p1_vals=None,
                frac_p1_vals=None,
                f_p1_vals=None,
                f_p2_vals=None,
                seed=5585,
                run=run_model,
            )
            if outdata is None:
                continue  # for testing the multiprocess setup
            # add percent reduction
            outdata['percent_reduction'] = (outdata.initial_conc - outdata.target_conc) / outdata.initial_conc * 100
        outdata = outdata[base_outkeys + other_outkeys[:1]]
        _save_compressed_file(outdata, lookup_dir.joinpath(f'piston_flow_lag_table_imp_{imp_time}.xlsx'))


if __name__ == '__main__': # todo re-run after tests
    run_model = True  # a flag it True will run the model if false will just setup and check inputs # todo flip flag
    test_size = False
    # epfm_lag_table(test_size) # just too big
    no_lag_table(test_size)
    piston_flow_lag_table(test_size)
