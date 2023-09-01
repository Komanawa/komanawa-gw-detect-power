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
from lookup_table_inits import implementation_times, per_reductions, base_vars, base_outkeys, other_outkeys, pf_mrts, \
    epfm_mrts, epfm_fracs, lookup_dir


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
            run=run_model,
        )
        if outdata is None:
            return  # for testing the multiprocess setup
        # add percent reduction
        outdata['percent_reduction'] = (outdata.initial_conc - outdata.target_conc) / outdata.inital_conc * 100
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
            outdata['percent_reduction'] = (outdata.initial_conc - outdata.target_conc) / outdata.inital_conc * 100
        outdata = outdata[base_outkeys + other_outkeys[:1]]
        _save_compressed_file(outdata, lookup_dir.joinpath(f'piston_flow_lag_table_imp_{imp_time}.xlsx'))


def epfm_lag_table(test_size=False):
    """
    generate the epfm lag table.  These tables are too large to host in the github repo. The function here is provided
    so that an experienced users can generate the tables if they want and then host them locally.
    :param epfm_outdir:
    :param test_size:
    :return:
    """
    all_outdata = {}
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
            outdata['power'] = np.random.random(len(outdata)) * 100

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
                max_conc_vals=indata.start.values,
                min_conc_vals=0,
                mrt_model_vals='binary_exponential_piston_flow',
                mrt_vals=indata.mrt.values,
                mrt_p1_vals=indata.mrt.values,
                frac_p1_vals=1,
                f_p1_vals=indata.f1.values,
                f_p2_vals=0.8,  # dummy value
                seed=5585,
                run=run_model,
            )
            if outdata is None:
                continue  # for testing the multiprocess setup
            # add percent reduction
            outdata['percent_reduction'] = (outdata.initial_conc - outdata.target_conc) / outdata.inital_conc * 100
        save_data = (outdata['power'].values).astype(np.uint8)
        all_outdata[f'imp_{imp_time}_perred_{int(per_red * 100)}'] = save_data
    outpath = lookup_dir.joinpath('epfm.npz')
    np.savez_compressed(outpath, **all_outdata)


if __name__ == '__main__':
    run_model = True  # a flag it True will run the model if false will just setup and check inputs
    test_size = False
    epfm_lag_table(test_size)
    no_lag_table(test_size)
    piston_flow_lag_table(test_size)
