"""
created matt_dumont 
on: 25/09/23
"""
import matplotlib.pyplot as plt
import pandas as pd
from pyhomogeneity import pettitt_test
from komanawa.kendall_stats import make_example_data
import numpy as np

y_test = []
for noise in [0, 2, 5]:
    # increasing
    x_inc, y_inc = make_example_data.make_multipart_sharp_change_data(make_example_data.multipart_sharp_slopes[0],
                                                                      noise=noise,
                                                                      na_data=False, unsort=False)

    idx = np.arange(0, len(x_inc), 5)
    x_inc = x_inc[idx]
    y_inc = y_inc[idx]
    y_test.append(y_inc)

if __name__ == '__main__':
    from pathlib import Path

    fig, axs = plt.subplots(nrows=3, sharey=True, sharex=True, figsize=(10, 10))
    for noise, y0, ax in zip([0, 2, 5], y_test, axs):
        print('noise', noise)
        sims = [0, 2, 20, 200, 2000, 20000]
        outdata = pd.DataFrame(columns=[f'p_{s}' for s in sims], dtype=float)
        for i in range(100):
            print(i, 'of 100')
            for sim in sims:
                h, cp, p, U, mu = pettitt_test(y_test[0], alpha=0.05, sim=sim)
                outdata.loc[i, f'p_{sim}'] = p
        outdata.to_hdf(Path(__file__).parent.joinpath('pettitt_test.hdf'), key=f'noise_{noise}')
        ax.boxplot([outdata[c] for c in outdata.columns], labels=outdata.columns)
        ax.set_title(f'noise: {noise}')
    fig.tight_layout()
    fig.savefig(Path(__file__).parents[1].joinpath('figures', 'pettitt_test_nitter.png'))
