"""
created matt_dumont 
on: 25/01/24
"""
import numpy as np


def make_step_test_data(delta, length):
    x1 = np.full(length, 12)
    x2 = x1 + delta
    return x1, x2

def make_linear_test_data(slope, length, delta=0):
    x1 = np.full(length, 12)
    x = np.arange(length)
    x2 = x1 + slope * x + delta
    return x1, x2

def make_bilinar_test_data(slope1, slope2, length, delta1=0, delta2=0):
    x = np.arange(length)
    x0 = np.full(length, 12)
    x1 = x0 + slope1 * x + delta1
    x2 = x0 + slope2 * x + delta2
    return x1, x2


def understand_pairttest():
    for alter in ['two-sided', 'greater', 'less']:
        for scale in [0.1, 1, 10, 100, 1000]:
            base, alt = make_step_test_data(2, 100)
            assert (alt>base).all()
            base = np.repeat(base[np.newaxis], 100, axis=0)
            alt = np.repeat(alt[np.newaxis], 100, axis=0)
            noise_alt = np.random.normal(0, scale, base.shape)
            noise_base = np.random.normal(0, scale, base.shape)
            alt = alt + noise_alt
            base = base + noise_base
            from scipy.stats import ttest_rel
            ttv = ttest_rel(base, alt, axis=1, alternative=alter)
            print(f'{alter=}, {scale=}, {np.mean(ttv.pvalue<=0.05)}')



if __name__ == '__main__':
    understand_pairttest()
    pass