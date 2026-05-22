"""
created matt_dumont 
on: 7/03/24
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path




if __name__ == '__main__':
    slope = 0.2
    intercept = 3
    nyears = 10
    annual = np.linspace(0, nyears, 1*nyears)
    monthly = np.linspace(0, nyears, 12*nyears)
    quarterly = np.linspace(0, nyears, 4*nyears)

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))
    np.random.seed(65483)
    seeds = list(np.random.randint(0, 165400, 100))

    for ax, x, xname in zip(axs, [annual, quarterly, monthly], ['annual', 'quarterly', 'monthly']):
        np.random.seed(seeds.pop())
        y = slope * x + intercept + np.random.normal(0, 1, len(x))
        y0 = slope * x + intercept
        ax.plot(x, y0, 'k--', label='true')
        ax.scatter(x, y, label='observed')
        ax.set_title(f'{xname} data')
    fig.supxlabel('years')
    fig.supylabel('NO$_{3}$-N (mg/L)')
    fig.tight_layout()
    fig.savefig(Path(__file__).parents[1].joinpath('figures','frequency_example.png'))
    plt.show()

