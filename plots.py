"""
created matt_dumont 
on: 14/08/23
"""

from detection_code.exponential_piston_flow import *
from pathlib import Path


def explainer_plot():
    plot = True
    input_series = pd.Series(index=[-35, 0., 50, 100,],
                             data=[1, 1, 18, 2.4,])
    mrt = 10
    mrt_p1 = 2.5
    frac_p1 = 0.2
    f_p1 = 0.8
    f_p2 = 0.75
    data = predict_future_conc_bepm(once_and_future_source_conc=input_series,
                                    predict_start=0,
                                    predict_stop=100,
                                    mrt_p1=mrt_p1, frac_p1=frac_p1, f_p1=f_p1, f_p2=f_p2, mrt=mrt, mrt_p2=None,
                                    fill_value=1,
                                    fill_threshold=0.05,
                                    pred_step=0.5)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(input_series.index, input_series.values, label='synthetic source concentration')
    ax.plot(data.index, data.values, label='predicted concentration')
    np.random.seed(6568)
    use_idx = np.arange(0, 50, 5)
    scat_data = data.loc[use_idx] + np.random.normal(0, 1.5, len(data.loc[use_idx]))
    ax.scatter(scat_data.index, scat_data.values, marker='o', label='observed concentration')
    ax.annotate('likely\ninflection point', (0,1), (-30,5), arrowprops={'arrowstyle': '->'})
    ax.annotate('reductions start', (50,18), (60,17), arrowprops={'arrowstyle': '->'})

    ax.legend()
    ax.set_xlabel('time (years)')
    ax.set_ylabel('concentration')

    fig.tight_layout()
    fig.savefig(Path().home().joinpath('Downloads','explainer_plot.png'))
    plt.show()


if __name__ == '__main__':
    explainer_plot()
