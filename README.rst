Kо̄manawa Groundwater Detection Power Calculator
##################################################

A python package to support calculating the likelihood of detecting a significant change in concentration at a monitoring point (e.g. groundwater well, stream, lake) for a future scenario given the measured noise.
Too often management decisions (e.g., plans to change land use to reduce nitrate concentrations in streams and groundwater) are made without considering whether the monitoring program in place can detect the modelled changes in concentration.
Measurement noise (or unexplained variability) and groundwater lag can obfuscate impacts and leave managers with insufficient data collection to assess the effectiveness of their management actions.
This package supports the calculation of detection power for a range of statistical tests (e.g. Mann-Kendall, Pettitt, etc.) and can help managers and scientists to design monitoring programs that are fit for purpose and can detect the changes in concentration that are expected from management actions.


This package is designed to be used within a python environment and does not have a command line interface.

Full documentation including API Documentation is available at: https://komanawa-solutions-ltd.github.io/komanawa-gw-detect-power/

API Documentation link: https://komanawa.github.io/komanawa-gw-detect-power/autoapi/komanawa/gw_detect_power/index.html

Installation
==================
The easiest way to install is to use pip and install directly from github.  This will ensure that
all dependencies are installed.

Install from PyPI
----------------------

.. code-block:: bash

    pip install komanawa-gw-detect-power

Install from Github
----------------------

.. code-block:: bash

    conda create -c conda-forge --name gw_detect  python=3.11 pandas=2.0.3 numpy=1.25.2 matplotlib=3.7.2 scipy=1.11.2 pytables=3.8.0 psutil=5.9.5
    conda activate gw_detect

    pip install pyhomogeneity
    pip install git+https://github.com/Komanawa/komanawa-kendall-stats.git
    pip install git+https://github.com/Komanawa/komanawa-gw-age-tools
    pip install git+https://github.com/Komanawa/komanawa-gw-detect-power


Dependencies
==================

* pandas>=2.0.3
* numpy>=1.25.2
* scipy>=1.11.2
* tables>=3.8.0
* psutil>=5.9.5

Optional Dependencies
----------------------

* pyhomogeneity (for the Pettitt test)
* komanawa-kendall-stats (for the Mann Kendall / MultiPart Mann Kendall / Multipart Seasonal Mann Kendall)
* komanawa-gw-age-tools (for the binary piston flow lag)


Quickstart
==================

`A quickstart Jupyter notebook example <worked_examples/quickstart.ipynb>`_ is available in the worked examples directory.


Further Worked Examples
============================

A set of worked examples (both Jupyter Notebooks and pure python scripts) are available in the `worked_examples
<worked_examples>`_ directory of the repository.  These examples are not included in the package installation but can be downloaded/cloned from the repository.
An overview of these worked examples is in the readme file of the linked directory (please scroll to the bottom of the github page to see the readme file).
