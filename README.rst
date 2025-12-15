Kо̄manawa Groundwater Detection Power Calculator
##################################################

Full documentation including API is available at: https://komanawa-solutions-ltd.github.io/komanawa-gw-detect-power/

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

.. todo add a quickstart example... once get details from reviewer.

Further Worked Examples
============================

A set of worked examples (both Jupyter Notebooks and pure python scripts) are available in the `worked_examples
<worked_examples>`_ directory of the repository.  These examples are not included in the package installation but can be downloaded/cloned from the repository.
.. todo check link