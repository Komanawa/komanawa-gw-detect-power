gw_detect_power worked examples
==================================

This repository contains worked examples for the `komanawa-gw-detect-power
<https://komanawa-solutions-ltd.github.io/komanawa-gw-detect-power>`_ package.  These examples were developed using the `Jupyter Notebook <https://jupyter.org/>`_ environment and funded by the New Zealand Our Land and Water National Science Challenge.

.. note::

    The examples provided require version v.2.0.0 or later of komanawa-gw-detect-power

Installation
==============

These examples are meant to be viewed on github as well as being run locally using the Jupyter Notebook environment. Please see the `Jupyter Notebook documentation <https://jupyter.org/install>`_ for installation instructions.

Python Environment
=====================

The python environment for these examples is identical for the full installation of `komanawa-gw-detect-power <https://komanawa-solutions-ltd.github.io/komanawa-gw-detect-power/#installation>`_.

Conda/pip installation
------------------------

.. code-block:: bash

    # create a new environment
    conda create --name gwdetect python
    conda activate gwdetect
    # install the required packages
    pip install komanawa-gw-detect-power
    pip install notebook


Overview
==========

The goal of these examples is to demonstrate how to calculate the detection power of monitoring points and is it applicable to both groundwater and surface water. The `Our Land and Water National Science Challenge: Monitoring Freshwater Improvement Actions Project <https://www.monitoringfreshwater.co.nz>`_ has signficantly more discussion on the topic of monitoring design and the use of detection power.  For more information we recommend, `Water Quality Monitoring for Management of Diffuse Nitrate Pollution <https://komanawa-solutions-ltd.github.io/komanawa-gw-detect-power/_static/Water_quality_monitoring_for_management_of_diffuse_nitrate_pollution_Final.pdf>`_. This document provides guidance on the design of water quality monitoring programs for the management of diffuse nitrate pollution.  It includes a section on statistical power and the use of the detection power calculator as well as other factors that should be considered when designing a water quality monitoring program.



Definitions
-------------

In this repo we have a couple key definitions:

* **Receptor**: The receptor is the location where the concentration is measured.  This is typically a groundwater well, stream or lake.
* **Source**: The source is the location where the concentration is changed.  This is typically a point source (e.g. a wastewater treatment plant) or a non-point source (e.g. a catchment/groundwater source area).
* **Noise**: here by noise we include the variation in the concentration at the receptor. This includes true sampling noise, but also includes any other variation in the concentration at the receptor that cannot be identified or corrected for (e.g. from weather events etc.). Typically the noise will be estimated as the standard deviation of the receptor concentration time series (assuming no trend), or the standard deviation of the residuals from a model (e.g. linear regression) of the receptor concentration time series.
* **True Receptor Concentration**: The true receptor concentration is the concentration at the receptor if there was no noise.


Basic Detection Methodology
------------------------------

#. Estimate noise.
#. Create a True receptor time series.
#. Resample the True receptor time series to your sampling frequncy and duration
#. Run Detection power Calculator (many times e.g. 1000, 10000)
    #. Create a noise realisation and add it to the True time series
    #. Run your statisitcal test
#. Detection power is the number of statistical tests with p<threshold / number of tests.

Jupyter Notebook Index
------------------------

The following Jupyter Notebooks are provided in this repository (we recommend this workflow):

#. `Understanding_detection_power <./Understanding_detection_power.ipynb>`_: a brief introduction to detection power and how it can be used to design monitoring programs.
#. `Let's talk noise <./lets_talk_noise.ipynb>`_: a brief introduction to the concept of noise and how it can be estimated.
#. `Generating_true_ts_data <./generating_true_ts_data.ipynb>`_: a brief introduction to generating time series data for the purposes of testing the detection power calculator.
#. `Slope_detection_power <./slope_detection_power.ipynb>`_: a worked example of how to calculate the detection power for a linear trend in a time series.
#. `Counterfactual_detection_power <./counterfactual_detection_power.ipynb>`_: a worked example of how to calculate the detection power for a counterfactual trend in a time series.
#. `Whakauru_Stream_worked_example <./Whakauru_Stream_worked_example.ipynb>`_: a full worked example of evaluating the detection power of the Whakauru Stream.
#. `Selwyn_well_I36_0477_worked_example <./Selwyn_well_I36_0477_worked_example.ipynb>`_: a full worked example of evaluating the detection power of the Selwyn well I36/0477.
#. `Counterfactual_from_models <./counterfactual_from_models.ipynb>`_: a full worked example of evaluating the detection power for a site for synthetic modelled scenario output.
#. `Increasing_run_efficiency <./increasing_run_efficency.ipynb>`_: a worked example of how to increase the run efficency of the detection power calculator.

Video Tutorials
----------------

As part of the Our Land and Water National Science Challenge, we held a workshop to accompany these worked examples. Videos can be found on the `Kо̄manawa Solutions Youtube Channel <https://www.youtube.com/channel/UCS-oF82WbkOXLdZVumzy56g>`_ and are listed below.  To access the full webinar recording please see: `Kо̄manawa Groundwater Detection Power Calculator <https://komanawa-solutions-ltd.github.io/komanawa-gw-detect-power/>`_.

#. `Which tool to use and when <https://youtu.be/s4Na3z6Gyh0>`_
#. `Making "true" receptor concentration time series <https://youtu.be/yvlqJMQqbcU>`_
#. `Whakauru Stream Worked Example <https://youtu.be/PXS0eMV9K9c>`_
#. `Counterfactual from Models <https://youtu.be/Bod_EEIMIr8>`_

