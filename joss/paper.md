---
title: 'Gw-detect-power: A Python Package to predict the detectability of land management changes on freshwater contaminants despite lag.'
tags:
  - Python
  - monitoring
  - water quality
  - groundwater
  - lag
authors:
  - name: Matt Dumont
    orcid: 0009-0007-2974-0335
    affiliation: 1
  - name: Michael Kittridge
    orcid: 0009-0000-1732-9097
    affiliation: 2
  - name: Richard McDowell
    orcid: 0000-0003-3911-4825
    affiliation: ['3', '4']
affiliations:
 - name: Komanawa Solutions Ltd, Christchurch, New Zealand
   index: 1
 - name: Headwaters Hydrology, Wellington, New Zealand
   index: 2
 - name: AgResearch, Lincoln Science Centre, Lincoln, New Zealand
   index: 3
 - name: Faculty of Agriculture and Life Sciences, Lincoln University, Lincoln, New Zealand
   index: 4
date: 25 July 2025
bibliography: paper.bib
---

# Summary

Understanding the level of monitoring that is required to detect land management actions on freshwater contaminants is an essential part of water quality management and monitoring network design @McDowell_2024.
Despite this need, detection power analysis is often excluded from monitoring programme design and freshwater management plans.
We present a Python package to predict the probability of detecting changes in water quality (detection power) due to prescribed improvements in surface and groundwater contaminant concentrations resulting from land management changes for a given sampling frequency and duration.
The goal of the package is to reduce the barriers for including detection power analysis in the design of water quality management and monitoring programmes.
Importantly, the package is designed to provide detection power estimates for water quality monitoring programmes with and without the effects of groundwater transport processes (lag).
The package supports multiple detection methodologies including linear regression, Mann-Kendall and Multipart-Mann-Kendall tests, as well as counterfactual analysis (pairwise comparisons) for parametric and non-parametric data with Paired Student-T and Wilcoxon tests, respectively.
Finally, the package provides links to worked examples (in a separate repository), a webinar, and other supporting documentation. 
Further details on the package can be found on the [package documentation webpage](https://komanawa.github.io/komanawa-gw-detect-power/).

# Statement of need

Land-based mitigation actions require considerable time and financial investment to reduce the loss of nutrients (e.g., nitrogen) from land into surface and groundwater environments.
Monitoring the effectiveness of these actions is a statutory requirement in New Zealand and is essential to maintain stakeholder confidence, ensure compliance, and effectively manage the natural environment. 
Assuming effective mitigations, their detection are often hampered by the lag between the implementation of land-based mitigation actions and the resulting improvements in water quality as well as the variability of water quality measurements due to natural processes.

There is a trade-off between the cost and probability of detecting changes in water quality.
Very frequent monitoring is expensive, but can also more quickly and accurately detect changes in water quality.
Less frequent monitoring is cheaper, but may not detect changes in water quality for many years if the changes are small relative to the natural variability.
In addition, groundwater transport processes can delay the arrival of contaminants and mix groundwater with of different ages.
These processes (referred to as lag) can delay and decrease the rate of change making it harder to detect.
Typical monitoring frequencies in New Zealand are quarterly or monthly for rivers and quarterly or annually for groundwater.
Without understanding the detection power of these monitoring programmes, it is possible to spend years and considerable resources on monitoring without detecting any changes in water quality or to over-invest in monitoring that is not needed.

Management agencies frequently overlook detection power analysis when designing monitoring programmes due to perceived effort and/or lack of familiarity with these techniques (@weiserTrendPowerToolLookupTool2021).
When such analysis is undertaken, the effects of lag are often ignored due to the complexity of incorporating these effects.
@ascottNeedIntegrateLegacy2021 has shown the need to integrate the effects of lag into the models and decision-making processes used to manage freshwater resources.
This package was created to lower the difficulty of conducting detection power assessments in the context of lag. 
It has been used to conduct a New Zealand wide analysis of the detection power of current monitoring programmes for groundwater and to undertake multiple detailed local case studies of specific catchments(@dumontDeterminingLikelihoodCost2024, @dumont2024power, @dumontAbilityCurrentMonitoring2024).
These studies highlight the cost and times required to detect changes in water quality and the importance of considering lag in the design of monitoring programmes.


# Acknowledgements
Funding to create this package was provided by the Our Land and Water National Science Challenge (contract C10X1507 from the Ministry of Business, Innovation and Employment).

# References
