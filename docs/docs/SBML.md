# Loading SBML models

## SBML loader and simulation
SBML models can be loaded and simulated as follows. 


```python
{!code/sbml.py!}
```


## Percentage of similar models
Status report on a large collection of SBML models loaded from [biomodels](https://www.ebi.ac.uk/biomodels/) and from a benchmark collection **[1]**. Discrepancies are models where we compared the output from JaxKineticModel to a simulation using tellurium **[2]**, a popular tool in systems biology. Discrepancies could be there because of numerical differences in the results, or potentially a missing feature in our current implementation (certain event rules are not implemented yet.)

| **Category**             | **Number of working models**                         
|----------------------------------|-------------------------------------------------
|Models working | 31|
| Failing models | 2|
| Discrepancies | 5 |



## References
[1] Hass, H., Loos, C., Raimundez-Alvarez, E., Timmer, J., Hasenauer, J., & Kreutz, C. (2019). Benchmark problems for dynamic modeling of intracellular processes. Bioinformatics, 35(17), 3073-3082.

[2] Choi, K., Medley, J. K., König, M., Stocking, K., Smith, L., Gu, S., & Sauro, H. M. (2018). Tellurium: an extensible python-based modeling environment for systems and synthetic biology. Biosystems, 171, 74-79.
