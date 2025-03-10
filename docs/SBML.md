# Loading SBML models

## SBML loader and simulation
SBML models can be loaded and simulated as follows. 


```python
{!code/sbml.py!}
```


## SBML test-suite passed tests
While `jaxkineticmodel` is compatible with SBML, not all .xml files are compatible/possible to simulate. This depends on the 
SBML model level and version that it was written. We here report a comparison in number of passed models with a comparison
to libroadrunner using the [sbml-test-suite](https://github.com/sbmlteam/sbml-test-suite/tree/release). 
Libroadrunner is a high-performance simulation engine for systems biology. We compare the simulations to the gold-standard
dataset that is provided in the sbml-test-suite. We selected 500 models from the test-suite and filtered out steady state models.

Discrepancies are models where we compared the output from JaxKineticModel to the ground-truth data and observed
differnces between the time-series. Overall,
libroadrunner simulates more than jaxkineticmodel due to some event rules not yet being implemented. In the future 
we hope to address all these edge-cases.

| Test Name              | Similar Simulation | Failed Simulation | Discrepancies | Total | Last run |
|------------------------|--------------------|-------------------|---------------|-------|----------|
| jaxkineticmodel (l2v1) | 169                | 68                | 52            | 289   | 15-02-25 |
| libroadrunner (l2v1)   | 185                | 62                | 42            | 289   | 15-02-25 |
| jaxkineticmodel (l2v2) | 188                | 99                | 56            | 343   | 12-02-25 |
| libroadrunner (l2v2)   | 219                | 68                | 56            | 343   | 12-02-25 |
| jaxkineticmodel (l2v3) | 189                | 98                | 56            | 343   | 12-02-25 |
| libroadrunner (l2v3)   | 220                | 68                | 55            | 343   | 12-02-25 |
| jaxkineticmodel (l2v4) | 191                | 103               | 56            | 133   | 12-02-25 |
| libroadrunner (l2v4)   | 219                | 73                | 58            | 133   | 12-02-25 |
| jaxkineticmodel (l3v1) | 173                | 206               | 62            | 441   | 12-02-25 |
| libroadrunner (l3v1)   | 197                | 141               | 103           | 441   | 12-02-25 |
| jaxkineticmodel (l3v2) | 171                | 230               | 59            | 460   | 12-02-25 |
| libroadrunner (l3v2)   | 197                | 160               | 103           | 460   | 12-02-25 |







## References
[1] Hass, H., Loos, C., Raimundez-Alvarez, E., Timmer, J., Hasenauer, J., & Kreutz, C. (2019). Benchmark problems for dynamic modeling of intracellular processes. Bioinformatics, 35(17), 3073-3082.

[2] Somogyi, E. T., Bouteiller, J. M., Glazier, J. A., König, M., Medley, J. K., Swat, M. H., & Sauro, H. M. (2015). libRoadRunner: a high performance SBML simulation and analysis library. Bioinformatics, 31(20), 3315-3321.
