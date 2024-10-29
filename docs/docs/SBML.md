# Loading SBML models

## SBML loader and simulation
SBML models can be loaded and simulated as follows. 

```python3
import matplotlib.pyplot as plt
import os
import sys, os
from source.load_sbml.sbml_load import *
from source.load_sbml.sbml_model import SBMLModel


filepath = (
      "models/sbml_models/working_models/simple_sbml.xml")

# load model from file_path
model = SBMLModel(filepath)
S=model._get_stoichiometric_matrix()
JaxKmodel = model.get_kinetic_model()

ts = jnp.linspace(0,100,2000)

# parameters in sbml can be either global or local parameters. For gradient descent purposes we want all of them global. 
params = get_global_parameters(model.model)
params = {**model.local_params, **params}

#simulate given the initial conditions defined in the sbml 
ys = JaxKmodel(ts=ts,
            y0=model.y0,
            params=params)
ys=pd.DataFrame(ys,columns=S.index)


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
