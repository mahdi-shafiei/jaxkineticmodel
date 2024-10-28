# Simulated design-build-test-learn-cycles

## A simulation of metabolic engineering experiments
DBTL cycles are widely used in the optimization of microorganisms for producing valuable compounds in a sustainable way. Despite the widespread use, many open questions exist when it comes to effectively using DBTL cycles. The reason for this is that due to the costly nature (in terms of time and money), effectively comparing design choices is never considered. It is for example highly unlikely that for the same optimization process, different sampling scenarios are compared, even though this might be valuable. An alternative cheap way to answer these types of questions is by simulating DBTL cycles **[1]**. Here, we show a reimplemented software of simulated-DBTL in Jax/Diffrax **[2]**. This allows users to test scenarios for their own lab settings. 

![sim-dbtl](images/dbtl_figure.png)


## Usage
Import required functions
```python
from source.kinetic_mechanisms import JaxKineticMechanisms as jm
from source.building_models import JaxKineticModelBuild as jkm
from source.load_sbml.sbml_load import *
from source.load_sbml.sbml_model import SBMLModel
import jax.numpy as jnp
import jax
import numpy as np
from source.utils import get_logger
logger = get_logger(__name__)

import matplotlib.pyplot as plt
import pandas as pd
import itertools

from source.simulated_dbtl.dbtl import DesignBuildTestLearnCycle
import xgboost as xgb

```

We first choose a kinetic model that we consider only as a black-box that outputs some target value. From this model scenarios are simulated that might be encountered in real metabolic engineering. At the heart of this implementation is the `DesignBuildTestLearnCycle`. This requires setting the initial parameters (the reference state), initial conditions, the timespan of process, and the target that we wish to simulate.

```python
#load model (Messiha et. al (2013))
filepath = ("models/sbml_models/working_models/Messiha2013.xml")
      
model = SBMLModel(filepath)
#we retrieve parameters from the model 
global_params = get_global_parameters(model.model)
params = {**model.local_params, **global_params}

#timespan of model
ts=jnp.linspace(0,6000,100)

dbtl_cycle=DesignBuildTestLearnCycle(model=model,
                          parameters=params,
                          initial_conditions=model.y0,
                          timespan=ts,
                          target=['PEP'])
```


#### Design phase
We now set up the combinatorial pathway optimization experiment. We define some parameter targets that we want to perturb. Then, each target gets some "promoter" values that perturb the parameters by multiplication. Then, we assign probabilities to each promoter-parameter values. If this function is empty, each promoter-parameter is equally probable. Finally, we generate the designs, for 40 samples and 5 chosen pro-parameter elements. 

```python
parameter_target_names=['lp.ADH.Kacald','lp.ENO.kcat_ENO1',
                        'lp.FBA.Kdhap','lp.HXK.kcat_HXK1',
                        'lp.PGK.kcat','lp.HXT.Vmax','lp.GND.Kp6g_GND1']


parameter_perturbation_value=[[0.2,0.5,1,1.5,2], #'lp.ADH.Kacald'
                              [1.2,1.5,1.8], # 'lp.ENO.kcat_ENO1'
                              [1.1,1.6,1.3],#'lp.FBA.Kdhap'
                              [0.6,1.1,2,3],#'lp.HXK.kcat_HXK1'
                              [1,2,3],#'lp.PGK.kcat'
                              [0.5,1,1.5],#'lp.HXT.Vmax'
                              [2,3,4]]#'lp.GND.Kp6g_GND1'
                              
dbtl_cycle.DESIGN_establish_library_elements(parameter_target_names,
                                               parameter_perturbation_value)
                                               
dbtl_cycle.DESIGN_assign_probabilities()

# The replacement is false means that each pro-parameter pair can only be chosen once from the list per strain design
strain_designs=dbtl_cycle.DESIGN_generate_strains(elements=6,samples=40,replacement=False)                                           
```

#### Build/Test phase
In the build phase we simulate all the designs and add a noise model to the outcomes for the target. The first simulation will take quite long, but after that it is compiled. If you do not run the class `DesignBuildTestLearnCycle` again, the simulations remain fast. The values that are taken in the dataset is the average of the last 10 datapoints of the target state variable.

```python
values=dbtl_cycle.BUILD_simulate_strains(strain_designs,plot=False)
noised_values=dbtl_cycle.TEST_add_noise(values,0.1,type='heteroschedastic')
data=dbtl_cycle.TEST_format_dataset(strain_designs=strain_designs,
                               production_values=noised_values,
                               reference_parameters=dbtl_cycle.parameters)
```

#### Learn phase
From here, the produced data can be used to compare whatever hyperparameter of the DBTL cycle you are interested: the performance of ML models, DoE v.s. Random sampling, etc.. As an example, we train an XGBoost model on the set of generated datapoints, as well as a quick validation on newly generated strain designs.

```python
xgbparameters={'tree_method': 'auto','reg_lambda':1,'max_depth':2,"disable_default_eval_metric":0}
alternative_params={'num_boost_round':10,'early_stopping_rounds':40}
bst,r2_scores=dbtl_cycle.LEARN_train_model(data=data,
                                           target="t6p",
                                           model_type="XGBoost",
                                           args=(xgbparameters,alternative_params),test_size=0.20)

dbtl_cycle.LEARN_validate_model(samples=12,
                     elements=12,
                     target='t6p'
                     plotting=True
                     )

``` 
![plot_validate](images/validate.png)


 


## References
[1] van Lent, P., Schmitz, J., & Abeel, T. (2023). Simulated design–build–test–learn cycles for consistent comparison of machine learning methods in metabolic engineering. ACS Synthetic Biology, 12(9), 2588-2599.

[2] Kidger, P. (2022). On neural differential equations. arXiv preprint arXiv:2202.02435.
