# Training metabolic kinetic models
Here, we showcase an parameter optimization process with simulated data **[1]**.

## The `Trainer` object
The `Trainer` object requires a few inputs. First, it requires a `SBMLModel` or a `NeuralODEBuild` object to be used. The second input is a datasets to fit on. Here, we show the fitting of a previously reported Serine Biosynthesis model **[1]** .

#### Setting up the trainer object + training
First, we load the necessary functions 

```python3
from jaxkineticmodel.parameter_estimation.initialize_parameters import generate_bounds,latinhypercube_sampling
import optax
from jaxkineticmodel.parameter_estimation.training import Trainer
from jaxkineticmodel.load_sbml.sbml_load import *
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
import jax.numpy as jnp
import jax
import numpy as np
import diffrax 
import matplotlib.pyplot as plt
import pandas as pd
import logging
from jaxkineticmodel.load_sbml.jax_kinetic_model import NeuralODE
```

Then, we load the model, data and initialize the trainer object. 

```python3 

#load model
model_name="Smallbone2013_SerineBiosynthesis"
filepath="models/sbml_models/working_models/"+model_name+".xml"
model = SBMLModel(filepath)


#load data
dataset=pd.read_csv("datasets/Smallbone2013 - Serine biosynthesis/Smallbone2013 - Serine biosynthesis_dataset.csv",index_col=0)

#initialize the trainer object. The required inputs are model and data. We will do 300 iterations of gradient descent
trainer=Trainer(model=model,data=dataset,n_iter=300)

```
We next perform a latin hypercube sampling for a certain initial guess, with lower and upperbound defined with respect to these values. We want five initializations (normally this should be higher).

```python3
base_parameters=dict(zip(trainer.parameters,np.ones(len(trainer.parameters))))
parameter_sets=trainer.latinhypercube_sampling(base_parameters,lower_bound=1/10,upper_bound=10,N=5)
```
To initiate training, you simply call the function `Trainer.train()`
```python3
optimized_parameters,loss_per_iteration,global_norms=trainer.train()


#plot
fig,ax=plt.subplots(figsize=(3,3))
for i in range(5):
    ax.plot(loss_per_iteration[i])
ax.set_xlabel("Iterations")
ax.set_ylabel("Log Loss")
ax.set_yscale("log")

```

![loss](images/loss_per_iter.png)

<span style="font-size: 0.8em;"><b>Figure 1:</b> Loss per iteration for five initializations.</span>
#### Additional rounds
Suppose the fit is not to your liking, or we first want to do a pre-optimization of a large set of initialization and then filter, one can simply continue the optimization as follows

```python3
params_round1=pd.DataFrame(optimized_parameters).T
trainer.parameter_sets=params_round1
trainer.n_iter=500
optimized_parameters2,loss_per_iteration2,global_norms2=trainer.train()

#plot
fig,ax=plt.subplots(figsize=(3,3))
for i in range(5):
    plt.plot(np.concatenate((np.array(loss_per_iteration[i]),loss_per_iteration2[i])))

ax.set_xlabel("Iterations")
ax.set_ylabel("Log Loss")
ax.set_yscale("log")

fig.savefig("docs/docs/images/loss_per_iter_extended.png",bbox_inches="tight")


```
![loss_extended](images/loss_per_iter_extended.png)

<span style="font-size: 0.8em;"><b>Figure 2:</b> Loss per iteration for five initializations, extended with 500 rounds of gradient descent.</span>
## References
[1] Smallbone, K., & Stanford, N. J. (2013). Kinetic modeling of metabolic pathways: Application to serine biosynthesis. Systems Metabolic Engineering: Methods and Protocols, 113-121.
