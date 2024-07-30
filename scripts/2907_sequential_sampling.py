

import sys
sys.path.append("/home/plent/Documenten/Gitlab/NeuralODEs/jax_neural_odes/")
from source.parameter_estimation import initialize_parameters
from source.load_sbml.sbml_load import *
from source.load_sbml.sbml_model import SBMLModel
from source.parameter_estimation.training import *
import time
from sklearn.linear_model import LogisticRegression
from typing import Callable, Sequence, Tuple, Optional

import matplotlib.pyplot as plt
## Mathematically, Sequential Latin Hypercube Sampling exist, but I found it to be quite slow. I will here just design a sampling algorithm that chooses
## new points until there are enough successfull initializations. 





# model_name="Palani2011.xml"
model_name="Chassagnole2002.xml"
    # model_name="Berzins2022 - C cohnii glucose and glycerol.xml"
filepath="models/sbml_models/working_models/"+model_name


lb=0.01
ub=100
N=1000

model = SBMLModel(filepath)
S=model._get_stoichiometric_matrix()
JaxKmodel = model.get_kinetic_model()

JaxKmodel.max_steps=10000
JaxKmodel = jax.jit(JaxKmodel)




params = get_global_parameters(model.model)
params = {**model.local_params, **params}
ts=jnp.linspace(0,3,10)

bounds=initialize_parameters.generate_bounds(params,lower_bound=lb,upper_bound=ub)
# lhs_parameter_initializations=initialize_parameters.latinhypercube_sampling(bounds,N)
dataset,params=initialize_parameters.generate_dataset(filepath,ts)


log_loss_func=jax.jit(create_log_params_means_centered_loss_func(JaxKmodel))


sample_func=initialize_parameters.create_sample_model_func(log_loss_func,ts,jnp.array(dataset))


## when it comes to sampling, I want to take an active learning approach. This mean that you first generate an enormous dataset with unlabelled data (e.g. through LHS), and then build a 
## classifier on as little points as possible

#1 generate a lot of points between lower and upper bounds
a=time.time()
parameter_seeds=initialize_parameters.latinhypercube_sampling(bounds,10000)
b=time.time()



a=time.time()
parameter_sets=initialize_parameters.sequential_sampling(sample_func,parameter_seeds,3)
b=time.time()
print(b-a)
    


print(parameter_sets)



for i in range(np.shape(parameter_sets)[0]):
    params_init=dict(parameter_sets.iloc[i,:])

    ys=JaxKmodel(ts=ts,y0=jnp.array(dataset)[0,:],params=params_init)
    plt.plot(ts,ys)



plt.plot(dataset,linestyle="--")
plt.yscale("symlog")
plt.show()


