import matplotlib.pyplot as plt
import os
import sys, os
sys.path.append('/home/plent/Documenten/Gitlab/NeuralODEs/jax_neural_odes')
from source.load_sbml.sbml_load import *
from source.load_sbml.sbml_model import SBMLModel
jax.config.update("jax_enable_x64", True)
from source.utils import get_logger
from source.parameter_estimation.initialize_parameters import *
import optax
from source.parameter_estimation.trainer import Trainer

### We now have a working sbml loader for simulation and will start developing the training framework for parameter estimation 
### What needs to be implemented: 3) everything required for training: take inspiration from trainer.py
###
###
###
###
###
###
###
###
model_name="Raia_CancerResearch.xml"
model_name="Bertozzi2020.xml"
filepath="models/sbml_models/working_models/"+model_name
filepath="models/sbml_models/working_models/"+model_name
ts=jnp.linspace(0,150,10)
dataset,params=generate_dataset(filepath,ts)



# plt.plot(ts,dataset)
# plt.show()



N=300
bounds=generate_bounds(params,lower_bound=0.1,upper_bound=10)
uniform_parameter_initializations=uniform_sampling(bounds,N)
lhs_parameter_initializations=latinhypercube_sampling(bounds,N)

id="lhs_"+"N="+str(N)+"run_1"

save_dataset(model_name,dataset)
save_parameter_initializations(model_name,lhs_parameter_initializations,id=id)


### If time
model=SBMLModel(filepath)
JaxKmodel = model.get_kinetic_model()
JaxKmodel = jax.jit(JaxKmodel)
# #parameters are not yet defined
params = get_global_parameters(model.model)
params = {**model.local_params, **params}








# loss=loss_func(lhs_parameter_initializations.iloc[0,:].to_dict(),ts,jnp.array(dataset))




params_init=lhs_parameter_initializations.iloc[0,:].to_dict()
trainer=Trainer(JaxKmodel,dataset,lr=1e-3,max_iter=230)

#this is just for now a test, will be part of trainer
for i in range(np.shape(lhs_parameter_initializations)[0]):
    params_init=lhs_parameter_initializations.iloc[i,:].to_dict()
    loss=trainer.loss_func(params_init,ts)
    trainer.train(params_init)

    