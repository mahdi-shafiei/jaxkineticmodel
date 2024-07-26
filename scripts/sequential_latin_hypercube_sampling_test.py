

import sys
sys.path.append("/home/plent/Documenten/Gitlab/NeuralODEs/jax_neural_odes/")
from source.parameter_estimation import initialize_parameters
from source.load_sbml.sbml_load import *
from source.load_sbml.sbml_model import SBMLModel
from source.parameter_estimation.training import *
import time


## Mathematically, Sequential Latin Hypercube Sampling exist, but I found it to be quite slow. I will here just design a sampling algorithm that chooses
## new points until there are enough successfull initializations. 



# model_name="Palani2011.xml"
model_name="Messiha2013.xml"
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




lhs_parameter_initializations_full=pd.DataFrame()

a=time.time()
while np.shape(lhs_parameter_initializations_full)[0]< 3:
    inits=[]
    lhs_parameter_initializations=initialize_parameters.latinhypercube_sampling(bounds,N)
    for init in range(np.shape(lhs_parameter_initializations)[0]):
        try:

            params_init=lhs_parameter_initializations.iloc[init,:].to_dict()
            loss=log_loss_func(params_init,ts,jnp.array(dataset))
            inits.append(init)
            print(init,loss)
        except:

            print("failed initialization")
            continue
    lhs_parameter_initializations_full=pd.concat([lhs_parameter_initializations_full,lhs_parameter_initializations.iloc[inits,:]])


b=time.time()

print("time",b-a)


print(lhs_parameter_initializations_full)