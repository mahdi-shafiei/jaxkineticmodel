

import sys
sys.path.append('/home/plent/Documenten/Gitlab/NeuralODEs/jax_neural_odes')
from source.load_sbml.sbml_load import *
from source.load_sbml.sbml_model import SBMLModel
jax.config.update("jax_enable_x64", True)
from source.utils import get_logger
from source.parameter_estimation.initialize_parameters import *
import optax
from source.parameter_estimation.training import *
import time
import argparse
from source.parameter_estimation.jacobian import *

model_name="Garde2020.xml"
filepath="models/sbml_models/working_models/"+model_name
model=SBMLModel(filepath)
jacobian_object=Jacobian(model)
logger.info("Jacobian Compiled")
compiled_jacobian=jacobian_object.compile_jacobian()


JaxKmodel = model.get_kinetic_model()
JaxKmodel = jax.jit(JaxKmodel)
# #parameters are not yet defined
params = get_global_parameters(model.model)
params = {**model.local_params, **params}


ts=jnp.linspace(0,10,1000)
ys=JaxKmodel(ts,model.y0,params)

# plt.plot(ts,ys)
# plt.show()

global_params,local_params=separate_params(params)
eigvals=jnp.linalg.eigvals(compiled_jacobian(model.y0,global_params=global_params,local_params=local_params))



# so from this data you would easily see that you need to be between a period of 2 and 3

lbs=[0.99,0.9,0.5,0.25,0.1,0.05,0.025,0.0125,0.01]
ubs=[1.01,1.1,2.0,4.0,10,20,40,80,100]

n_feasibles=[]
N=40000
for i in range(len(lbs)):
    print(i)
    lb=lbs[i]
    ub=ubs[i]

    bounds=generate_bounds(params,lower_bound=lb,upper_bound=ub)
    # uniform_parameter_initializations=uniform_sampling(bounds,N)
    lhs_parameter_initializations=latinhypercube_sampling(bounds,N)

    lhs_parameter_initializations=jacobian_object.filter_oscillations(compiled_jacobian,model.y0,lhs_parameter_initializations,period_bounds=[2.2,2.4])
    n_feasible=np.shape(lhs_parameter_initializations)[0]
    n_feasibles.append(n_feasible)

plt.scatter(ubs,n_feasibles)
plt.xlabel("upper bound (n times true parameter)")
plt.ylabel("log(n feasible)")
plt.yscale("log")
plt.title("Finding ")
plt.show()