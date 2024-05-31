import sympy as sp
import jax.numpy as jnp
from sympy.utilities.lambdify import lambdify
import jax
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt
import diffrax
from diffrax import diffeqsolve, ODETerm, Dopri5,SaveAt
import time
from functools import partial
import optax
import libsbml
from jax_kinetic_model import NeuralODE


from sbml_load import *
from collections import OrderedDict
import inspect

from source.utils import get_logger

logger = get_logger(__name__)

logger.debug('Loading SBML model')
## a simple sbml model
filepath="jax_implementation/developing_sbml/sbml_models/Garde2020.xml"
model=load_sbml_model(file_path=filepath)


### All the loading of functions

##recreate create_fluxes, but then for jax
v,v_symbol_dictionaries=create_fluxes_v(model)


S,species_names,reaction_names=get_stoichiometric_matrix(model)
y0=get_initial_conditions(model)



y0=species_match_to_S(y0,species_names)
v=reaction_match_to_S(v,reaction_names)
parameters=get_global_parameters(model)

## we now only gather globally defined parameters, 
#but need to pass local parameters ass well. 

met_point_dict=construct_flux_pointer_dictionary(v_symbol_dictionaries,reaction_names,species_names)
logger.info(f"points towards the species in y to be used for flux eval: {met_point_dict}")


# it is probably not wise to pass param_point_dict directly to model,
#because then when we perform gradient calculations, we might
#actually get different gradients for the same parameters
param_point_dict=construct_param_point_dictionary(v_symbol_dictionaries,
                                                  reaction_names,parameters)

logger.info(f"points towards the parameters to be used for flux eval: {param_point_dict}")
# ####
# # Simulation
# ###

ts=jnp.arange(0,10,0.1)
model = NeuralODE(v=v, S=S, 
                  flux_point_dict=met_point_dict,
                  species_names=species_names,
                  )
model=jax.jit(model)

ys=model(ts,y0,param_point_dict)

plt.plot(ts,ys)
# plt.show()
start=time.time()
for i in range(3):

    param_point_dict[0]['k1']=param_point_dict[0]['k1']*(1/(i+1))

    ys=model(ts,y0,param_point_dict)
    plt.plot(ts,ys[:,0])
plt.show()
end=time.time()
