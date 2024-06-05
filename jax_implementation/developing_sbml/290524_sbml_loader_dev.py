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
import os

from sbml_load import *
from collections import OrderedDict
import inspect


jax.config.update("jax_enable_x64", True)
from source.utils import get_logger

logger = get_logger(__name__)

logger.debug('Loading SBML model')
## a simple sbml model

filepath="jax_implementation/developing_sbml/sbml_models/Garde2020.xml"
# filepath="jax_implementation/developing_sbml/sbml_models/Garde2020.xml"
model=load_sbml_model(file_path=filepath)


### All the loading of functions

S=get_stoichiometric_matrix(model)
y0=get_initial_conditions(model)
y0=jnp.array(list(y0.values()))

##recreate create_fluxes, but then for jax
v,v_symbol_dictionaries,local_params=create_fluxes_v(model)
met_point_dict=construct_flux_pointer_dictionary(v_symbol_dictionaries,list(S.columns),list(S.index))



y0=get_initial_conditions(model)
y0=jnp.array(list(y0.values()))



ts=jnp.arange(0,5,0.1)
JaxKmodel = NeuralODE(v=v, S=S, 
                  met_point_dict=met_point_dict,
                  v_symbol_dictionaries=v_symbol_dictionaries)

## we now only gather globally defined parameters, 
#but need to pass local parameters ass well. 

JaxKmodel=jax.jit(JaxKmodel)


# it is probably not wise to pass param_point_dict directly to model,
#because then when we perform gradient calculations, we might
#actually get different gradients for the same parameters


# ####
# # Simulation
# ###

ts=jnp.arange(0,10,0.1)
# #parameters are not yet defined
params=get_global_parameters(model)
params={**local_params,**params}

JaxKmodel(ts=jnp.array([0]),
                       y0=y0,
                       params=params)
# # # print(v_local_param_dict)
ys=JaxKmodel(ts=ts,
      y0=y0,
      params=params)

for i in range(len(S.index)):
      plt.plot(ts,ys[:,i],label=S.index[i])

# plt.plot(ts,ys[:,4],label=species_names[4])
plt.legend()
# 
plt.show()
