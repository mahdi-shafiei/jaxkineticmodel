import matplotlib.pyplot as plt
import os

import sys, os
sys.path.append('/home/plent/Documenten/Gitlab/NeuralODEs/jax_neural_odes')
from source.load_sbml.sbml_load import *
from source.load_sbml.sbml_model import SBMLModel



jax.config.update("jax_enable_x64", True)
from source.utils import get_logger

logger = get_logger(__name__)

logger.debug('Loading SBML model')

# a simple sbml model
filepath = (
    "models/sbml_models/"
    # "failing_models/model_GL-GNT-bypass_13Cflux.xml"
    "working_models/Bertozzi2020.xml"
)
model = SBMLModel(filepath)

JaxKmodel = model.get_kinetic_model()

## we now only gather globally defined parameters, 
# but need to pass local parameters ass well.

JaxKmodel = jax.jit(JaxKmodel)

# it is probably not wise to pass param_point_dict directly to model,
# because then when we perform gradient calculations, we might
# actually get different gradients for the same parameters


# ####
# # Simulation
# ###

ts = jnp.arange(0, 100, 0.1)
# #parameters are not yet defined
params = get_global_parameters(model.model)
params = {**model.local_params, **params}

JaxKmodel(ts=jnp.array([0]),
          y0=model.y0,
          params=params)
# # # print(v_local_param_dict)
ys = JaxKmodel(ts=ts,
               y0=model.y0,
               params=params)

for i in range(len(model.S.index)):
    plt.plot(ts, ys[:, i], label=model.S.index[i])

# plt.plot(ts,ys[:,4],label=species_names[4])
plt.legend()
# 
plt.show()


# #optional visual comparison for tellurium
# import tellurium as te
# model = te.loadSBMLModel(filepath)
# sol_tell = model.simulate(0, 100, 200)
# time_tell=sol_tell['time']
# colnames=sol_tell.colnames[1:]
# sol_tell=sol_tell[:,1:]