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
    "working_models/Palani2011.xml"
)
# filepath="models/sbml_models/working_models/Borghans_BiophysChem1997.xml"
# filepath="models/sbml_models/working_models/Raia_CancerResearch.xml"

model = SBMLModel(filepath)
S=model._get_stoichiometric_matrix()
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

ts = jnp.linspace(0,2,100)
# #parameters are not yet defined
params = get_global_parameters(model.model)
params = {**model.local_params, **params}
print("params",len(params))


JaxKmodel(ts=jnp.array([0]),
          y0=model.y0,
          params=params)
# # # print(v_local_param_dict)
ys = JaxKmodel(ts=ts,
               y0=model.y0,
               params=params)
ys=pd.DataFrame(ys,columns=S.index)

print(np.shape(ys))
# for i in range(len(model.S.index)):
#     plt.plot(ts, ys[:, i], label=model.S.index[i])

# # plt.plot(ts,ys[:,4],label=species_names[4])
# plt.legend()
# # 
# plt.show()


#optional visual comparison for tellurium
import tellurium as te
model = te.loadSBMLModel(filepath)
sol_tell = model.simulate(0, 50, 200)
time_tell=sol_tell['time']
colnames=sol_tell.colnames
print(np.shape(sol_tell))

for name in S.index:
      print(name)
      name_tell="["+name+"]"
      plt.plot(time_tell,sol_tell[name_tell],label=name_tell)
      plt.plot(ts,ys[name],label=name,linewidth=2,linestyle="--")


# plt.legend()
plt.show()