
import cProfile

import diffrax
import jax.numpy as jnp
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel

import pandas as pd
import matplotlib.pyplot as plt
import time
import jax
import os
import sys
import roadrunner
import tellurium as te

jax.config.update("jax_enable_x64", True)

print(os.getcwd())



model_name="dano1"
filepath = (f"models/sbml_models/discrepancies/{model_name}.xml")
##

# # load model from file_path
model = SBMLModel(filepath)
S=model._get_stoichiometric_matrix()

JaxKmodel = model.get_kinetic_model()
JaxKmodel._change_solver(solver=diffrax.ImplicitEuler(),acoeff=0.2,rtol=1e-3,atol=1e-6)

ts = jnp.linspace(0,10,200)

#simulate given the initial conditions defined in the sbml

#
#
#
#
#
JaxKmodel=jax.jit(JaxKmodel)
ys = JaxKmodel(ts=ts,
            y0=model.y0,
            params=model.parameters)
ys=pd.DataFrame(ys,columns=S.index)
plt.plot(ts,ys)
plt.title("jaxkmodel")
plt.show()

#
# start=time.time()
for i in range(1000):
    print(i)
    ys = JaxKmodel(ts=ts,
                y0=model.y0,
                params=model.parameters)
# # end=time.time()
# # jaxkmodel_timing=end-start
# # print(jaxkmodel_timing)
# # # #
# rr = roadrunner.RoadRunner(filepath)
# rr.integrator.absolute_tolerance = 1e-10
# rr.integrator.relative_tolerance = 1e-7
# rr.integrator.initial_time_step = 1e-11
# rr.integrator.max_steps=300000
# rr.simulate(0,10,200)
# #
# rr.plot()
# #
# # #%%
# tellurium_model = te.loadSBMLModel(filepath)
# tellurium_model.integrator.rtol = 1e-7
# tellurium_model.integrator.atol = 1e-10
# tellurium_model.integrator.initial_time_step = 1e-11
# tellurium_model.integrator.max_steps = 300000
#
# sol_tellurium = tellurium_model.simulate(0, 10, 200)
#
# # tellurium_model.plot()

#%%
