
import cProfile

import diffrax
import jax.numpy as jnp
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel

import pandas as pd
import matplotlib.pyplot as plt
import time
import jax
import os
import equinox
import sys
import roadrunner
import tellurium as te

jax.config.update("jax_enable_x64", True)

print(os.getcwd())



model_name="Smallbone2013_SerineBiosynthesis"
filepath = (f"models/sbml_models/working_models/{model_name}.xml")
##

# # # load model from file_path
model = SBMLModel(filepath)
S=model._get_stoichiometric_matrix()


model.compile()
JaxKmodel = model.get_kinetic_model()



# JaxKmodel._change_solver(solver=diffrax.Kvaerno3())


ts = jnp.linspace(0,10,200)

# simulate given the initial conditions defined in the sbml

JaxKmodel=equinox.filter_jit(JaxKmodel)
# #
ys = JaxKmodel(ts=ts,
            y0=model.y0,
            params=model.parameters)
#
ys=pd.DataFrame(ys,columns=S.index)
for i in ys.columns:
    plt.plot(ts,ys[i],label=i)
plt.title("jaxkmodel")
plt.legend()
plt.show()

# print(model.parameters)
# #
# # #
# start=time.time()
# for i in range(100):
#     print(i)
#     ys = JaxKmodel(ts=ts,
#                 y0=model.y0,
#                 params=model.parameters)
# end=time.time()
# jaxkmodel_timing=end-start
# print(jaxkmodel_timing)
# # #
# rr = roadrunner.RoadRunner(filepath)
# rr.integrator.absolute_tolerance = 1e-10
# rr.integrator.relative_tolerance = 1e-7
# rr.integrator.initial_time_step = 1e-11
# rr.integrator.max_steps=300000
# rr.simulate(0,1000,200)
# #
# rr.plot()
# # # #
# # # # # #%%
# tellurium_model = te.loadSBMLModel(filepath)
# tellurium_model.integrator.rtol = 1e-7
# tellurium_model.integrator.atol = 1e-10
# tellurium_model.integrator.initial_time_step = 1e-11
# tellurium_model.integrator.max_steps = 300000
#
# sol_tellurium = tellurium_model.simulate(0, 10, 200)
#
# tellurium_model.plot()

#%%
