import cProfile

import diffrax
import jax.numpy as jnp

import pandas as pd
import matplotlib.pyplot as plt
import time
import jax
import os
import equinox
import sys
import roadrunner
import tellurium as te

from jaxkineticmodel.load_sbml import sbml_model

jax.config.update("jax_enable_x64", True)

print(os.getcwd())



filepath= "models/manual_implementations/sbml_export/glycolysis_feastfamine_pulse1.xml"

##

# # # load model from file_path
model = sbml_model.SBMLModel(filepath)
S = model._get_stoichiometric_matrix()

#

jaxkmodel = model.get_kinetic_model()


ts = jnp.linspace(0,100,200)
#
# # simulate given the initial conditions defined in the sbml
#
jaxkmodel = jax.jit(jaxkmodel)
# #
ys = jaxkmodel(ts=ts,
               y0=model.y0,
               params=model.parameters)
#
ys=pd.DataFrame(ys,columns=S.index)

plt.plot(ts,ys)
plt.show()
# #
# # # print(model.parameters)
# # # #
# # # # #
#
start=time.time()
for i in range(100):

    ys = jaxkmodel(ts=ts,
                y0=model.y0,
                params=model.parameters)
end=time.time()
print((end-start)/100)
#
start=time.time()
for i in range(100):
    ys = jaxkmodel(ts=ts,
                y0=model.y0,
                params=model.parameters)
end=time.time()
print((end-start)/100)
#
start=time.time()
for i in range(100):
    ys = jaxkmodel(ts=ts,
                y0=model.y0,
                params=model.parameters)
end=time.time()
print((end-start)/100)


start=time.time()
for i in range(100):
    ys = jaxkmodel(ts=ts,
                y0=model.y0,
                params=model.parameters)
end=time.time()
print((end-start)/100)

# # # #
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
