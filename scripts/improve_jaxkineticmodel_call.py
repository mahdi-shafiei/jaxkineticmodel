"""compile"""

import jax.numpy as jnp
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
from jaxkineticmodel.load_sbml.sbml_load_utils import separate_params, construct_param_point_dictionary
import jax
import time


jax.config.update("jax_enable_x64", True)

filepath = ("models/sbml_models/working_models/Sneyd_PNAS2002.xml")

# load model from file_path
model = SBMLModel(filepath)
S = model._get_stoichiometric_matrix()

jaxkmodel = model.get_kinetic_model()

my_func = jax.jit(jaxkmodel.func)
globals, locals = separate_params(model.parameters)
# ensures that global params are loaded flux specific (necessary for jax)
global_params = construct_param_point_dictionary(
    jaxkmodel.v_symbols, jaxkmodel.reaction_names, globals)  # this is required,

# first call (not compiled)

start = time.time()
my_func.__call__(0, y=model.y0, args=(global_params, locals))
end = time.time()
print(end - start)

start = time.time()
for i in range(100):
    my_func.__call__(0, y=model.y0, args=(global_params, locals))
end = time.time()
print((end - start) / 100)

my_func.__call__(0, y=model.y0, args=(global_params, locals))  #65 us

#0.05892658233642578
#8.245229721069336e-05
