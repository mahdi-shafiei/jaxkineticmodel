"""Script for profiling jaxkineticmodel"""
import jax
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
from jaxkineticmodel.load_sbml.sbml_load_utils import separate_params, construct_param_point_dictionary
import jax
import jax.numpy as jnp
import time

jax.config.update("jax_enable_x64", True)

filepath = ("models/sbml_models/working_models/simple_sbml.xml")
model = SBMLModel(filepath)

ts = jnp.linspace(0,1,3)
jaxkmodel = model.get_kinetic_model()
ys = jaxkmodel(ts, model.y0 ,model.parameters)
ys.block_until_ready()
global_params, local_params = separate_params(model.parameters)
# ensures that global params are loaded flux specific (necessary for jax)
global_params = construct_param_point_dictionary(
    jaxkmodel.v_symbols, jaxkmodel.reaction_names, global_params)  # this is required,
#
flux_array = jnp.zeros(len(jaxkmodel.reaction_names),dtype=jnp.float64)

dy=jaxkmodel.func.__call__(t=0, y=model.y0, args=(global_params,local_params,flux_array))
dy.block_until_ready()
# dy=jaxkmodel.func.__call__(t=0, y=model.y0, args=(global_params,local_params))
# dy.block_until_ready()

jax.profiler.start_trace("/tmp/jax-trace",create_perfetto_link=True)
# load model from file_path
dy=jaxkmodel.func.__call__(t=0, y=model.y0, args=(global_params,local_params,flux_array))
dy.block_until_ready()




jax.profiler.stop_trace()