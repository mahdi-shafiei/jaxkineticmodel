import sys
sys.path.insert(0,"/home/plent/Documenten/Gitlab/NeuralODEs/jax_neural_odes")

from source.kinetic_mechanisms import JaxKineticMechanisms as jm
from source.building_models import JaxKineticModelBuild as jkm
from source.load_sbml.sbml_load import *
from source.load_sbml.sbml_model import SBMLModel
import jax.numpy as jnp
import jax
import numpy as np
from source.utils import get_logger
logger = get_logger(__name__)
import diffrax 
import matplotlib.pyplot as plt
import pandas as pd


# a simple sbml model
filepath = (
      "models/sbml_models/working_models/Smallbone2011_TrehaloseBiosynthesis.xml")

model = SBMLModel(filepath)
S=model._get_stoichiometric_matrix()
JaxKmodel = model.get_kinetic_model()
JaxKmodel = jax.jit(JaxKmodel)

ts = jnp.linspace(0,10,2000)
# #parameters are not yet defined
global_params = get_global_parameters(model.model)
params = {**model.local_params, **global_params}

ys=JaxKmodel(ts,model.y0,params)


global_params, local_params = separate_params(params)
global_params = construct_param_point_dictionary(JaxKmodel.v_symbol_dictionaries,
                                                         JaxKmodel.reaction_names,
                                                         global_params) 

print(np.shape(ys))

args=(global_params, local_params,JaxKmodel.time_dict)

