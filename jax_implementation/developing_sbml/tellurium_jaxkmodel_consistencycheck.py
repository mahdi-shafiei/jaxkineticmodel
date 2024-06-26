import jaxlib.xla_extension
import sympy as sp
import jax.numpy as jnp
from sympy.utilities.lambdify import lambdify
import jax
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt

from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt

from functools import partial

import libsbml
from jax_kinetic_model import NeuralODE,create_fluxes_v
import os

from sbml_load import *

import tellurium as te

jax.config.update("jax_enable_x64", True)
from source.utils import get_logger

pathname = "jax_implementation/developing_sbml/sbml_models/"
files = os.listdir(pathname)
sbml_files = []
for file in files:
    if file.endswith(".xml"):
        sbml_files.append(file)
    if file.endswith(".sbml"):
        sbml_files.append(file)

for sbml_file in sbml_files:
    print(sbml_file)
    file_path = pathname + sbml_file
    try:
        # simulate for jax kinetic model
        model = load_sbml_model(file_path=file_path)
        params=get_global_parameters(model)
        assignments_rules = get_assignment_rules_dictionary(model)

        S = get_stoichiometric_matrix(model)
        y0 = get_initial_conditions(model)
        y0=overwrite_init_conditions_with_init_assignments(model,params,assignments_rules,y0)

        y0 = jnp.array(list(y0.values()))

        ##recreate create_fluxes, but then for jax
        v, v_symbol_dictionaries, local_params = create_fluxes_v(model)
        met_point_dict = construct_flux_pointer_dictionary(v_symbol_dictionaries, list(S.columns), list(S.index))


        JaxKmodel = NeuralODE(v=v, S=S,
                              met_point_dict=met_point_dict,
                              v_symbol_dictionaries=v_symbol_dictionaries)
        JaxKmodel = jax.jit(JaxKmodel)

        # #parameters are not yet defined
        
        params = {**local_params, **params}

        ts = jnp.linspace(0, 10, 100)
        ys = JaxKmodel(ts=ts, y0=y0, params=params)

        model = te.loadSBMLModel(file_path)
        sol = model.simulate(0, 10, 100)[:, 1:]

        S_tellurium = model.getFullStoichiometryMatrix()
        if np.sum(np.abs(S_tellurium) - np.abs(np.array(S))) == 0:
            if np.sum(sol - ys) < 0.001:

                print("numerical solve is identical:" + str(np.sum(sol - ys)))
                os.rename(file_path, pathname + "working_models/" + sbml_file)
            else:
                print("numerical solve is not identical" + str(np.sum(sol - ys)))

                os.rename(file_path, pathname + "discrepancies/" + sbml_file)
        else:
            print("discrepancy because of S in " + sbml_file)
            os.rename(file_path, pathname + "discrepancies/" + sbml_file)
    except jaxlib.xla_extension.XlaRuntimeError as e:
        if 'maximum number of solver steps' not in str(e):
            raise
        logger.error("Maximum number of solver steps reached")
        os.rename(file_path, pathname + "max_steps_reached/" + sbml_file)
    except Exception as e:
        logger.error(f"An exception of type {type(e)} was raised")
        logger.exception(e)
        os.rename(file_path, pathname + "failing_models/" + sbml_file)
