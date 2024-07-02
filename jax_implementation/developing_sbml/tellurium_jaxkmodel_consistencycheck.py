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


def calc_euclidean(actual, predic):
    return np.sqrt(np.sum((actual - predic) ** 2))




pathname = "jax_implementation/developing_sbml/sbml_models/"
files = os.listdir(pathname)
sbml_files = []
for file in files:
    if file.endswith(".xml"):
        sbml_files.append(file)
    if file.endswith(".sbml"):
        sbml_files.append(file)



working_models_counter=0
max_steps_reached_counter=0
failing_models_counter=0
discrepancy_counter=0
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


        tellurium_model = te.loadSBMLModel(file_path)
        sol_tellurium = tellurium_model.simulate(0,100,200)


        ts = jnp.array(sol_tellurium['time'])
        ys = JaxKmodel(ts=ts, y0=y0, params=params)
        ys=pd.DataFrame(ys,columns=S.index)
        

        ## we need to do another estimate for the error tolerance


        #calculate the MSE between two timeseries because this should be a more stable error measure
        rtols=[]

        for name in S.index:

            # mse=np.sum(sol_tellurium["["+name+"]"]-ys[name])**2
            max_tell=np.max(sol_tellurium["["+name+"]"])
            max_ys=np.max(ys[name])+0.0001
            max_denominator=np.max([max_tell,max_ys])
            rtol=np.abs(sol_tellurium["["+name+"]"]-ys[name])/max_denominator
            # cross_correlation=crosscorr(sol_tellurium["["+name+"]"],ys[name],lag=1)

            rtols.append(rtol)

        mse=np.mean(rtols)


        
        for i,k in enumerate(S.index):
            print(i,k)
            name="["+k+"]"
            plt.plot(ts,sol_tellurium[name],label=name)
            plt.plot(ts,ys[k],label=S.index[i],linewidth=2,linestyle="--")
        plt.legend()
        plt.show()

        S_tellurium = tellurium_model.getFullStoichiometryMatrix()
        if np.sum(np.abs(S_tellurium) - np.abs(np.array(S))) == 0:
            if mse < 0.001:
                print("numerical solve is identical: mse="+str(mse))
                working_models_counter+=1
                os.rename(file_path, pathname + "working_models/" + sbml_file)
            else:
                print("numerical solve is not identical: mse="+str(mse))
                discrepancy_counter+=1
                # os.rename(file_path, pathname + "discrepancies/" + sbml_file)
        else:
            print("discrepancy because of S in " + sbml_file)
            discrepancy_counter+=1
            os.rename(file_path, pathname + "discrepancies/" + sbml_file)
    except jaxlib.xla_extension.XlaRuntimeError as e:
        if 'maximum number of solver steps' not in str(e):
            raise
        logger.error("Maximum number of solver steps reached")
        os.rename(file_path, pathname + "max_steps_reached/" + sbml_file)
        max_steps_reached_counter+=1
    except Exception as e:
        logger.error(f"An exception of type {type(e)} was raised")
        logger.exception(e)
        # os.rename(file_path, pathname + "failing_models/" + sbml_file)
        failing_models_counter+=1

print("failing_models:",failing_models_counter)
print("working_models:",working_models_counter)
print("max steps reached model:",max_steps_reached_counter)
print("discrepancies:",discrepancy_counter)
