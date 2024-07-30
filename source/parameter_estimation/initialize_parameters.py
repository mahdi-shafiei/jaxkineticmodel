from source.load_sbml.sbml_load import *
from source.load_sbml.sbml_model import SBMLModel
import os


import numpy as np
import pandas as pd
from scipy.stats import qmc
from skopt.space import Space
from skopt.sampler import Sobol
from skopt.sampler import Lhs
from skopt.sampler import Halton
from skopt.sampler import Hammersly
from skopt.sampler import Grid

## contains parameter initializations within bounds and some helpers for generating synthetic data


def generate_dataset(filepath,ts):
    """Function to generate a dataset from sbml model.
    Used to look at convergence properties"""
    model = SBMLModel(filepath)
    S=model._get_stoichiometric_matrix()
    JaxKmodel = model.get_kinetic_model()
    JaxKmodel = jax.jit(JaxKmodel)
    params = get_global_parameters(model.model)
    params = {**model.local_params, **params}

    ys=JaxKmodel(ts=ts,
          y0=model.y0,
          params=params)
    ys=pd.DataFrame(ys,columns=S.index,index=ts)
    return ys,params

def save_dataset(model_name,dataset):
    """Saves dataset in folder in datasets"""
    model_name=model_name.replace(".xml","")
    model_name=model_name.replace(".sbml","")
    output_filedir="datasets/"+model_name+"/"
    if not os.path.exists(output_filedir):
        os.mkdir(output_filedir)
        filename=output_filedir+model_name+"_dataset.csv"
        dataset.to_csv(filename)
    else:
        print(f"The directory '{output_filedir}' already exists.")
    
    return dataset


def generate_bounds(parameters,lower_bound,upper_bound):
    """Generates bounds for the parameter fitting process. 
    Only useful for models with already known parameters
    lb and ub are w.r.t. true parameter"""
    lbs,ubs,names=[],[],[]

    for key in parameters.keys():
        if parameters[key]!=0:
            lb=parameters[key]*lower_bound
            ub=parameters[key]*upper_bound
        else:
            lb=0
            ub=0.00001  
        lbs.append(lb)
        ubs.append(ub)
        names.append(key)
    bounds=pd.DataFrame({'lb':lbs,'ub':ubs},index=names)
    return bounds


def uniform_sampling(bounds,N):
    """Takes a uniform sample between lower and upperbound values
    Bounds is a pandas dataframe with columns lb and ub and indices are parameter names"""
    parameter_sets=[]
    names=list(bounds.index)
    # print(names)
    for i in range(N):
        random_init={}
        for name in names:

            random_init[name]=np.random.uniform(bounds.loc[name]['lb'],
                                                      bounds.loc[name]['ub'],size=1)[0]
        parameter_sets.append(random_init)

    parameter_sets=pd.DataFrame(parameter_sets)
    return parameter_sets


def latinhypercube_sampling(bounds,N):
    sampler=qmc.LatinHypercube(d=len(bounds.index))
    samples=sampler.random(N)

    lb=bounds['lb']
    ub=bounds['ub']
    
    sample_scaled=qmc.scale(samples,lb,ub)
    names=list(bounds.index)
    parameter_sets=[]
    for i in range(np.shape(sample_scaled)[0]):

        parameter_sets.append(dict(zip(names,sample_scaled[i,:])))
    parameter_sets=pd.DataFrame(parameter_sets)
    return parameter_sets


def save_parameter_initializations(model_name,dataset,id):
    """Saves parameter initializations in folder. Requires setting id string"""
    model_name=model_name.replace(".xml","")
    model_name=model_name.replace(".sbml","")
    output_filedir="parameter_initializations/"+model_name+"/"
    if not os.path.exists(output_filedir):
        os.mkdir(output_filedir)
        filename=output_filedir+model_name+"_parameterset_"+"id_"+id+".csv"
        dataset.to_csv(filename)
    else:
        print(f"The directory '{output_filedir}' already exists. ")
        filename=output_filedir+model_name+"_parameterset_"+"id_"+id+".csv"
        dataset.to_csv(filename)
    return dataset


def save_optimized_params(model_name,result,id):
    """Saves parameter initializations in folder. Requires setting id string"""
    model_name=model_name.replace(".xml","")
    model_name=model_name.replace(".sbml","")
    output_filedir="results/"+model_name+"/"
    if not os.path.exists(output_filedir):
        os.mkdir(output_filedir)
        filename=output_filedir+model_name+"_parameters_"+"id_"+id+".csv"
        result.to_csv(filename)
    else:
        print(f"The directory '{output_filedir}' already exists. ")
        filename=output_filedir+model_name+"_parameters_"+"id_"+id+".csv"
        result.to_csv(filename)
    return result



def save_losses(model_name,result,id):
    """Saves parameter initializations in folder. Requires setting id string"""
    model_name=model_name.replace(".xml","")
    model_name=model_name.replace(".sbml","")
    output_filedir="results/"+model_name+"/"
    if not os.path.exists(output_filedir):
        os.mkdir(output_filedir)
        filename=output_filedir+model_name+"_losses_"+"id_"+id+".csv"
        result.to_csv(filename)
    else:
        print(f"The directory '{output_filedir}' already exists. ")
        filename=output_filedir+model_name+"_losses_"+"id_"+id+".csv"
        result.to_csv(filename)
    return result

def save_norms(model_name,result,id):
    """Saves parameter initializations in folder. Requires setting id string"""
    model_name=model_name.replace(".xml","")
    model_name=model_name.replace(".sbml","")
    output_filedir="results/"+model_name+"/"
    if not os.path.exists(output_filedir):
        os.mkdir(output_filedir)
        filename=output_filedir+model_name+"_norms_"+"id_"+id+".csv"
        result.to_csv(filename)
    else:
        print(f"The directory '{output_filedir}' already exists. ")
        filename=output_filedir+model_name+"_norms_"+"id_"+id+".csv"
        result.to_csv(filename)
    return result


def create_sample_model_func(log_loss_func,ts,ys):
    """Creates an evaluation function to evaluate whether a parameter set is feasible"""
    def evaluate_sample_model(parameter_set):
        try:
            loss=log_loss_func(parameter_set,ts,ys)
            success=1
        except:
            # print("d")
            success=0
        return success
    return evaluate_sample_model


def sequential_sampling(sample_func,parameter_seeds,N_samples):
    """Samples until N of feasible samples is reached,
    sample_func: the function that checks feasibility of the parameter. Right now this is simply the loss function. We might think about doing it in an alternative way (like largest eigenvalue of the jacobian)


    parameter seeds: the parameters initialized using lhs,
    N_samples: number of samples we wish to further optimize"""

    indices=np.arange(0,np.shape(parameter_seeds)[0])
    success=0
    succesfull_indices=[]
    while success<N_samples:
        selected_indices=np.random.choice(indices,size=10,replace=False)
        indices=set(list(indices))-set(list(selected_indices))

        for i in indices:

            loss=sample_func(dict(parameter_seeds.iloc[i,:]))
            if loss==1:

                success+=1
                succesfull_indices.append(i)

            if success >= N_samples:
                break
    parameter_seeds=parameter_seeds.iloc[succesfull_indices,:]
    return parameter_seeds
