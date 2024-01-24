import argparse
import numpy as np
import pandas as pd
import os
import torch
import datetime
# import torch.multiprocessing as mp
import time
import sys
from trainer import Trainer
from multiprocessing import Process,Pool,Queue,Manager,Value
from kinetic_mechanisms.KineticMechanisms import *
from kinetic_mechanisms.KineticModifiers import *
from kinetic_mechanisms.KineticMechanismsCustom import *
from torch.distributions.uniform import Uniform
import matplotlib.pyplot as plt
from torch import nn
import torchdiffeq
import re as re 
from parameter_initializations.sampling_methods import uniform_sampling,latinhypercube_sampling
from load_sbml.create_fluxes_sbml import *
from load_sbml.load_sbml_model import *
import numpy as np
import libsbml
from torch.profiler import profile, record_function, ProfilerActivity

from torchviz import make_dot

model_name="../models/SBML_models/simple_sbml.xml"
data_name="../data/rawdata_simple_sbml.csv"

model=load_sbml_model(model_name)
initial_concentration_dict=get_initial_conditions(model)
# worker_ode=torch_SBML_kinetic_model(model,parameter_dict=parameters)
parameters,boundaries,compartments=get_model_parameters(model)
parameter_sets=pd.DataFrame(pd.Series(parameters)).T
parameter_sets['k']=2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data=pd.read_csv(data_name,index_col=0)
indices=np.arange(0,len(initial_concentration_dict),1)
metabolites_names=list(initial_concentration_dict.keys())
metabolites=dict(zip(metabolites_names,indices))
loss_function_metabolites=indices

error_thresh=0.001
max_iter=3
gpu=False
lr=1e-3

# print(parameters)
loss_per_iteration=[]
optimized_parameters=[]
# for i in range(np.shape(parameter_sets)[0]):
parameter_dict=dict(parameter_sets.iloc[0,:])

time_points=data.columns.to_list()
time_points=[float(i) for i in time_points]
tensor_timepoints=torch.tensor(time_points,dtype=torch.float64,requires_grad=False)

def loss_func():
    """calculates loss function
    1) Only compare data with known values (NaN values should be set to -1),
    2) We could modify it such that concentrations can not be negative
    3) Conservation constraints coulds also be added of metabolites could be added"""
    indices=loss_function_metabolites
    #initial values
    tensor_c0=torch.Tensor(np.array(data.iloc[:,0]))

    # tensor_c0=tensor_c0.reshape(shape=(len(tensor_c0)))
    target=torch.Tensor(np.array(data)).T
    try:
        # print("shape target", np.shape(target))

        predicted_c =odeint_adjoint(func=model, y0=tensor_c0.T, t=tensor_timepoints,method="cvode")
        predicted_c=predicted_c[:,:] #seems like a mistake somewhere in the script
        # target=(1/self.yscale)*target[None,:][0] #scales the equations according to paper (see comment above)
        # predicted_c=(1/self.yscale)*predicted_c[None,:][0]
        predicted_c=predicted_c[:,indices]
        target=target[:,indices]
        ls =torch.mean(torch.square((predicted_c - target)))
    except RuntimeWarning as ex:
        print(ex.args[0]) #potentially add extra argument
        pass
    return ls



# #this is necessary for some reason
# parameter_dict={key:torch.tensor([value],dtype=torch.float64,requires_grad=True) for key,value in parameter_dict.items()}
fluxes=create_fluxes(parameter_dict,boundaries,compartments,model)
model=torch_SBML_kinetic_model(model,fluxes=fluxes).to(device)



loss=loss_func()
dot = make_dot(loss, params=dict(model.named_parameters()))
dot.format = 'png'  # Choose the format for the saved image (e.g., 'png', 'svg', 'pdf')
dot.render("computation_graph_2")  # Save the graph to a file


