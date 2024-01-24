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




model_name="../models/SBML_models/BIOMD0000000458_url.xml"
data_name="../data/rawdata_BIOMD0000000458_url.csv"

model=load_sbml_model(model_name)
initial_concentration_dict=get_initial_conditions(model)
# worker_ode=torch_SBML_kinetic_model(model,parameter_dict=parameters)
parameters,boundaries,compartments=get_model_parameters(model)
parameter_sets=pd.DataFrame(pd.Series(parameters)).T
print(parameter_sets)

# parameter_sets['PSA_kcatC']=parameter_sets['PSA_kcatC']*0.5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data=pd.read_csv(data_name,index_col=0)


indices=np.arange(0,len(initial_concentration_dict),1)
metabolites_names=list(initial_concentration_dict.keys())
metabolites=dict(zip(metabolites_names,indices))
loss_function_metabolites=indices

error_thresh=0.01
max_iter=1500
gpu=False
lr=1e-5
a=time.time()
# print(parameters)
loss_per_iteration=[]
optimized_parameters=[]
for i in range(np.shape(parameter_sets)[0]):
    parameter_dict=dict(parameter_sets.iloc[i,:])

    #this is necessary for some reason
    # parameter_dict={key:torch.tensor([value],dtype=torch.float64,requires_grad=True) for key,value in parameter_dict.items()}
    fluxes=create_fluxes(parameter_dict,boundaries,compartments,model)
    model=torch_SBML_kinetic_model(model,fluxes=fluxes).to(device)


    
    trainer=Trainer(model,data,loss_func_targets=loss_function_metabolites,
                    max_iter=max_iter,err_thresh=error_thresh,gpu=gpu,lr=lr,scaling=False) #remove scaling here and add as additional step

    trainer.scale_data_and_loss(scaling=False) 

    # try:

    trainer.train()
    loss_per_iteration.append(trainer.get_loss_per_iteration)
    optimized_parameters.append(list(trainer.ode.parameters()))

    #     print("cannot solve ODEs, continue")
    #     continue

print(optimized_parameters)
b=time.time()
print(b-a)
