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
from load_sbml.load_sbml_model import *
import numpy as np
import libsbml




model_name="../models/SBML_models/simple_sbml.xml"
data_name="../data/rawdata_simple_sbml.csv"

model=load_sbml_model(model_name)
initial_concentration_dict=get_initial_conditions(model)
# worker_ode=torch_SBML_kinetic_model(model,parameter_dict=parameters)
parameters=get_model_parameters(model)
parameter_sets=pd.DataFrame(parameters)
parameter_sets['k']=2


data=pd.read_csv(data_name,index_col=0)


indices=np.arange(0,len(initial_concentration_dict),1)
metabolites_names=list(initial_concentration_dict.keys())
metabolites=dict(zip(metabolites_names,indices))
loss_function_metabolites=indices

error_thresh=0.001
max_iter=1500
gpu=False
lr=1e-3

# print(parameters)
loss_per_iteration=[]
optimized_parameters=[]
for i in range(np.shape(parameter_sets)[0]):
    parameter_dict=dict(parameter_sets.iloc[i,:])

    #this is necessary for some reason
    parameter_dict={key:torch.tensor([value],dtype=torch.float64,requires_grad=True) for key,value in parameter_dict.items()}

    model=torch_SBML_kinetic_model(model,parameter_dict=parameter_dict)

    # print(list(model.named_parameters()))
#     # print(parameter_dict)
#     fluxes=create_fluxes(parameter_dict)
#     model=Monod_Model(fluxes,metabolites)

    
    trainer=Trainer(model,data,loss_func_targets=loss_function_metabolites,
                    max_iter=max_iter,err_thresh=error_thresh,gpu=gpu,lr=lr,scaling=False) #remove scaling here and add as additional step
    print(trainer.ode.fluxes)

    trainer.scale_data_and_loss(scaling=False) 

    # try:
    trainer.train()
    
    loss_per_iteration.append(trainer.get_loss_per_iteration)
    optimized_parameters.append(list(trainer.ode.parameters()))

    # except: 
    #     print("cannot solve ODEs, continue")
    #     continue
