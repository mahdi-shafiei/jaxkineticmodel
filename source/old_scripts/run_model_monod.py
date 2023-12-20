#### The monod model with 9 parameters 
####


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
from torchdiffeq import odeint_adjoint
from parameter_initializations.sampling_methods import uniform_sampling,latinhypercube_sampling

sys.path.append("../models/monod_model_p9/")
from Monod_fluxes import create_fluxes
from Monod_Bioprocess_Model_p9 import Monod_Model



# {"p_vmax_L":0.57,"p_vmax_P":0.14,"p_vmax_A":0.13,"p_Ks_L":19.4,"p_Ks_P":19.4,
#  "p_Ks_A":10.1,"p_kf_AL":0.71,"p_kf_PL":0.45,"p_kf_AP":0.94}
N_param_sets=30
# generate a random guess
lb=[0.3,0.1,0.1,15,15,8,0.55,0.3,0.7]
ub=[0.7,0.2,0.2,24,24,12,0.85,0.6,1.3]
names=["p_vmax_L","p_vmax_P","p_vmax_A","p_Ks_L","p_Ks_P",
 "p_Ks_A","p_kf_AL","p_kf_PL","p_kf_AP"]
bounds=tuple(zip(lb,ub))
bounds=dict(zip(names,bounds))
parameter_sets=latinhypercube_sampling(bounds,N_param_sets)
# parameter_sets=uniform_sampling(bounds,N_param_sets)


world_size=1
error_thresh=0.01
max_iter=1500
gpu=False
lr=1e-3
output_dir="../results/"

#input to model
metabolites_names=['LACT','ACT','PYR',"X"]
indices=[0,1,2,3]
metabolites=dict(zip(metabolites_names,indices)) #is an input to monod model
loss_function_metabolites=[0,1,2,3]

#data to fit to
data=pd.read_csv("../models/monod_model_p9/rawdata_monod_model_9p.csv",index_col=0)

a=time.time()
if world_size<=0:
    a=time.time()
    loss_per_iteration=[]
    optimized_parameters=[]
    for i in range(np.shape(parameter_sets)[0]):
        parameter_dict=dict(parameter_sets.iloc[i,:])
        # print(parameter_dict)
        fluxes=create_fluxes(parameter_dict)
        model=Monod_Model(fluxes,metabolites)

        
        trainer=Trainer(model,data,loss_func_targets=loss_function_metabolites,
                        max_iter=max_iter,err_thresh=error_thresh,gpu=gpu,lr=lr,scaling=False) #remove scaling here and add as additional step
        trainer.scale_data_and_loss(scaling=False) 

        try:
            trainer.train()
            loss_per_iteration.append(trainer.get_loss_per_iteration)
            optimized_parameters.append(list(trainer.ode.parameters()))
            
        except: 
            print("cannot solve ODEs, continue")
            continue



else:
    ## Values do not save properly yet
    def task(parameter_dict,loss_list,optim_param_list,index):
        #Required for multiprocessing
        # fluxes=create_fluxes(parameter_dict)  
        # print(list(model.named_parameters()))
        fluxes=create_fluxes(parameter_dict)
        model=Monod_Model(fluxes,metabolites)
        trainer=Trainer(model,data,loss_func_targets=loss_function_metabolites,max_iter=max_iter,err_thresh=error_thresh,gpu=gpu,lr=lr,scaling=False)
        trainer.scale_data_and_loss(scaling=False) 
        try:
            trainer.train()
            # loss_list[index] = trainer.get_loss_per_iteration
            # optim_param_list[index] = list(trainer.ode.parameters())
        except:
            print("cannot solve ODEs, continue")
            # loss_list[index] = None
            # optim_param_list[index] = None
        lpi=trainer.get_loss_per_iteration
        optim_param_list=list(trainer.ode.parameters())
        task_dictionary={'loss_per_iteration':lpi,"optimized_parameters":optim_param_list}
        return task_dictionary


    def run_n_tasks(parameter_sets):
        loss_per_iteration=[]
        optimized_parameters=[]

        parameter_sets=[dict(parameter_sets.iloc[i,:]) for i in range(np.shape(parameter_sets)[0])]
        loss_list=list([None] * len(parameter_sets))
        optim_param_list=list([None] * len(parameter_sets))
        indices=np.arange(0, len(parameter_sets),1)
        args=zip(parameter_sets,loss_list,optim_param_list,indices)
        with Pool(processes=4) as pool:
            results=pool.starmap(task,args)
        for result in results:
            loss_per_iteration.append(result['loss_per_iteration'])
            optimized_parameters.append(result['optimized_parameters'])
        return loss_per_iteration,optimized_parameters
        
    loss_per_iteration, optimized_parameters = run_n_tasks(parameter_sets)



index=np.arange(0,len(loss_per_iteration),1)
loss_dictionary=dict(zip(index,loss_per_iteration))


# # loss_per_iteration=np.array(loss_per_iteration,dtype=object).reshape(np.shape(loss_per_iteration)[0],-1)
loss_per_iteration=pd.DataFrame(loss_per_iteration).T
loss_per_iteration.to_csv(output_dir+"2412_monod_model_loss_per_iteration_lhs.csv")

names_parameters=list(parameter_sets.iloc[0,:].keys())

optimized_parameters=pd.DataFrame(torch.Tensor(optimized_parameters).detach().numpy(),columns=names_parameters)
optimized_parameters=pd.DataFrame(optimized_parameters).T
optimized_parameters.to_csv(output_dir+"2412_monod_model_optimized_parameters_lhs.csv")
b=time.time()
print(b-a)
