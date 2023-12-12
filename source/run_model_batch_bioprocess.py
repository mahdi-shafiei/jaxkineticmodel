#### The bioprocess model is slightly simpler than the other models we will test. In this model, we will directly load the parameters into the model, and not
#### create separate fluxes, as this is just an extra unnecessary step.
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

sys.path.append("../models/batch_bioprocess_p4/")
from Batch_Bioprocess_Model_p4 import Bioprocess




a=time.time()
data=pd.read_csv("../models/batch_bioprocess_p4/rawdata_batch_bioprocess.csv",index_col=0)

metabolites_names=data.index.to_list()
indices=[0,1]
metabolites=dict(zip(metabolites_names,indices))
loss_function_metabolites=[0,1] #this is only necessary if you miss data about a certain metabolite
parameter_sets=pd.read_csv("../models/batch_bioprocess_p4/Batch_Bioprocess_parametersets.csv",index_col=0)


world_size=1
error_thresh=0.01
max_iter=2000
gpu=False
lr=1e-3
output_dir="../results/"

if world_size<=0:
    a=time.time()
    loss_per_iteration=[]
    optimized_parameters=[]
    for i in range(np.shape(parameter_sets)[0]):
        parameter_dict=dict(parameter_sets.iloc[i,:])
        # print(parameter_dict)
        print(parameter_dict)
        model=Bioprocess(parameter_dict)

        
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



### Not working yet
else:
    ## Values do not save properly yet
    def task(parameter_dict,loss_list,optim_param_list,index):
        #Required for multiprocessing
        # fluxes=create_fluxes(parameter_dict)  
        # print(list(model.named_parameters()))
        model=Bioprocess(parameter_dict)
        trainer=Trainer(model,data,loss_func_targets=[0,1],max_iter=max_iter,err_thresh=error_thresh,gpu=gpu,lr=lr,scaling=False)
        trainer.scale_data_and_loss(scaling=False) 
        try:
            trainer.train()
            loss_list[index] = trainer.get_loss_per_iteration
            optim_param_list[index] = list(trainer.ode.parameters())
        except:
            print("cannot solve ODEs, continue")
            loss_list[index] = None
            optim_param_list[index] = None

    def run_tasks(parameter_sets):
        with Manager() as manager:
            loss_list = manager.list([None] * len(parameter_sets))
            
            optim_param_list = manager.list([None] * len(parameter_sets))
            print(len(loss_list),len(optim_param_list))
            processes = []
            for i, parameter_set in enumerate(parameter_sets):
                process = Process(target=task, args=(parameter_set, loss_list, optim_param_list, i))
                processes.append(process)
                process.start()

            for process in processes:
                process.join()

            # Convert shared lists to regular lists
            loss_list = list(loss_list)
            optim_param_list = list(optim_param_list)
        return loss_list, optim_param_list
    parameter_sets=[dict(parameter_sets.iloc[i,:]) for i in range(np.shape(parameter_sets)[0])]
    loss_per_iteration, optimized_parameters = run_tasks(parameter_sets)




index=np.arange(0,len(loss_per_iteration),1)
loss_dictionary=dict(zip(index,loss_per_iteration))


# loss_per_iteration=np.array(loss_per_iteration,dtype=object).reshape(np.shape(loss_per_iteration)[0],-1)
loss_per_iteration=pd.DataFrame(loss_per_iteration).T

loss_per_iteration.to_csv(output_dir+"batch_bioprocess_loss_per_iteration.csv")

names_parameters=list(parameter_sets[0].keys())

optimized_parameters=pd.DataFrame(torch.Tensor(optimized_parameters).detach().numpy(),columns=names_parameters)
optimized_parameters=pd.DataFrame(optimized_parameters).T
optimized_parameters.to_csv(output_dir+"batch_bioprocess_optimized_parameters.csv")
b=time.time()
print(b-a)