#### The pollu model which contains 25 parameters 
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
import argparse

import re as re
sys.path.append("../models/pollution_model_p25/")
from Pollu_fluxes import create_fluxes
from Pollu_model_p25 import Pollu_Model

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-n',"--name",type=str,required=True,help="Name of the training process that is used to save the file")
    # parser.add_argument('-p',"--parameter_sets",type=str,required=True,help="Parameter set file")
    parser.add_argument('-d',"--data",type=str,required=True,help="time series data (NxT dataframe) used to fit")
    parser.add_argument('-o',"--output_dir",type=str,required=True,help="output directory for loss per iteration and the optimized parameters")

    args=parser.parse_args()
    data=pd.read_csv(args.data,index_col=0)
    metabolites_names=data.index.to_list()

    #This part needs to be set per model # CHANGE IF NECESSARY
    #input to model
    indices=np.arange(len(metabolites_names))
    metabolites=dict(zip(metabolites_names,indices)) #is an input to monod model
    loss_function_metabolites=indices

    # parameter_sets=pd.read_csv(args.parameter_sets,index_col=0)

    parameter_sets = {'k1': .35e0,'k2': .266e2,'k3': .123e5,'k4': .86e-3,'k5': .82e-3,'k6': .15e5,'k7': .13e-3,
           'k8': .24e5,'k9': .165e5,'k10': .9e4,'k11': .22e-1,'k12': .12e5,'k13': .188e1,'k14': .163e5,'k15': .48e7,
           'k16': .35e-3,'k17': .175e-1,'k18': .1e9,'k19': .444e12,'k20': .124e4,'k21': .21e1,
           'k22': .578e1,'k23': .474e-1,'k24': .178e4,'k25': .312e1}
    
    parameter_sets['k3']=parameter_sets['k3']*0.5
    parameter_sets['k2']=parameter_sets['k2']*0.3
    parameter_sets['k25']=parameter_sets['k25']*1.3
    parameter_sets['k17']=parameter_sets['k17']*40
    parameter_sets['k19']=parameter_sets['k19']*4
    parameter_sets['k12']=parameter_sets['k12']*0.2
    parameter_sets['k6']=parameter_sets['k6']*0
    #for now I will keep these constant between scripts. Later think of making them into parameters to pass 
    error_thresh=0.0001
    max_iter=1500
    gpu=False
    lr=1e-3

    loss_per_iteration=[]
    optimized_parameters=[]
    # for i in range(np.shape(parameter_sets)[0]):
    # parameter_dict=dict(parameter_sets.iloc[i,:])
    # print(parameter_dict)
    # print(parameter_dict)
    fluxes=create_fluxes(parameter_sets)
    model=Pollu_Model(fluxes,metabolites)

    
    trainer=Trainer(model,data,loss_func_targets=loss_function_metabolites,
                    max_iter=max_iter,err_thresh=error_thresh,gpu=gpu,lr=lr,scaling=False) #remove scaling here and add as additional step
    trainer.scale_data_and_loss(scaling=False) 

    try:
        trainer.train()
        loss_per_iteration.append(trainer.get_loss_per_iteration)
        optimized_parameters.append(list(trainer.ode.parameters()))
        
    except: 
        print("cannot solve ODEs, continue")
        # continue

if __name__=="__main__":
    main()