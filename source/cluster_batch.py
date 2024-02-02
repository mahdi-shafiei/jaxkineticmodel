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
# import torchdiffeq
import re as re 
from parameter_initializations.sampling_methods import uniform_sampling,latinhypercube_sampling

sys.path.append("../models/batch_bioprocess_p4/")
from Batch_Bioprocess_Model_p4 import Bioprocess


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-n',"--name",type=str,required=True,help="Name of the training process that is used to save the file")
    parser.add_argument('-p',"--parameter_sets",type=str,required=True,help="Parameter set file")
    parser.add_argument('-d',"--data",type=str,required=True,help="time series data (NxT dataframe) used to fit")
    parser.add_argument('-o',"--output_dir",type=str,required=True,help="output directory for loss per iteration and the optimized parameters")

    args=parser.parse_args()
    data=pd.read_csv(args.data,index_col=0)
    metabolites_names=data.index.to_list()
    

    #This part needs to be set per model # CHANGE IF NECESSARY
    indices=[0,1]
    metabolites=dict(zip(metabolites_names,indices))
    loss_function_metabolites=[0,1] 

    parameter_sets=pd.read_csv(args.parameter_sets,index_col=0)

    
    #for now I will keep these constant between scripts. Later think of making them into parameters to pass
    error_thresh=1e-4
    max_iter=3000
    gpu=False
    lr=1e-3

    output_dir=args.output_dir



    a=time.time()
    loss_per_iteration=[]
    optimized_parameters=[]
    for i in range(np.shape(parameter_sets)[0]):
        parameter_dict=dict(parameter_sets.iloc[i,:])

        ## In essence, for any training task considered here, this is what we need to change (model)
        model=Bioprocess(parameter_dict)

        trainer=Trainer(model,data,loss_func_targets=loss_function_metabolites,
                        max_iter=max_iter,err_thresh=error_thresh,lr=lr,scaling=True) #remove scaling here and add as additional step


        try:
            trainer.train()
            loss_per_iteration.append(trainer.get_loss_per_iteration)

            named_parameters=dict(trainer.ode.named_parameters())
            named_parameters={i:float(named_parameters[i]) for i in named_parameters}
            optimized_parameters.append(named_parameters)
            
        except: 
            print("cannot solve ODEs, continue")
            continue



    param_match=re.findall("_[a-z]*_\d*.csv",args.parameter_sets)    
    param_match=param_match[0].strip(".csv")
    
    output_filename_loss=output_dir+args.name+"_loss_per_iteration"+param_match+".csv"
    output_filename_optim_params=output_dir+args.name+"_optim_param"+param_match+".csv"
    print(output_filename_loss)
    # loss_dictionary=dict(zip(index,loss_per_iteration))
    # # loss_per_iteration=np.array(loss_per_iteration,dtype=object).reshape(np.shape(loss_per_iteration)[0],-1)
    loss_per_iteration=pd.DataFrame(loss_per_iteration).T
    loss_per_iteration.to_csv(output_filename_loss)

    optimized_parameters=pd.DataFrame(optimized_parameters).T

    optimized_parameters.to_csv(output_filename_optim_params)
    b=time.time()
    print(b-a)

if __name__=="__main__":
    main()