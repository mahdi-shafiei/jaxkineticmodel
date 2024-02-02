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

sys.path.append("../functions/load_sbml/")
from create_fluxes_sbml import *
from load_sbml_model import *
import numpy as np
import libsbml


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-n',"--name",type=str,required=True,help="Name of the training process that is used to save the file")
    parser.add_argument('-m',"--model",type=str,required=True,help="SBML model and name")
    parser.add_argument('-p',"--parameter_sets",type=str,required=True,help="Parameter set file")
    parser.add_argument('-d',"--data",type=str,required=True,help="time series data (NxT dataframe) used to fit")
    parser.add_argument('-o',"--output_dir",type=str,required=True,help="output directory for loss per iteration and the optimized parameters")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args()
    data=pd.read_csv(args.data,index_col=0)
    
    metabolites_names=data.index.to_list()
    
    #this is slightly different from the other models,
    model=load_sbml_model(args.model)
    initial_concentration_dict=get_initial_conditions(model)
    
    _,boundaries,compartments=get_model_parameters(model)

    indices=np.arange(0,len(initial_concentration_dict),1)
    metabolites_names=list(initial_concentration_dict.keys())
    metabolites=dict(zip(metabolites_names,indices))
    loss_function_metabolites=indices



    parameter_sets=pd.read_csv(args.parameter_sets,index_col=0)

    #for now I will keep these constant between scripts. Make them into parameters to pass
    error_thresh=0.001
    max_iter=3000
    lr=1e-3

    # print(parameters)
    loss_per_iteration=[]
    optimized_parameters=[]
    for i in range(np.shape(parameter_sets)[0]):
        parameter_dict=dict(parameter_sets.iloc[i,:])

        #this is necessary for some reason
        # parameter_dict={key:torch.tensor([value],dtype=torch.float64,requires_grad=True) for key,value in parameter_dict.items()}
        fluxes=create_fluxes(parameter_dict,boundaries,compartments,model)
        model_n=torch_SBML_kinetic_model(model,fluxes=fluxes).to(device)
        try:        
            trainer=Trainer(model_n,data,loss_func_targets=loss_function_metabolites,
                            max_iter=max_iter,err_thresh=error_thresh,lr=lr,scaling=True,rtol=1e-4,atol=1e-6) 

            trainer.train()        
            loss_per_iteration.append(trainer.get_loss_per_iteration)
            optimized_parameters.append(list(trainer.ode.parameters()))
        except:
            print("cannot solve ODEs, continue")
            continue


    param_match=re.findall("_[a-z]*_\d*.csv",args.parameter_sets)    
    param_match=param_match[0].strip(".csv")


    output_dir=args.output_dir
    output_filename_loss=output_dir+args.name+"_loss_per_iteration"+param_match+".csv"
    output_filename_optim_params=output_dir+args.name+"_optim_param"+param_match+".csv"

    # loss_dictionary=dict(zip(index,loss_per_iteration))
    # # loss_per_iteration=np.array(loss_per_iteration,dtype=object).reshape(np.shape(loss_per_iteration)[0],-1)
    loss_per_iteration=pd.DataFrame(loss_per_iteration).T
    loss_per_iteration.to_csv(output_filename_loss)

    names_parameters=[]
    for name, param in model_n.named_parameters():
        if param.requires_grad:
            names_parameters.append(name)
            
    optimized_parameters=pd.DataFrame(torch.Tensor(optimized_parameters).detach().numpy(),columns=names_parameters)
    optimized_parameters=pd.DataFrame(optimized_parameters).T
    optimized_parameters.to_csv(output_filename_optim_params)

if __name__=="__main__":
    main()