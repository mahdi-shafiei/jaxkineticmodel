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

# sys.path.insert(1,"../mapk_signalling/")

# from map_signalling_model import MAPK_Signalling

import matplotlib.pyplot as plt




### What main needs to do 

# a parameter set dataset, with initial guesses. 
# load a model object: I 
# do not think we should pass this through the cmd,
# let it for now just be in main
# Parse arguments similar to polyODE



def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-n',"--model_name",type=str,required=True,help="name of the kinetic model used")
    parser.add_argument('-f',"--file",type=str,required=True,help="File name for input concentration profiles along with timeseries")
    parser.add_argument('-p',"--parameter_sets",type=str,required=True,help="Parameter sets")
    parser.add_argument("-w","--work_dir",type=str,required=True,help="Working directory")

    parser.add_argument('-d',"--weight_decay", type=float,default=0.0,help="the weight decay for ODE training")
    parser.add_argument('-m',"--max_iter", type=int,default=1000,help="Maximum number of iterations")
    parser.add_argument('-e',"--error_thresh",type=float,default=0.001, help="the threshold on where to stop training")
    parser.add_argument('-l',"--lr",type=float,default=1e-3,help="Learning rate")
    parser.add_argument("-g",'--gpu',type=bool,default=False, help="Use GPU or not")
    parser.add_argument('-j',"--jobs",type=int,default=-1,help="the number of parallel jobs")
    parser.add_argument("-o","--output_dir",type=str,required=False,default="../results/",help="Directory to save all results in")
    
    args=parser.parse_args()
    world_size=args.jobs
    os.chdir(os.path.expandvars(args.work_dir))
    data=pd.read_csv(args.file,index_col=0)
    

    if args.model_name=="bioprocess":
        metabolites_names=data.index.to_list()
        indices=[0,1]
        metabolites=dict(zip(metabolites_names,indices))
        lfc=[0,1]

        #This works, but this is not the proper way to do it I think.
        sys.path.insert(1,args.work_dir)
        from Batch_Bioprocess_Model_p4 import Bioprocess
        from bioprocess_fluxes import create_fluxes


    elif args.model_name=="mapk":
        metabolites_names=data.index.to_list()
        indices=[0,1,2,3,4,5]
        metabolites=dict(zip(metabolites_names,indices))
        lfc=[0,1,2,3,4,5]
        sys.path.insert(1,args.work_dir)
        from map_signalling_model import MAPK_Signalling
        from mapk_fluxes import create_fluxes


    # Load parameter sets
    parameter_sets=pd.read_csv(args.parameter_sets,index_col=0)

    if world_size<=0:
        a=time.time()
        loss_per_iteration=[]
        for i in range(np.shape(parameter_sets)[0]):
            parameter_dict=dict(parameter_sets.iloc[i,:])
            # print(parameter_dict)
            fluxes=create_fluxes(parameter_dict)

            #should be a better way to do this
            if args.model_name=="bioprocess":
                model=Bioprocess(fluxes,metabolites=metabolites)
            elif args.model_name=="mapk":
                model=MAPK_Signalling(fluxes,metabolites=metabolites)

            
            trainer=Trainer(model,data,loss_func_targets=lfc,max_iter=args.max_iter,err_thresh=args.error_thresh,gpu=args.gpu,lr=args.lr)
            try:
                lpi=trainer.train()
                loss_per_iteration.append(lpi)
            except: 
                print("cannot solve ODEs, continue")
                continue
    else:

        ## Values do not save properly yet
        def task(parameter_dict,loss_dict,index):
            #Required for multiprocessing
            if args.model_name=="bioprocess":
                fluxes=create_fluxes(parameter_dict)
                model=Bioprocess(fluxes,metabolites=metabolites)
            elif args.model_name=="mapk":
                fluxes=create_fluxes(parameter_dict)
                model=MAPK_Signalling(fluxes,metabolites=metabolites)
            # model=Bioprocess(parameter_dict=parameter_dict)
            trainer=Trainer(model,data,loss_func_targets=lfc,max_iter=args.max_iter,err_thresh=args.error_thresh,gpu=args.gpu,lr=args.lr)
            try:
                lpi=trainer.train()
                loss_dict[index]=lpi #replace with a MP.Value object?
            except:
                print("cannot solve ODEs, continue")
                loss_dict[index]=None
            return loss_dict
        
        #https://superfastpython.com/multiprocessing-return-value-from-process/
        loss_per_iteration=[]
        with Manager() as manager:
            loss_dict=manager.dict()
            parameter_sets=[dict(parameter_sets.iloc[i,:]) for i in range(np.shape(parameter_sets)[0])]

            processes = []
            for i,parameter_set in enumerate(parameter_sets):
                process = Process(target=task, args=(parameter_set, loss_dict, i))
                processes.append(process)
                process.start()
            
            for process in processes:
                process.join()
            
            # Collect the lpi values from the loss_dict
            for i in range(np.shape(parameter_sets)[0]):
                lpi = loss_dict[i]
                loss_per_iteration.append(lpi)


    # print(params)
    loss_per_iteration=np.array(loss_per_iteration)
    loss_per_iteration=pd.DataFrame(loss_per_iteration)
    loss_per_iteration.to_csv(args.output_dir+"loss_per_iteration.csv")

if __name__=="__main__":
    main()
