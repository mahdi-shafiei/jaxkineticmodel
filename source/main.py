import argparse

import numpy as np
import pandas as pd
import os
import torch
import datetime
import torch.multiprocessing as mp

import sys
from trainer import Trainer

sys.path.append("../batch_bioprocess/")
from torch.distributions.uniform import Uniform
from Batch_Bioprocess_Model_p4 import Bioprocess
import matplotlib.pyplot as plt

# def __init__(self,model,data,
#               loss_func_targets,scaling=True,rtol=1e-3,atol=1e-6,
                #  lr=1e-3,max_iter=100,
                # weight_decay=0.0,err_thresh=0.1,gpu=False):




### What main needs to do 

# a parameter set dataset, with initial guesses. 
# load a model object: I 
# do not think we should pass this through the cmd,
# let it for now just be in main


# Parse arguments similar to polyODE

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-f',"--file",type=str,required=True,help="File name for input concentration profiles along with timeseries")
    parser.add_argument('-p',"--parameter_sets",type=str,required=True,help="Parameter sets")

    parser.add_argument('-d',"--weight_decay", type=float,default=0.0,help="the weight decay for ODE training")
    parser.add_argument('-m',"--max_iter", type=int,default=200,help="Maximum number of iterations")
    parser.add_argument('-e',"--error_thresh",type=float,default=0.001, help="the threshold on where to stop training")
    parser.add_argument('-l',"--lr",type=float,default=1e-3,help="Learning rate")
    parser.add_argument("-g",'--gpu',type=bool,default=False, help="Use GPU or not")
    parser.add_argument('-j',"--jobs",type=int,default=-1,help="the number of parallel jobs")
    parser.add_argument("-o","--output_dir",type=str,required=False,default="../results/",help="Directory to save all results in")
    parser.add_argument("-w","--work_dir",type=str,default="Batch_Bioprocess/.",help="Working directory")
    args=parser.parse_args()


    world_size=args.jobs
    os.chdir(os.path.expandvars(args.work_dir))
    data=pd.read_csv(args.file,index_col=0)
    print('Target data from', args.file)


    # Load parameter sets
    parameter_sets=pd.read_csv(args.parameter_sets,index_col=0)

    if world_size<=0:
        print("run parameter sets sequentially")
        loss_per_iteration=[]
        for i in range(np.shape(parameter_sets)[0]):
            parameter_dict=parameter_sets.iloc[i,:]
            model=Bioprocess(parameter_dict=parameter_dict)
            trainer=Trainer(model,data,loss_func_targets=[0,1],max_iter=args.max_iter,err_thresh=args.error_thresh)
            lpi=trainer.train()
            loss_per_iteration.append(lpi)
        
    else:
        pass
        ### Parallelization still needed to be done
        


    loss_per_iteration=pd.DataFrame(loss_per_iteration)
    loss_per_iteration.to_csv(str(args.output_dir)+"06_10_2023.csv")

    
    
    # plt.plot(loss_per_iteration)
    # plt.show()

if __name__=="__main__":
    main()
