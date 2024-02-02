import sys

sys.path.append('../models/glycolysis/')
from fluxes import create_fluxes


import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import inspect
import os
import time
from scipy.integrate import OdeSolver,DenseOutput
from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem
from assimulo.solvers import ExplicitEuler
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.integrate import OdeSolver,DenseOutput

from scipy.interpolate import CubicSpline
from torchdiffeq import odeint_adjoint,odeint

sys.path.append("../functions/symplectic_adjoint/")
from torch_symplectic_adjoint import odeint_symplectic_adjoint
from torch import optim
# from TorchDiffEqPack import  odesolve,odesolve_adjoint_sym12,odesolve_adjoint
import torch 
import torch.nn as nn
import traceback

import torch.autograd.profiler as profiler


# scaling=1/(torch.max(tensor_concentrations,0)[0]-torch.min(tensor_concentrations,0)[0])

class Trainer:
    def __init__(self,model,data,loss_func_targets,scaling=False,rtol=1e-8,atol=1e-12,
                 lr=1e-3,max_iter=100,weight_decay=0.0,err_thresh=0.1):
        #data: a pandas dataframe, where rows are metabolites and columns are the timepoints

        super(Trainer,self).__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device=device
        # #switch between GPU and CPU
        # if gpu==False and torch.cuda.is_available()==False:
        #     print("Using cpu")
        #     self.device=torch.device("cpu")
        # else:
        #     self.device=torch.device(f"cuda:{device}")
        #     print("Using gpu")
        
        #The pytorch kinetic model
        self.ode=model
        self.data=data

        self.loss_func_targets=loss_func_targets #Some metabolites are used for interpolation, as they are boundary conditions. We do not want to train on them
        self.max_iter=max_iter
        self.weight_decay=weight_decay
        self.err_thresh=err_thresh
        self.lr=lr
        self.rtol=rtol
        self.atol=atol

        if scaling==False:
            self.yscale=torch.ones(np.shape(self.data)[0])
        elif scaling==True: #make scaling for loss, and make sure that if y_max-y_min is zero, these are not used in loss_function calculations.
            temp=torch.Tensor(np.array(data))
            y_max,y_min=torch.max(temp,1)[0],torch.min(temp,1)[0]
            
            yscale=y_max-y_min
            no_changes_in_conc=np.where(yscale==0)[0]
            if len(no_changes_in_conc)!=0:
                mask=np.arange(len(loss_func_targets))
                mask=mask[mask!=no_changes_in_conc]
                print("use following metabolites to minimize loss: ", list(self.data.index[mask]))
                #Sometimes this type of formatting is buggy??
                loss_func_targets=loss_func_targets[mask]
                self.loss_func_targets=loss_func_targets


            self.yscale=torch.index_select(yscale, 0, torch.LongTensor(loss_func_targets))
            print(self.yscale)
            

        print("Scaling ",self.yscale)
        



    def train(self):
        ## Performs the training 
        #time_points to evaluate
        time_points=self.data.columns.to_list()
        time_points=[float(i) for i in time_points]
        self.tensor_timepoints=torch.tensor(time_points,dtype=torch.float64,requires_grad=False)
        #Concentrations to train on
        self.metabolite_names=self.data.index.to_list()
        self.tensor_concentrations=torch.tensor(np.array(self.data.T),dtype=torch.float64,requires_grad=False)
        # self.tensor_concentrations=self.tensor_concentrations.reshape(shape(1,np.shape(data)[0]))
        # print(np.shape(self.tensor_concentrations))
        #The optimizer and scheduler used for minimizing the loss function()
        optimizer = optim.AdamW(self.ode.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        # print(optimizer.parameters())

        #do we even need this?
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')      
        

        def loss_func():
            """calculates loss function
            1) Only compare data with known values (NaN values should be set to -1),
            2) We could modify it such that concentrations can not be negative
            3) Conservation constraints coulds also be added of metabolites could be added"""
            indices=self.loss_func_targets
            #initial values
            tensor_c0=self.tensor_concentrations[0,:]
            target=self.tensor_concentrations

            try:
                # print("shape target", np.shape(target))
                predicted_c =odeint_symplectic_adjoint(func=self.ode, y0=tensor_c0, t=self.tensor_timepoints,
                                            atol=self.atol,rtol=self.rtol)
                                           # ) #adjoint_options={ "norm" : "seminorm" })
                # predicted_c =odeint_adjoint(func=self.ode, y0=tensor_c0, t=self.tensor_timepoints,atol=self.atol,rtol=self.rtol,
                #                             method="cvode",adjoint_atol=1e-8,adjoint_rtol=1e-6,adjoint_options={ "norm" : "seminorm" })

                
                #take out metabolites that we wish not to include in loss function
                predicted_c=predicted_c[:,indices]
                target=target[:,indices]

                # Rescale the solutions
                target=target*(1/self.yscale.reshape(shape=(1,len(self.yscale))))
                predicted_c=predicted_c*(1/self.yscale.reshape(shape=(1,len(self.yscale))))

                

                ls =torch.mean(torch.square((predicted_c - target)))

            except RuntimeWarning as ex:
                print(ex.args[0]) #potentially add extra argument
                pass
            return ls

        #training process
        try:
            self.get_loss_per_iteration=[]
            self.optimized_parameters=[]
            for i in range(self.max_iter):
                # scheduler_step=False
                optimizer.zero_grad()
                loss=loss_func()

                self.get_loss_per_iteration.append(loss.detach().numpy())
                print('Iter '+str(i)," Loss "+str(loss.item()))
                if loss<self.err_thresh:
                    print("Reached Error Threshold. Break")
                    break

                loss.backward()  #loss.backward

                
                # print("Before optimization:", list(self.ode.parameters())[0])
                optimizer.step()
                # if scheduler_step:
                    # scheduler.step(loss)
                
        except Exception as e:
            print(e)
            traceback.print_exception(*sys.exc_info())
            self.get_loss_per_iteration.append(-1)
            print("Numerical integration problem")

            


