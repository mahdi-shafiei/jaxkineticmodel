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
from torchdiffeq import odeint_adjoint 
from torch import optim

import torch 
import torch.nn as nn
import traceback




# scaling=1/(torch.max(tensor_concentrations,0)[0]-torch.min(tensor_concentrations,0)[0])

class Trainer:
    def __init__(self,model,data,loss_func_targets,scaling=False,rtol=1e-6,atol=1e-3,
                 lr=1e-3,max_iter=100,weight_decay=0.0,err_thresh=0.1,gpu=False):
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
        self.scale=None
        self.yscale=None

    
    def scale_data_and_loss(self,scaling):
        self.tensor_concentrations=torch.tensor(np.array(self.data.T),dtype=torch.float64,requires_grad=False)
        time_points=self.data.columns.to_list()
        time_points=[float(i) for i in time_points]
        self.tensor_timepoints=torch.tensor(time_points,dtype=torch.float64,requires_grad=False)
        ### adds a scaling of the data for better behaved training. scaling of the output of equations is performed in the forward pass of the ODE solver, as well as in the loss function
        #Useful for stiff systems: see paper (Stiff Neural Ordinary Differential Equations)
        if scaling==True:
            y_scale=(torch.max(self.tensor_concentrations,0)[0]-torch.min(self.tensor_concentrations,0)[0])
            t_scale=torch.mean(self.tensor_timepoints[1:]-self.tensor_timepoints[:-1])
            scale=y_scale/t_scale

        elif scaling==False:
            #just a multiplication and division by one in the loss function, has no effect.
            y_scale=torch.ones(len(self.tensor_concentrations[0,:]))
            t_scale=torch.Tensor([1])
            scale=y_scale/t_scale
        self.scale=scale
        self.yscale=y_scale
        return scale,y_scale

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

        optimizer = optim.AdamW(self.ode.parameters(), lr=self.lr)

        # print(optimizer.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
        
        def loss_func():
            """calculates loss function
            1) Only compare data with known values (NaN values should be set to -1),
            2) We could modify it such that concentrations can not be negative
            3) Conservation constraints coulds also be added of metabolites could be added"""
            indices=self.loss_func_targets
            #initial values
            tensor_c0=self.tensor_concentrations[0,:]
            # tensor_c0=tensor_c0.reshape(shape=(len(tensor_c0)))
            target=self.tensor_concentrations

            try:
                # print("shape target", np.shape(target))
                predicted_c =odeint_adjoint(func=self.ode, y0=tensor_c0, t=self.tensor_timepoints,
                                            atol=self.atol,rtol=self.rtol,method="cvode")
                predicted_c=predicted_c[:,:] #seems like a mistake somewhere in the script
                target=(1/self.yscale)*target[None,:][0] #scales the equations according to paper (see comment above)
                predicted_c=(1/self.yscale)*predicted_c[None,:][0]
                predicted_c=predicted_c[:,indices]
                target=target[:,indices]
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
                scheduler_step=True
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
                if scheduler_step:
                    scheduler.step(loss)
                
        except Exception as e:
            print(e)
            self.get_loss_per_iteration.append(-1)
            print("Numerical integration problem")

            


