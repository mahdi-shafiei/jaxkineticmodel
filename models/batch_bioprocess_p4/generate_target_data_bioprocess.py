
import argparse
import numpy as np
import pandas as pd
import os
import torch
import datetime
# import torch.multiprocessing as mp
import time
import sys
# from trainer import Trainer
from multiprocessing import Process,Pool,Queue,Manager,Value
# from kinetic_mechanisms.KineticMechanisms import *
# from kinetic_mechanisms.KineticModifiers import *
# from kinetic_mechanisms.KineticMechanismsCustom import *
from torch.distributions.uniform import Uniform
import matplotlib.pyplot as plt
from torch import nn
from torchdiffeq import odeint_adjoint,odeint
# sys.path.append("../models/batch_bioprocess_p4/")
from Batch_Bioprocess_Model_p4 import Bioprocess
# from bioprocess_fluxes import create_fluxes

parameter_sets=pd.read_csv("Batch_Bioprocess_parametersets.csv",index_col=0)
parameter_dict=dict(parameter_sets.iloc[0,:])
parameter_dict['qsmax']=-0.3
parameter_dict['Ks']=0.01
parameter_dict['a']=-1.6
parameter_dict['ms']=-0.01
print(parameter_dict)
ode=Bioprocess(parameter_dict)


data=pd.read_csv("rawdata_batch_bioprocess.csv",index_col=0)
tensor_concentrations=torch.tensor(np.array(data.T),dtype=torch.float64,requires_grad=False)
time_points=data.columns.to_list()
time_points=[float(i) for i in time_points]
tensor_timepoints=torch.tensor(time_points,dtype=torch.float64,requires_grad=False)


tensor_c0=torch.Tensor([3,0.01])

predicted_c =odeint(func=ode, y0=tensor_c0, t=tensor_timepoints,method="cvode")


target_data=pd.DataFrame(predicted_c.detach().numpy().T,index=['Glucose','Biomass'],columns=time_points)
target_data.to_csv("rawdata_batch_bioprocess.csv")