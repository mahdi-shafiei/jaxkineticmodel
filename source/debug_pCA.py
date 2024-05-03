import os
import sys
import time
import re
from torchdiffeq import odeint,odeint_adjoint
sys.path.append("../")
from functions.symplectic_adjoint.torch_symplectic_adjoint import odeint_symplectic_adjoint
import argparse

from functions.kinetic_mechanisms.KineticMechanisms import *
from functions.kinetic_mechanisms.KineticModifiers import *
from functions.kinetic_mechanisms.KineticMechanismsCustom import *

from torch import nn
from scipy.interpolate import CubicSpline
import pandas as pd

from models.pCA_model_p20.pCA_fluxes import *
from trainer import Trainer

parameter_dict={"v1_vmax":0.4,"v1_Kgluc":0.1,
                'v1_Ko2':0.02,'v2_vmax':0.2,'v2_Kgluc':0.15,
                'v4_vmax':0.5,'v4_Kgluc':0.02,'v4_Ko2':0.02,
                'v5_vmax':0.3,'v5_Kshk':0.2,
                'v6_vmax':0.2,'v6_Kshk':0.2,
                'v7_vmax':0.1,'v7_Kphe':0.3,'v8_vmax':0.11,
                'v8_Kcin':0.2,'v8_Ko2':0.03}






data=pd.read_csv("data/pCA_timeseries/pCA_fermentation_data_200424.csv",index_col=0)

#load the online data (required for fitting)
online_data=pd.read_excel("data/pCA_timeseries/Coumaric acid fermentation data for Paul van Lent.xlsx",sheet_name=3,header=2)

metabolites_names=data.index.to_list()
indices=np.arange(0,len(metabolites_names))
metabolites=dict(zip(metabolites_names,indices))



    
error_thresh=0.5
max_iter=2000
gpu=False
lr=1e-3




fluxes=create_fluxes(parameter_dict)

#this is specific to the pca model. The glucose uptake rate is a given

glucose_feed=online_data['Rate feed C (carbon source) (g/h)'].values[:-1]
t=online_data['Age (h)'].values[:-1]
fluxes['vGluc_Feed']=glucose_rate_polynomial_approximation(t,glucose_feed,N=40)


oxygen=online_data['Dissolved oxygen (%)'].values[:2869]*0.28 #mmol/L
t=online_data['Age (h)'].values[:2869]
fluxes['vOx_uptake']=oxygen_uptake_polynomial_approximation(t,oxygen,3)


model=pCAmodel(fluxes,metabolites)

for i in model.named_parameters():
    print(i)

loss_function_metabolites=[1,2,5,6]

trainer=Trainer(model,data,loss_func_targets=loss_function_metabolites,
                    max_iter=max_iter,err_thresh=error_thresh,lr=lr,scaling=False,rtol=1e-3,atol=1e-6) #remove scaling here and add as additional step

#make sure that scaling is performed for glucose (since in data yscale=0, but in first simulation it is almost 400. )
# I will for now arbitrarily set this to 30, because this is an important property to be learned 



trainer.train()
# loss_per_iteration.append(trainer.get_loss_per_iteration)

#the fuck am i doing here
named_parameters=dict(trainer.ode.named_parameters())
named_parameters={i:float(named_parameters[i]) for i in named_parameters}
# optimized_parameters.append(named_parameters)


