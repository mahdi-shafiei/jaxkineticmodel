
import sys
sys.path.append("../../functions/")
from kinetic_mechanisms.KineticMechanisms import *
from kinetic_mechanisms.KineticModifiers import *
from kinetic_mechanisms.KineticMechanismsCustom import *
import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint_adjoint 
import time
import pandas as pd


parameter_dict = {'k1': .35e0,'k2': .266e2,'k3': .123e5,'k4': .86e-3,'k5': .82e-3,'k6': .15e5,'k7': .13e-3,
           'k8': .24e5,'k9': .165e5,'k10': .9e4,'k11': .22e-1,'k12': .12e5,'k13': .188e1,'k14': .163e5,'k15': .48e7,
           'k16': .35e-3,'k17': .175e-1,'k18': .1e9,'k19': .444e12,'k20': .124e4,'k21': .21e1,
           'k22': .578e1,'k23': .474e-1,'k24': .178e4,'k25': .312e1}

def create_fluxes(parameter_dict):
    r1=Torch_MA_Irrev(k_fwd=parameter_dict['k1'],to_be_learned=[True])
    r2=Torch_MA_Irrev(k_fwd=parameter_dict['k2'],to_be_learned=[True])
    r3=Torch_MA_Irrev(k_fwd=parameter_dict['k3'],to_be_learned=[True])
    # r=Torch_MA_Irrev_multiple_substrates(k_fwd=parameter_dict['k2'],to_be_learned=[True])
    r4=Torch_MA_Irrev(k_fwd=parameter_dict['k4'],to_be_learned=[True])
    r5=Torch_MA_Irrev(k_fwd=parameter_dict['k5'],to_be_learned=[True])
    r6=Torch_MA_Irrev(k_fwd=parameter_dict['k6'],to_be_learned=[True])
    r7=Torch_MA_Irrev(k_fwd=parameter_dict['k7'],to_be_learned=[True])
    r8=Torch_MA_Irrev(k_fwd=parameter_dict['k8'],to_be_learned=[True])
    r9=Torch_MA_Irrev(k_fwd=parameter_dict['k9'],to_be_learned=[True])
    r10=Torch_MA_Irrev(k_fwd=parameter_dict['k10'],to_be_learned=[True])
    r11=Torch_MA_Irrev(k_fwd=parameter_dict['k11'],to_be_learned=[True])
    r12=Torch_MA_Irrev(k_fwd=parameter_dict['k12'],to_be_learned=[True])
    r13=Torch_MA_Irrev(k_fwd=parameter_dict['k13'],to_be_learned=[True])
    r14=Torch_MA_Irrev(k_fwd=parameter_dict['k14'],to_be_learned=[True])
    r15=Torch_MA_Irrev(k_fwd=parameter_dict['k15'],to_be_learned=[True])
    r16=Torch_MA_Irrev(k_fwd=parameter_dict['k16'],to_be_learned=[True])
    r17=Torch_MA_Irrev(k_fwd=parameter_dict['k17'],to_be_learned=[True])
    r18=Torch_MA_Irrev(k_fwd=parameter_dict['k18'],to_be_learned=[True])
    r19=Torch_MA_Irrev(k_fwd=parameter_dict['k19'],to_be_learned=[True])
    r20=Torch_MA_Irrev(k_fwd=parameter_dict['k20'],to_be_learned=[True])
    r21=Torch_MA_Irrev(k_fwd=parameter_dict['k21'],to_be_learned=[True])
    r22=Torch_MA_Irrev(k_fwd=parameter_dict['k22'],to_be_learned=[True])
    r23=Torch_MA_Irrev(k_fwd=parameter_dict['k23'],to_be_learned=[True])
    r24=Torch_MA_Irrev(k_fwd=parameter_dict['k24'],to_be_learned=[True])
    r25=Torch_MA_Irrev(k_fwd=parameter_dict['k25'],to_be_learned=[True])
    v={"r1":r1,"r2":r2,"r3":r3,"r4":r4,"r5":r5,"r6":r6,
       "r7":r7,"r8":r8,"r9":r9,"r10":r10,"r11":r11,"r12":r12,
       "r13":r13,"r14":r14,"r15":r15,"r16":r16,"r17":r17,"r18":r18,
       "r19":r19,"r20":r20,"r21":r21,"r22":r22,"r23":r23,"r24":r24,"r25":r25}
    return v


fluxes=create_fluxes(parameter_dict)
