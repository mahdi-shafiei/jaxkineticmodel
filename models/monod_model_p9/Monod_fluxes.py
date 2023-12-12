import os
import sys

sys.path.append("../functions/")
from kinetic_mechanisms.KineticMechanisms import *
from kinetic_mechanisms.KineticModifiers import *
from kinetic_mechanisms.KineticMechanismsCustom import *

parameter_dict={"p_vmax_L":0.57,"p_vmax_P":0.14,"p_vmax_A":0.13,"p_Ks_L":19.4,"p_Ks_P":19.4,"p_Ks_A":10.1,"p_kf_AL":0.71,"p_kf_PL":0.45,"p_kf_AP":0.94}

def create_fluxes(parameter_dict):
    par_dict=parameter_dict
    mu_L=Torch_Irrev_MM_Uni(km_substrate=par_dict['p_Ks_L'],vmax=par_dict['p_vmax_L'],to_be_learned=[True,True])
    mu_A=Torch_Irrev_MM_Uni(km_substrate=par_dict['p_Ks_A'],vmax=par_dict['p_vmax_A'],to_be_learned=[True,True])
    mu_P=Torch_Irrev_MM_Uni(km_substrate=par_dict['p_Ks_P'],vmax=par_dict['p_vmax_P'],to_be_learned=[True,True])
    r_AL=Torch_MA_Irrev(k_fwd=par_dict['p_kf_AL'],to_be_learned=[True])
    r_PL=Torch_MA_Irrev(k_fwd=par_dict['p_kf_PL'],to_be_learned=[True])
    r_AP=Torch_MA_Irrev(k_fwd=par_dict['p_kf_AP'],to_be_learned=[True])
    v={"mu_L":mu_L,"mu_A":mu_A,"mu_P":mu_P,"r_AL":r_AL,"r_PL":r_PL,"r_AP":r_AP}
    return v


fluxes=create_fluxes(parameter_dict)