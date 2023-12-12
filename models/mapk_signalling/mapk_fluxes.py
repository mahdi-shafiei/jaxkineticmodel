from kinetic_mechanisms.KineticMechanisms import *
from kinetic_mechanisms.KineticMechanismsCustom import *
from kinetic_mechanisms.KineticModifiers import *


def create_fluxes(parameter_dict):
    par_dict=parameter_dict

    v1_modifier=SimpleInhibitor(k_I=par_dict['p_receptor_Ki'],to_be_learned=[True])
    v1=Torch_Irrev_MM_Uni_w_Modifiers(vmax=par_dict['p_receptor_Vmax'],km_substrate=par_dict['p_receptor_Km'],modifiers=[v1_modifier],to_be_learned=[True,True])
    v2=Torch_Irrev_MM_Uni(vmax=par_dict['p_phosphatase1_Vmax'],km_substrate=par_dict['p_phosphatase1_Km'],to_be_learned=[True,True])
    v3=Torch_Irrev_MM_Uni(vmax=par_dict['p_MAPKKKP_kcat'],km_substrate=par_dict['p_MAPKKKP_Km'],to_be_learned=[False,True])
    v4=Torch_Irrev_MM_Uni(vmax=par_dict['p_phosphatase2_Vmax'],km_substrate=par_dict['p_phosphatase2_Km'],to_be_learned=[True,True])
    v5=Torch_Irrev_MM_Uni(vmax=par_dict['p_MAPKK_Kcat'],km_substrate=par_dict['p_MAPKK_Km'],to_be_learned=[False,True])
    v6=Torch_Irrev_MM_Uni(vmax=par_dict['p_phosphatase3_Vmax'],km_substrate=par_dict['p_phosphatase3_Km'],to_be_learned=[True,True])
    v={'receptor':v1,'phosphatase1':v2,'mapkkk':v3,'phosphatase2':v4,'mapkk':v5,'phosphatase3':v6}
    return v
