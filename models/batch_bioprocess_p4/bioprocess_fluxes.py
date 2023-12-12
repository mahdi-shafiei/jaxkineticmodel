import os
import sys

sys.path.append("../functions/")
from kinetic_mechanisms.KineticMechanisms import *
from kinetic_mechanisms.KineticModifiers import *
from kinetic_mechanisms.KineticMechanismsCustom import *



#parameter_dict={"p_qsmax":-0.3,"p_Ks":0.01,"p_a":-1.6,"p_ms":-0.01}

def create_fluxes(parameter_dict):
    par_dict=parameter_dict
    qs=Torch_Irrev_Biomass(vmax=par_dict['qsmax'],km_substrate=par_dict['Ks'],a=par_dict['a'],ms=par_dict['ms'],to_be_learned=[True,True,True,True])
    v={"qs":qs}
    return v

