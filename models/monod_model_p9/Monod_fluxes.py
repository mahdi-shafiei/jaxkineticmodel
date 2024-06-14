import os
import sys

# sys.path.append("../functions/")
from functions.kinetic_mechanisms.KineticMechanisms import *
from functions.kinetic_mechanisms.KineticModifiers import *
from functions.kinetic_mechanisms.KineticMechanismsCustom import *

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


class Monod_Model(torch.nn.Module):
    def __init__(self,fluxes,metabolites):
        super(Monod_Model,self).__init__()
        self.fluxes=nn.ParameterDict(fluxes)
        self.metabolites=metabolites

    def calculate_fluxes(self,concentrations):
        self.fluxes['mu_L'].value=self.fluxes['mu_L'].calculate(concentrations[self.metabolites['LACT']])
        self.fluxes['mu_A'].value=self.fluxes['mu_A'].calculate(concentrations[self.metabolites['ACT']])
        self.fluxes['mu_P'].value=self.fluxes['mu_P'].calculate(concentrations[self.metabolites['PYR']])
        self.fluxes['r_AL'].value=self.fluxes['r_AL'].calculate(concentrations[self.metabolites['LACT']]*concentrations[self.metabolites['X']])
        self.fluxes['r_PL'].value=self.fluxes['r_PL'].calculate(concentrations[self.metabolites['LACT']]*concentrations[self.metabolites['X']])
        self.fluxes['r_AP'].value=self.fluxes['r_AP'].calculate(concentrations[self.metabolites['PYR']]*concentrations[self.metabolites['X']])
    
    def forward(self,_,conc_in):
        self.calculate_fluxes(conc_in)
        # Additional parameters needed
        Yxl=torch.Tensor([17.0])
        Yxa=torch.Tensor([11.1])
        Yxp=torch.Tensor([16.7])
        K_e=torch.Tensor([0.013])

        LACT=((-(conc_in[self.metabolites['X']] * self.fluxes['mu_L'].value) /Yxl)-
                self.fluxes['r_PL'].value-self.fluxes['r_AL'].value)
        ACT=(self.fluxes['r_AL'].value+ self.fluxes['r_AP'].value - (conc_in[self.metabolites['X']]*self.fluxes['mu_A'].value)/Yxa)
        PYR=self.fluxes['r_PL'].value - ((conc_in[self.metabolites['X']]*self.fluxes['mu_P'].value)/Yxp)-self.fluxes['r_AP'].value
        X=conc_in[self.metabolites['X']]*(self.fluxes['mu_A'].value+self.fluxes['mu_P'].value+self.fluxes['mu_L'].value-K_e)
        dXdt=torch.cat([LACT,ACT,PYR,X],dim=0)
        return dXdt