import os
import sys

from torchdiffeq import odeint,odeint_adjoint
from functions.symplectic_adjoint.torch_symplectic_adjoint import odeint_symplectic_adjoint

# sys.path.append("../functions/")
from functions.kinetic_mechanisms.KineticMechanisms import *
from functions.kinetic_mechanisms.KineticModifiers import *
from functions.kinetic_mechanisms.KineticMechanismsCustom import *

from torch import nn
from scipy.interpolate import CubicSpline
import pandas as pd

parameter_dict={"v1_vmax":0.4,"v1_Kgluc":0.1,"v1_Knh3":0.1,
                'v1_Ko2':0.02,'v2_vmax':0.2,'v2_Kgluc':0.15,
                'v4_vmax':4.5,'v4_Kgluc':0.02,'v4_Ko2':0.02,
                'v5_vmax':0.3,'v5_Kshk':0.2,'v5_Knh3':0.1,
                'v6_vmax':0.2,'v6_Kshk':0.2,'v6_Knh3':0.1,
                'v7_vmax':0.1,'v7_Kphe':0.3,'v8_vmax':0.11,
                'v8_Kcin':0.2,'v8_Ko2':0.03}


def create_fluxes(parameter_dict):
    pardict=parameter_dict
    #v1 will be modelled in three terms and multiplied in the mass balance like v11*v22*v33
    v11=Torch_Irrev_MM_Uni(vmax=pardict['v1_vmax'],km_substrate=pardict['v1_Kgluc'],to_be_learned=[True,True])
    # v13=Torch_Irrev_MM_Uni(vmax=1.0,km_substrate=pardict['v1_Ko2'],to_be_learned=[False,True])
    #v2 is simple, the gluc to shiki reaction
    v2=Torch_Irrev_MM_Uni(vmax=pardict['v2_vmax'],km_substrate=pardict['v2_Kgluc'],to_be_learned=[True,True])
    #v3 will not be passed here, but will be defined within the forward calculation. There are no learnable parameters in there.
    #v4 we do the same again as v1
    v41=Torch_Irrev_MM_Uni(vmax=pardict['v4_vmax'],km_substrate=pardict['v4_Kgluc'],to_be_learned=[True,True])
    # v42=Torch_Irrev_MM_Uni(vmax=1.0,km_substrate=pardict['v4_Ko2'],to_be_learned=[False,True])
    #v5 we do the same as v1
    v51=Torch_Irrev_MM_Uni(vmax=pardict['v5_vmax'],km_substrate=pardict['v5_Kshk'],to_be_learned=[True,True])
    #v6 we do the same as v1
    v61=Torch_Irrev_MM_Uni(vmax=pardict['v6_vmax'],km_substrate=pardict['v6_Kshk'],to_be_learned=[True,True])
    #v7
    v7=Torch_Irrev_MM_Uni(vmax=pardict['v7_vmax'],km_substrate=pardict['v7_Kphe'],to_be_learned=[True,True])
    #v8
    v81=Torch_Irrev_MM_Uni(vmax=pardict['v8_vmax'],km_substrate=pardict['v8_Kcin'],to_be_learned=[True,True])
    # v82=Torch_Irrev_MM_Uni(vmax=1.0,km_substrate=pardict['v8_Ko2'],to_be_learned=[False,True])

    substrate=torch.Tensor([0]) #dummy variable to set up tracing
    # v11=torch.jit.trace(v11,(substrate))
    # v2=torch.jit.trace(v2,(substrate))
    # v41=torch.jit.trace(v41,(substrate))
    # v51=torch.jit.trace(v51,(substrate))
    # v61=torch.jit.trace(v61,(substrate))
    # v7=torch.jit.trace(v7,(substrate))
    # v81=torch.jit.trace(v81,(substrate))

    v = {'v11': v11,'v2': v2,'v41': v41,'v51': v51,'v61': v61,'v7': v7,'v81': v81}
    return v



class pCAmodel(torch.nn.Module):
    def __init__(self,fluxes,metabolites):
        super(pCAmodel,self).__init__()
        self.fluxes=nn.ParameterDict(fluxes)
        self.metabolites=metabolites

        ## We need to do the following
        # change calculate to forward
        # in create_fluxes: make torch.jit.trace(v_x,substrate)
        # profile what the conseuqences are using the profiling.py

    



    def calculate_fluxes(self,_,concentrations):
        self.fluxes['v11'].value=self.fluxes['v11'].forward(concentrations[self.metabolites['GLUC']])
        # self.fluxes['v13'].value=self.fluxes['v13'].forward(torch.Tensor(self.boundary(_)))

        self.fluxes['v2'].value=self.fluxes['v2'].forward(concentrations[self.metabolites['GLUC']])

        self.fluxes['v41'].value=self.fluxes['v41'].forward(concentrations[self.metabolites['GLUC']])
        # self.fluxes['v42'].value=self.fluxes['v42'].forward(torch.Tensor(self.boundary(_)))

        self.fluxes['v51'].value=self.fluxes['v51'].forward(concentrations[self.metabolites['SHK']])

        self.fluxes['v61'].value=self.fluxes['v61'].forward(concentrations[self.metabolites['SHK']])


        self.fluxes['v7'].value=self.fluxes['v7'].forward(concentrations[self.metabolites['PHE']])

        self.fluxes['v81'].value=self.fluxes['v81'].forward(concentrations[self.metabolites['CIN']])
        # self.fluxes['v82'].value=self.fluxes['v82'].forward(torch.Tensor(self.boundary(_)))
    
    def forward(self,_,conc_in):
        self.calculate_fluxes(_,conc_in)
        #not the usual way we do this, but want to stick close to the spirit of wouters model
        # v1=self.fluxes['v11'].value #*  self.fluxes['v13'].value


        #non constant boundary condition
        V_GLUC=torch.Tensor([float(self.fluxes['vGluc_Feed'](_))])
        # V_OX_UPT=torch.Tensor([float(self.fluxes['vOx_uptake'](_))])
        # print(V_GLUC,V_OX_UPT)

        #v3 is seperately defined: the dG free energy is assumed to be constant in this cell. This means that the catabolic reaction is balanced by internal loops 
        # using the formula described earlier.
        #dG2 was determined with the group contribution method. We can double check all these values if we would like

        v3=(240*self.fluxes['v11'].value + 145*self.fluxes['v2'].value + self.fluxes['v41'].value + 185*self.fluxes['v51'].value
             + 180*self.fluxes['v61'].value  +20.8*self.fluxes['v7'].value+ 416*self.fluxes['v81'].value )/2843
        # print(conc_in[self.metabolites['O2']])
        #now we set up the mass balances!

        d_GLUC_dt = V_GLUC -0.175*self.fluxes['v11'].value - 7*self.fluxes['v2'].value - v3 #this one will be removed by a polynomial in the end
        d_SHK_dt = 6*self.fluxes['v2'].value - 5*self.fluxes['v51'].value - 16*self.fluxes['v61'].value
        d_TYR_dt = 3*self.fluxes['v51'].value 
        d_PHE_dt = -self.fluxes['v7'].value + 9*self.fluxes['v61'].value 
        d_CIN_dt = self.fluxes['v7'].value- self.fluxes['v81'].value
        d_PCA_dt = self.fluxes['v81'].value
        d_CO2_dt = 0.05*self.fluxes['v11'].value + 8*self.fluxes['v51'].value  + 31*self.fluxes['v61'].value 
        d_Biomass_dt = self.fluxes['v11'].value

        dXdt=torch.cat([d_GLUC_dt,d_SHK_dt,d_TYR_dt,
              d_PHE_dt,d_CIN_dt,d_PCA_dt,d_CO2_dt,d_Biomass_dt],dim=0)

        #one thing we should check if Sv=0

        return dXdt
    

def glucose_rate_polynomial_approximation(t,y,N):
    """Polynomial approximation of the time series data,glucose."""
    #specific for this problem we know the initial feed rate is 1.4, and the 0s in the beginning should be ignored
    y[y<=0.2]=1.4
    convolved_y=np.convolve(y,np.ones(N)/N,mode='valid')
    cs=CubicSpline(t[:len(convolved_y)],convolved_y)
    return cs

# def oxygen_uptake_polynomial_approximation(t,y,N):
#     """Polynomial approximation of the time series data."""
#     #specific for this problem we know the initial feed rate is 1.4, and the 0s in the beginning should be ignored
#     convolved_y=np.convolve(y,np.ones(N)/N,mode='valid')
#     cs=CubicSpline(t[:len(convolved_y)],convolved_y)
#     return cs




if __name__=="__main__":
    fluxes=create_fluxes(parameter_dict)
    # metabolites_names=['GLUC','SHK',"TYR",'PHE','CIN','PCA','CO2','NH3','O2','BIOMASS']
    metabolites_names=['GLUC','SHK',"TYR",'PHE','CIN','PCA','CO2','BIOMASS']

    online_data=pd.read_excel("data/pCA_timeseries/Coumaric acid fermentation data for Paul van Lent.xlsx",sheet_name=3,header=2)

    glucose_feed=online_data['Rate feed C (carbon source) (g/h)'].values[:-1]
    t=online_data['Age (h)'].values[:-1]
    fluxes['vGluc_Feed']=glucose_rate_polynomial_approximation(t,glucose_feed,N=40)



    # oxygen=online_data['Actual oxygen uptake rate (mol/h)'].values[:2869]
    # t=online_data['Age (h)'].values[:2869]

    # oxygen=online_data['Dissolved oxygen (%)'].values[:2869]*0.00028 #mmol/L
    # t=online_data['Age (h)'].values[:2869]


    # fluxes['vOx_uptake']=oxygen_uptake_polynomial_approximation(t,oxygen,3)


    indices=np.arange(0,len(metabolites_names))
    metabolites=dict(zip(metabolites_names,indices))
    model=pCAmodel(fluxes,metabolites)


    initial_values=torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01])
    # initial_values=torch.Tensor(list([1,0.1,0.1,
    #                                   0.1,0.1,0,
    #                                   1.2,10,0.4]))
    time_points=np.linspace(0,95.27555555664003,40)
    tensor_timepoints=torch.tensor(time_points,dtype=torch.float64,requires_grad=False)

    predicted_c =odeint_symplectic_adjoint(func=model, y0=initial_values,method="adaptive_heun", t=tensor_timepoints,rtol=1e-3,atol=1e-6)
    # predicted_c =odeint_adjoint(func=model, y0=initial_values,t=tensor_timepoints,method="cvode",rtol=1e-5,atol=1e-9)







    plt.title("True system")
    for i in range(0,len(initial_values)):
        plt.plot(tensor_timepoints.detach().numpy(),predicted_c.detach().numpy()[:,i],label=metabolites_names[i])
    

    # plt.plot(time_points,fluxes['vGluc_Feed'](time_points),label="Gluc")
    # plt.yscale('symlog')
    plt.legend(bbox_to_anchor=(0.8, 0.5))
    plt.show()