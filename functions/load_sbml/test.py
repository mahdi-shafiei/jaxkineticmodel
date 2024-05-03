

import create_fluxes_sbml as torch_sbml
from helper_functions import *
from torchdiffeq import odeint_adjoint
import matplotlib.pyplot as plt
import os
from functions.symplectic_adjoint.torch_symplectic_adjoint import odeint_symplectic_adjoint


#Print wd, and change directory, make  


if __name__=="__main__":
    # name="Berzins2022 - C cohnii glucose and glycerol.xml"
    name="BIOMD0000000062_url.xml"
    # model=load_sbml_model("SBML_models/not_yet_working/"+name)
    s_model=load_sbml_model("models/SBML_models/working_models/"+name)
    initial_concentration_dict=get_initial_conditions(s_model)
    initial_values=torch.Tensor(list(initial_concentration_dict.values()))
    global_parameters=get_global_parameters(s_model)

    a=time.time()
    #local parameters should be retrieved in the torch_SBML rate law class
    compartments=get_compartments(s_model)
    constant_boundaries=get_constant_boundary_species(s_model)
    b=time.time()

    initial_concentration_dict=get_initial_conditions(s_model)

    
    fluxes=torch_sbml.create_fluxes(global_parameters,
                         constant_boundaries,
                         compartments,
                         s_model)
    c=time.time()

    # print(fluxes)
    model=torch_sbml.torch_SBML_kinetic_model(s_model,fluxes=fluxes)

    

    initial_values=torch.Tensor(list(initial_concentration_dict.values()))
    time_points=np.linspace(0,500,1000)
    tensor_timepoints=torch.tensor(time_points,dtype=torch.float64,requires_grad=False)

    predicted_c =odeint_symplectic_adjoint(func=model, y0=initial_values,method="adaptive_heun", t=tensor_timepoints,rtol=1e-4,atol=1e-7)
    # predicted_c =odeint_adjoint(func=model, y0=initial_values,t=tensor_timepoints,method="cvode",rtol=1e-5,atol=1e-9)
    # b=time.time()


    plt.title("True system")
    for i in range(len(initial_values)):
        plt.plot(tensor_timepoints.detach().numpy(),predicted_c.detach().numpy()[:,i],label=i)
    plt.show()
