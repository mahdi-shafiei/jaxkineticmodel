from create_fluxes_sbml import *
import numpy as np
import libsbml

from load_sbml_model import *
import matplotlib.pyplot as plt
import time
import pandas as pd
from torchdiffeq import odeint_adjoint

import sys
sys.path.append("../../functions/symplectic_adjoint/")
from torch_symplectic_adjoint import odeint_symplectic_adjoint


#if sbml model
model=load_sbml_model("../../models/SBML_models/Kinetic glycolysis assay model.sbml")

initial_concentration_dict=get_initial_conditions(model)

initial_values=torch.Tensor(list(initial_concentration_dict.values()))
time_points=np.linspace(0,300,1000)
tensor_timepoints=torch.tensor(time_points,dtype=torch.float64,requires_grad=False)
parameters,boundaries,compartments=get_model_parameters(model)
print("parameters ",len(parameters))
# print("boundaries ",boundaries)
# print("compartments ",compartments)




fluxes=create_fluxes(parameters,boundaries,compartments,model)
model=torch_SBML_kinetic_model(model,fluxes=fluxes)

a=time.time()
predicted_c =odeint_symplectic_adjoint(func=model, y0=initial_values,method="adaptive_heun", t=tensor_timepoints,rtol=1e-4,atol=1e-7)
# predicted_c =odeint_adjoint(func=model, y0=initial_values,t=tensor_timepoints,method="cvode",rtol=1e-3,atol=1e-6)
b=time.time()
print(b-a)

# plt.title("Parameter perturbation")
for i in range(len(initial_values)):
    plt.plot(tensor_timepoints.detach().numpy(),predicted_c.detach().numpy()[:,i],label=list(initial_concentration_dict.keys())[i])
# plt.yscale("log")
plt.legend()
plt.show()


df=pd.DataFrame(predicted_c.detach().numpy().T,index=list(initial_concentration_dict.keys()),columns=time_points)
df.to_csv("rawdata_.csv")