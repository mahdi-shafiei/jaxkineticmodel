
from sbml_load_functions import *
import numpy as np

from sbml_kinetic_model_class import torch_SBML_kinetic_model
import matplotlib.pyplot as plt
import time

model=load_sbml_model("SBML_models/BIOMD0000000380_url.xml")

initial_concentration_dict=get_initial_conditions(model)

initial_values=torch.Tensor(list(initial_concentration_dict.values()))
time_points=np.linspace(0,10,1000)
tensor_timepoints=torch.tensor(time_points,dtype=torch.float64,requires_grad=False)
parameters=get_model_parameters(model)
print(len(parameters))


# print(parameters)
# perturb_value=[0.25,0.5,1,2,4,6]

# parameters['BR']=parameters['BR']*perturb_value[i]

worker_ode=torch_SBML_kinetic_model(model,parameter_dict=parameters)

a=time.time()
predicted_c =odeint_adjoint(func=worker_ode, y0=initial_values, t=tensor_timepoints,method="cvode",rtol=1e-4,atol=1e-6)
b=time.time()
print(b-a)
print(initial_values)
# plt.title("Parameter perturbation")
for i in range(len(initial_values)):
    plt.plot(time_points,predicted_c.detach().numpy()[:,i],label=list(initial_concentration_dict.keys())[i])
# plt.yscale("log")
plt.legend()
plt.show()