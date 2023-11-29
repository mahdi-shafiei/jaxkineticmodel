
from sbml_load_functions import *
import numpy as np
from sbml_kinetic_model_class import torch_SBML_kinetic_model
import matplotlib.pyplot as plt


model=load_sbml_model("SBML_models/BIOMD0000000503_url.xml")

initial_concentration_dict=get_initial_conditions(model)

initial_values=torch.Tensor(list(initial_concentration_dict.values()))
time_points=np.linspace(0,100,1000)
tensor_timepoints=torch.tensor(time_points,dtype=torch.float64,requires_grad=False)


parameters=get_model_parameters(model)
# perturb_value=[0.25,0.5,1,2,4,6]

# parameters['BR']=parameters['BR']*perturb_value[i]
worker_ode=torch_SBML_kinetic_model(model,parameter_dict=parameters)
predicted_c =odeint_adjoint(func=worker_ode, y0=initial_values, t=tensor_timepoints,method="cvode",rtol=1e-3,atol=1e-6)


plt.title("Parameter perturbation")
# plt.plot(time_points,predicted_c.detach().numpy()[:,0],label=list(initial_concentration_dict.keys())[0])
plt.plot(time_points,predicted_c.detach().numpy()[:,1],label=list(initial_concentration_dict.keys())[1])
# plt.plot(time_points,predicted_c.detach().numpy()[:,2],label=list(initial_concentration_dict.keys())[2])
# plt.plot(time_points,predicted_c.detach().numpy()[:,3],label=list(initial_concentration_dict.keys())[3])
# plt.plot(time_points,predicted_c.detach().numpy()[:,4],label=list(initial_concentration_dict.keys())[4])
# plt.plot(time_points,predicted_c.detach().numpy()[:,5],label=list(initial_concentration_dict.keys())[5])
plt.legend()
plt.show()