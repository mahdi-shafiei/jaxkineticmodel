"""comparison of jaxkinetic compilation time"""

from jaxkineticmodel.parameter_estimation.training import Trainer
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp
import jax
import time


jax.config.update("jax_enable_x64", True)

#load model

model_name = "glycolysis_feastfamine_pulse1"
filepath = "models/manual_implementations/sbml_export/" + model_name + ".xml"

model = SBMLModel(filepath)

kinetic_model = model.get_kinetic_model(compile=False)
#
constants={'D':0.1,'ECbiomass':3.7683659} #non-trainable parameters
kinetic_model._prepare_model(constants)

## right now

dataset = pd.read_csv("datasets/VanHeerden_Glucose_Pulse/FF1_timeseries_format.csv",
                      index_col=0)

# #initialize the trainer object. The required inputs are model and data. We will do 300 iterations of gradient descent



trainer = Trainer(model=kinetic_model, data=dataset.T.iloc[:-4,], n_iter=250, optim_space="log")
parameters_init = pd.DataFrame(pd.Series(kinetic_model.parameters)).T
trainer.parameter_sets = parameters_init

# parameter_sets = trainer.latinhypercube_sampling(model.parameters, lower_bound=0.99999, upper_bound=1.000001, N=1)
start=time.time()
optimized_parameters, loss_per_iteration = trainer.train()
end=time.time()
print("time to optimize", end-start)
pd.DataFrame(optimized_parameters).to_csv("results/PyPESTO_optimized_params/jaxkineticmodel_optim_params.csv")
#
# # # print(parameter_sets.to_dict())
# # individ = parameter_sets.T.to_dict()[0]
# # print(trainer.loss_func(individ, trainer.ts, jnp.array(trainer.dataset)))
#
#
#
# start=time.time()
# optimized_parameters = pd.DataFrame(optimized_parameters).T
# trainer.parameter_sets = optimized_parameters
# trainer.n_iter = 100
# optimized_parameters, loss_per_iteration = trainer.train()
# end=time.time()
# print("time to optimize", end-start)
# # pd.DataFrame(optimized_parameters).to_csv("results/PyPESTO_optimized_params/jaxkineticmodel_optim_params.csv")

