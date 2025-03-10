"""Checking the gradient steps for pypesto"""

import amici
import matplotlib as mpl
import numpy as np
import time
import petab
import logging
import pypesto.optimize as optimize
import pypesto.petab
import pypesto.profile as profile
import pypesto.sample as sample
import pypesto.store as store
import pypesto.visualize as visualize
import pypesto.visualize.model_fit as model_fit
import pypesto
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
import pandas as pd
from statsmodels.iolib.summary import summary

sbml_file_dir = "models/manual_implementations/sbml_export"
output_dir = "results/PyPESTO_comparison/"

model_name_spline = "glycolysis_feastfamine_pulse1_pypesto_spline"

dataset = pd.read_csv("datasets/VanHeerden_Glucose_Pulse/FF1_timeseries_format.csv",
                      index_col=0).iloc[:, :-1]
timepoints=[float(i) for i in dataset.columns]
metab_names=['ICATP', 'ICglucose', 'ICADP', 'ICG6P', 'ICtreh', 'ICF6P', 'ICG1P',
       'ICT6P', 'ICFBP', 'ICGAP', 'ICDHAP', 'ICG3P', 'IC3PG', 'ICglyc', 'IC2PG', 'ICPEP',
       'ICPYR', 'ICAMP']



model_module = amici.import_model_module(model_name_spline, output_dir)
model = model_module.getModel()

print(model.getObservableIds())



model.setTimepoints(timepoints)
model.setFixedParameters([3.7683659,0.1])
model.setParameters(model.getParameters())
solver = model.getSolver()
rdata = amici.runAmiciSimulation(model, solver)
#
# amici.plotting.plotStateTrajectories(rdata)
# plt.show()


edata = amici.ExpData(
                len(metab_names),
                     0,
                     0,
                     timepoints)

for k, name in enumerate(metab_names):
    edata.setObservedData(dataset.loc[name].values,k)
rdata = amici.runAmiciSimulation(model, solver, edata)


# # we make some more adjustments to our model and the solver
model.requireSensitivitiesForAllParameters()
#
solver.setSensitivityMethod(amici.SensitivityMethod_adjoint)
solver.setSensitivityOrder(1)

objective = pypesto.AmiciObjective(
    amici_model=model, amici_solver=solver, edatas=[edata], max_sensi_order=1
)


problem= pypesto.Problem(objective=objective,
                         lb=np.array(model.getParameters())*0.01,
                         ub=np.array(model.getParameters())*100,
                         x_guesses=[np.array(model.getParameters())])

optimizer_options = {"maxiter": 1e4, "fatol": 1e-12, "frtol": 1e-12}
optimizer = optimize.FidesOptimizer(
    options=optimizer_options, verbose=1
)



start=time.time()
n_starts = 1
engine = pypesto.engine.SingleCoreEngine()
result = optimize.minimize(
    problem=problem,
    optimizer=optimizer,
    n_starts=n_starts,
    engine=engine,
)
end=time.time()
print('run_time', end-start)
print(result.summary())

#%%
# display(Markdown(result.summary()))
names= model.getParameterNames()
values=list(result.optimize_result.x[0])
parameters=dict(zip(names,values))
parameters=pd.DataFrame(pd.Series(parameters))
parameters.to_csv("results/PyPESTO_optimized_params/glycolysis_parameters_pyPESTO.csv")