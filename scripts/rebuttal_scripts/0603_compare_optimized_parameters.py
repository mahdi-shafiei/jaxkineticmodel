

import pandas as pd
from jaxkineticmodel.parameter_estimation.training import Trainer
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
import jax.numpy as jnp
import numpy as np
import jax
import matplotlib.pyplot as plt


optimized_params_jkm=pd.read_csv("results/"
                                 "PyPESTO_optimized_params/"
                                 "jaxkineticmodel_optimized_parameters/"
                                 "18032025_optimized_parameters_run6_n500.csv" , index_col=0).to_dict()['0']
print(optimized_params_jkm)



dataset = pd.read_csv("datasets/VanHeerden_Glucose_Pulse/FF1_timeseries_format.csv",
                      index_col=0)
timepoints_true=[float(i) for i in dataset.columns]
#
optimized_params_pypesto=pd.read_csv("results/PyPESTO_optimized_params/glycolysis_parameters_pyPESTO.csv",index_col=0).to_dict()['0']
#

model_name = "glycolysis_feastfamine_pulse1"
filepath = "models/manual_implementations/sbml_export/" + model_name + ".xml"

model = SBMLModel(filepath)


S=model._get_stoichiometric_matrix()
jaxkmodel=model.get_kinetic_model()
ts = jnp.linspace(0,400,1000)
#
# # simulate given the initial conditions defined in the sbml
#

jaxkmodel=jax.jit(jaxkmodel)

optimized_params_pypesto['D'] = 0.01
optimized_params_pypesto['ECbiomass'] = 3.78

ys = jaxkmodel(ts=ts,
            y0=model.y0,
            params=optimized_params_pypesto,)
#
ys_pypesto=pd.DataFrame(ys,columns=S.index)



optimized_params_jkm['D'] = 0.01
optimized_params_jkm['ECbiomass'] = 3.78
#
#%% calculate a mean squared error estimate

metab_names=['ICATP', 'ICglucose', 'ICADP', 'ICG6P', 'ICtreh', 'ICF6P', 'ICG1P',
       'ICT6P', 'ICFBP', 'ICGAP', 'ICDHAP', 'ICG3P', 'IC3PG', 'ICglyc', 'IC2PG', 'ICPEP',
       'ICPYR', 'ICAMP']

ys = jaxkmodel(ts=timepoints_true,
            y0=model.y0,
            params=optimized_params_pypesto,)
ys_pypesto=pd.DataFrame(ys,columns=S.index)

ys = jaxkmodel(ts=timepoints_true,
            y0=model.y0,
            params=optimized_params_jkm,)
ys_jkm=pd.DataFrame(ys,columns=S.index)

mse_jkm = (np.array(ys_jkm[metab_names])-np.array(dataset.T[metab_names]))**2
mse_jkm= mse_jkm[np.isnan(mse_jkm)==False]
mse_jkm = np.sum(mse_jkm)/ len(mse_jkm)
print('mse jaxkineticmodel', mse_jkm)

mse_pypesto = (np.array(ys_pypesto[metab_names])-np.array(dataset.T[metab_names]))**2
mse_pypesto = mse_pypesto[np.isnan(mse_pypesto)==False]
mse_pypesto = np.sum(mse_pypesto)/ len(mse_pypesto)
print('mse pypesto', mse_pypesto)

#%%
ys = jaxkmodel(ts=ts,
            y0=model.y0,
            params=optimized_params_jkm,)
ys_jkm=pd.DataFrame(ys,columns=S.index)


ys = jaxkmodel(ts=ts,
            y0=model.y0,
            params=optimized_params_pypesto,)
#
ys_pypesto=pd.DataFrame(ys,columns=S.index)
metab_names=['ICATP', 'ICglucose', 'ICADP', 'ICG6P', 'ICtreh', 'ICF6P', 'ICG1P',
       'ICT6P', 'ICFBP', 'ICGAP', 'ICDHAP', 'ICG3P', 'IC3PG', 'ICglyc', 'IC2PG', 'ICPEP',
       'ICPYR', 'ICAMP']


for meta in metab_names:
    fig,ax=plt.subplots(figsize=(4,4))
    ax.plot(ts, ys_pypesto[meta], label=f"{meta}_pypesto",c="black")
    ax.plot(ts, ys_jkm[meta], label=f"{meta}_jaxkineticmodel", linestyle='--',c="black")
    ax.scatter(timepoints_true,dataset.loc[meta,], label=meta, c="black")
    ax.legend()
    fig.savefig(f"figures/pypesto_visual_comparison/{meta}_pypesto.png")

