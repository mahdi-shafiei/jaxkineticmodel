

import pandas as pd
from jaxkineticmodel.parameter_estimation.training import Trainer
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt


optimized_params_jkm=pd.read_csv("results/PyPESTO_optimized_params/"
                                 "jaxkineticmodel_optim_params.csv" , index_col=0).to_dict()['0']



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
ys = jaxkmodel(ts=ts,
            y0=model.y0,
            params=optimized_params_jkm,)
ys_jkm=pd.DataFrame(ys,columns=S.index)

for meta in S.index:
    plt.plot(ts, ys_pypesto[meta], label=meta)
    plt.plot(ts, ys_jkm[meta], label=meta, linestyle='--')
    try:
        plt.scatter(timepoints_true,dataset.loc[meta,:], label=meta)
    except:
        continue
    plt.legend()
    plt.show()
