"""This script runs a comparison in terms of similarity on the sbml-test-suite for increaising functionality"""


import os
import pandas as pd
import datetime
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
import jax.numpy as jnp
from jaxkineticmodel.utils import get_logger
import numpy as np
import roadrunner

logger = get_logger(__name__)
logger.disabled = True

file_path="/home/plent/Documenten/Gitlab/sbml-test-suite/cases/semantic"

n_models=500

sbml_level="l3v2.xml"


def parse_file(filename):
    data = {}

    with open(filename, 'r') as file:
        for line in file:
            if ":" in line:
                key, value = line.split(":", 1)  # Split at first colon
                key, value = key.strip(), value.strip()

                # Convert numeric values
                if value.replace('.', '', 1).isdigit():
                    value = float(value) if '.' in value else int(value)
                # Convert comma-separated values into lists
                elif "," in value:
                    value = [v.strip() for v in value.split(",")]
                # Keep empty values as empty strings
                elif value == "":
                    value = None

                data[key] = value

    return data



sbml_test_models=os.listdir(file_path)[0:n_models]
#[0:n_models]
print("number of considered test-suite models",len(sbml_test_models))
timeseries_models=[]


for sbml_model in sbml_test_models:
    if os.path.isdir(os.path.join(file_path,sbml_model)):
        model_path=os.path.join(file_path,sbml_model)
        model_files=os.listdir(model_path)
        for file in model_files:
            if file.endswith("settings.txt"):
                settings=parse_file(os.path.join(model_path,file))
                if settings['start'] == None and settings['duration'] == None:
                    pass
                else:
                    timeseries_models.append(sbml_model)

print("number of timeseries models",len(timeseries_models))

sbml_counter=0
l2v2_counter_jax={'similar_simulation':0,'failed_simulation':0,'discrepancies':0}
l2v2_counter_libroadrunner={'similar_simulation':0,'failed_simulation':0,'discrepancies':0}

for sbml_model in timeseries_models:
    if os.path.isdir(os.path.join(file_path,sbml_model)):
        model_path=os.path.join(file_path,sbml_model)
        model_files=os.listdir(model_path)
        for file in model_files:
            if file.endswith("results.csv"):
                ground_truth_data=pd.read_csv(os.path.join(model_path,file))
                try:
                    time=ground_truth_data['time']
                    y_ground_truth=ground_truth_data.drop(columns=['time'])
                except:
                    time=ground_truth_data['Time']
                    y_ground_truth = ground_truth_data.drop(columns=['Time'])


        for file in model_files:
            if file.endswith(sbml_level):
                sbml_counter+=1
                try:
                    #calculate mse for jax
                    model = SBMLModel(os.path.join(model_path,file))
                    S = model._get_stoichiometric_matrix()

                    JaxKmodel = model.get_kinetic_model()
                    time=jnp.array(time)

                    ys_jax = JaxKmodel(ts=time,
                                   y0=model.y0,
                                   params=model.parameters)
                    ys_jax=pd.DataFrame(ys_jax,columns=S.index)

                    y_ground_truth=y_ground_truth[S.index]
                    mse_jax=ys_jax.subtract(y_ground_truth)
                    mse_jax=np.sum(np.array(mse_jax),axis=None)

                    if mse_jax <= 0.001:
                        l2v2_counter_jax['similar_simulation'] += 1
                    else:
                        l2v2_counter_jax['discrepancies'] += 1
                except:
                    l2v2_counter_jax['failed_simulation'] += 1
                    pass

                try:
                    time=np.array(time)

                    # same for libroadrunner
                    rr = roadrunner.RoadRunner(os.path.join(model_path, file))

                    rr.integrator.absolute_tolerance = 1e-10
                    rr.integrator.relative_tolerance = 1e-7
                    rr.integrator.initial_time_step = 1e-11
                    rr.integrator.max_steps = 300000

                    rr.simulate(time[0],time[-1],points=len(time))

                    ys_libroad_runner = rr.getSimulationData()


                    variables = ys_libroad_runner.colnames

                    variables = [i.replace("]", "") for i in variables]
                    variables = [i.replace("[", "") for i in variables]

                    ys_libroad_runner = pd.DataFrame(ys_libroad_runner, columns=variables)

                    ys_libroad_runner = ys_libroad_runner.drop(columns=['time'])
                    mse_libroad_runner = ys_libroad_runner.subtract(y_ground_truth)

                    mse_libroad_runner = np.sum(np.array(mse_libroad_runner), axis=None)
                    if mse_libroad_runner < 0.001:

                        l2v2_counter_libroadrunner['similar_simulation'] += 1
                    else:
                        l2v2_counter_libroadrunner['discrepancies'] += 1
                except:
                    l2v2_counter_libroadrunner['failed_simulation'] += 1



l2v2_counter_jax['total']=np.sum(list(l2v2_counter_jax.values()))
l2v2_counter_libroadrunner['total']=np.sum(list(l2v2_counter_libroadrunner.values()))
tests=pd.DataFrame({'jax_tests':l2v2_counter_jax,'libroad_runner_tests':l2v2_counter_libroadrunner}).T
date=datetime.datetime.today()

tests.to_csv(f"results/sbml_test_suite/level_{sbml_level}_ntests_{sbml_counter}_date_{date.day}_{date.month}_{date.year}.csv")



# print(f"number of succesful simulations (jaxkineticmodel) l2v1 model {l2v2_counter_jax / sbml_counter}")
# print(f"number of succesful simulation (libroadrunner) l2v1 models {l2v2_counter_libroadrunner / sbml_counter}")




