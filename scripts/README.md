## Overview of the results described in Neural Ordinary Differential Equations Inspired Parameterization of Kinetic Models
We used a HPC cluster to run all experiments.

### Overview of scripts needed to get results reported in paper [results](../results)
#### Initialization bounds latin hypercube sampling (Figure 2) in [EXP4_Glycolysis_Fitting_Datasets](../results/EXP4_Glycolysis_Fitting_Datasets)
If you want to reproduce the results from the 25 SBML models which are reported in [EXP1_initialization_bounds](../results/EXP1_initialization_bounds_lhs_V2), you need
to run [2207_trainer_script.py](simulated_data_trainer_script.py). This can be run through the command line. For example:
```
python3 scripts/simulated_data_trainer_script.py -n Becker_Science2010.xml -t_end 15 -s 100 -i 1 -b 10 -o "results/EXP1_initialization_bounds_lhs/Becker_Science2010"
```


#### Glycolysis model fitting [EXP4_Glycolysis_Fitting_Datasets](../results/EXP4_Glycolysis_Fitting_Datasets)
For the model shown in figure 4, run the following command for 8000 iterations of gradient descent. This should take roughly 4 hours.
```
python3 scripts/simulated_data_trainer_script.py -n 8000 -d "datasets/VanHeerden_Glucose_Pulse/FF1_timeseries_format.csv"
```
Fitting multiple datasets as described in the Supporting Information can be performed using the [1709_train_gp_twodatasets.py](experiments/1709_train_gp_twodatasets.py)
and [1709_train_gp_all_datasets.py](experiments/1709_train_gp_all_datasets.py) scripts in a similar fashion.

## Overview of notebooks generating the figures
Figure 2A,B,C: [Fig2ABC_parameterization_analysis_SBML.ipynb](experiments/Fig2ABC_parameterization_analysis_SBML.ipynb)
Figure 2D,E,F) [Fig2DEF_lossplots_and_timeseries_examples.ipynb](experiments/Fig2DEF_lossplots_and_timeseries_examples.ipynb)

Figure 3A,B,C,D) [Figure3ABCD_parameter_distance_to_optimum.ipynb](experiments/Figure3ABCD_parameter_distance_to_optimum.ipynb)

Figure 4) [Figure4_glucpulse_fit_assessment.ipynb](experiments/Figure4_glucpulse_fit_assessment.ipynb)


### Supporting information figures 
Figure SI4: [Figure4_glucpulse_fit_assessment.ipynb](experiments/Figure4_glucpulse_fit_assessment.ipynb)