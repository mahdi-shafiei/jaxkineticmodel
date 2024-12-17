## Overview of the results described in Neural Ordinary Differential Equations Inspired Parameterization of Kinetic Models

### Overview of scripts needed to get results reported in paper [results](../results)
#### Initialization bounds latin hypercube sampling (Figure 2) in [EXP4_Glycolysis_Fitting_Datasets](../results/EXP4_Glycolysis_Fitting_Datasets)
If you want to reproduce the results from the 25 SBML models which are reported in [EXP1_initialization_bounds](../results/EXP1_initialization_bounds_lhs_V2), you need
to run [2207_trainer_script.py](2207_trainer_script.py). This can be run through the command line. For example
```
python3 scripts/2207_trainer_script.py -n Becker_Science2010.xml -t_end 15 -s 100 -i 1 -b 10 -o "results/EXP1_initialization_bounds_lhs/Becker_Science2010"
```

## Overview of notebooks generating the figures
Figure 2A,B,C: [Fig2ABC_parameterization_analysis_SBML.ipynb](experiments/Fig2ABC_parameterization_analysis_SBML.ipynb)
Figure 2D,E,F) [Fig2DEF_lossplots_and_timeseries_examples.ipynb](experiments/Fig2DEF_lossplots_and_timeseries_examples.ipynb)

Figure 3A,B,C,D) [Figure3ABCD_parameter_distance_to_optimum.ipynb](experiments/Figure3ABCD_parameter_distance_to_optimum.ipynb)

Figure 4) [Figure4_glucpulse_fit_assessment.ipynb](experiments/Figure4_glucpulse_fit_assessment.ipynb)


### Supporting information figures 
Figure SI4: [Figure4_glucpulse_fit_assessment.ipynb](experiments/Figure4_glucpulse_fit_assessment.ipynb)