
### SBML loading models (Leon)
1. Updated version of the loader can be found in functions/load_sbml
2. The loader of sbml models is now working for most models. Models that are not working essentially fail to simulate (so there is nothing wrong with SBML loader class, but it could be attributed to too many parameters, too slow for the solver, etc..)
3. I have changed the structure of the loader a little bit and added some extra steps (will prepare for tomorrow).
4. I have added a simple test.py to see which models load and etc. My idea is to use this to time the integration steps and loading functions. I also added a small test script of tellurium, which is specifically optimized for simulating sbml models. I do not expect to beat this run time (since it is written in c) but getting closer would be nice. 


### Running a training process on the cluster
The procedure for training a kinetic model is considered in the run_training.sh bash file. This script consists of:
1. A parameter initialization method:
`-n --name`: name that should be given to the output parameter sets 
`-m --method`: sampling method to be used: latin_hypercube or uniform. 
`-s --size`: number of initial guesses that should be considered.
`-d --divide_sets`: when running on the cluster, we would like to divide the workload between multiple jobs. So if -d=5, the number of files generated is size/divide. 
If not set, then divide is the same as size.
`-f -- bound_file`: a file with the lower and upper bound of the individual parameters, where a sample should be taken from.
`-o --output_dir`: a directory, where initial parameter guesses are saved.


2. A loop over the initial guesses, which are loaded into cluster_train_model.py. Requires as arguments
`-n --name:` name of the training process that is used to save the file. 
`-p --parameter_sets`: file containing the parameter sets described above
`-d --data: time series data (NxT dataframe) used to fit
`-o --output_dir: output directory for loss per iteration and the optimized parameters 

Optional parameters
`-d --weight_decay`: the weight decay used by the trainer optimizer

On the cluster, the bash file consists of a call of sbatch instead of the cluster_train_model.py. 
Arguments passed to sbatch are then simply used within the sbatch script that is used to send jobs to the cluster. 
