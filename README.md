# glycolysis
To run models, following arguments can be passed through the command line

`-n --model_name` (required): used to construct the metabolite dictionary that is passed as an argument to the kinetic model
`-f --file` (required): the file with time series data (a N metabolites by T timepoints matrix).
`-p --parameter_sets` (required): Parameter sets that are used as an initial guess. Multiple parameter initializations can be run either sequentially or parallel.
`-w --work_dir` (required): working directory of where the kinetic model is stored.
`-d --weight_decay`: the weight decay for ODE training. Default=0.0
`-m --max_iter`: maximum number of iterations for the learning process. Default=1000.0
`-e --error_thresh`: the threshold on where to stop training. If this threshold is reached, we consider the problem "solved". Default=0.001
`-l --lr`: learning rate of the Adams optimizer. Default=1e-3
`-g --gpu`: whether to use GPU or not. So far not implemented properly.
`-j --jobs`: number of jobs to run parallel. Default=-1
`-o --output_dir`: directory to save results in. Default is "../results/"


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
