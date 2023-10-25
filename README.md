# glycolysis
To run models, following arguments can be passed through the command line

-n --model_name (required): used to construct the metabolite dictionary that is passed as an argument to the kinetic model\n
-f --file (required): the file with time series data (a N metabolites by T timepoints matrix).\n
-p --parameter_sets (required): Parameter sets that are used as an initial guess. Multiple parameter initializations can be run either sequentially or parallel.\n
-w --work_dir (required): working directory of where the kinetic model is stored.\n

-d --weight_decay: the weight decay for ODE training. Default=0.0\n
-m --max_iter: maximum number of iterations for the learning process. Default=1000.0\n
-e --error_thresh: the threshold on where to stop training. If this threshold is reached, we consider the problem "solved". Default=0.001\n
-l --lr: learning rate of the Adams optimizer. Default=1e-3\n
-g --gpu: whether to use GPU or not. So far not implemented properly.\n
-j --jobs: number of jobs to run parallel. Default=-1\n
-o --output_dir: directory to save results in. Default is "../results/"\n


### Example for running Bioprocess:

python main.py -n bioprocess -f "rawdata_batch_bioprocess.csv" -p "Batch_Bioprocess_parametersets.csv" -w "../batch_bioprocess/."  -m 100

### Example for running MAPK signalling:

