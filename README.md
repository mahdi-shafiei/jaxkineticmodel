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


### Example for running Bioprocess:

`python main.py -n bioprocess -f "rawdata_batch_bioprocess.csv" -p "Batch_Bioprocess_parametersets.csv" -w "../batch_bioprocess/."  -m 100`

### Example for running MAPK signalling:
`python main.py -n mapk -f "rawdata_mapk_signalling.csv" -p "mapk_parametersets.csv" -w "../mapk_signalling/" -m 100`
