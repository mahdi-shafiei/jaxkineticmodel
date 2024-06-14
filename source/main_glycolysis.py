#data prep + create fluxes + call trainer
import sys

sys.path.append('../models/glycolysis/')
from fluxes import create_fluxes

sys.path.append('../models/glycolysis/')
from glycolysis import *
from trainer import Trainer
import argparse
import pandas as pd
import time
import re as re
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-n',"--name",type=str,required=True,help="Name of the training process that is used to save the file")
    parser.add_argument('-d',"--data",type=str,required=False,help="Name of the input file for the timeseries data")
    parser.add_argument('-ic',"--extra_ic",type=str,required=False,help="Name of the input file for the initial conditions")
    parser.add_argument('-e', "--error_thresh", type=float, default=5.0E-3, help="Error threshold for the loss function")
    parser.add_argument('-m', "--max_iter", type=int, default=10, help="Maximum number of iterations")
    parser.add_argument('-o',"--output_dir",type=str,required=True,help="Output directory for loss per iteration and the optimized parameters")

    
    
   
    args=parser.parse_args()
    output_dir=args.output_dir
    time_series_data_name = args.data
    initial_conditions_names = args.extra_ic
    #"../data/glycolysis_timeseries/FF1_timeseries_format.csv"
    
    data=pd.read_csv(time_series_data_name,index_col=0) 
    
    
    # Filter the data
    to_be_removed = ['ICCIT', 'ICisoCIT', 'ICaKG', 'ICSUCC', 'ICglyc',
                     'ICFUM', 'ICMAL', 'IC6PG', 'ICRibu5P', 
                     'ICR5P', 'ICX5P', 'ICS7P', 'ICE4P', 'ICM6P']    
    
    data = data.drop(to_be_removed)
    data_metabolite_names =  data.index.to_list()
    
    
    print(len(data_metabolite_names))
    #"../data/glycolysis_timeseries/initial_conditions.csv"
    other_metabolites_ic = pd.read_csv(initial_conditions_names, index_col=0) 
    combined_data = pd.concat([data, other_metabolites_ic], ignore_index=True)
    
    metabolite_names = data_metabolite_names + other_metabolites_ic.index.to_list()
    metabolite_indices = {metabolite: i for i, metabolite in enumerate(metabolite_names)} 
  
    print(metabolite_names)
    #Not all metabolites have complete time-series
    loss_function_targets = []
    for metabolite in data_metabolite_names:
        if data.loc[metabolite].isnull().any():
            continue  # Skip this metabolite if there are NaN values
        else:
            loss_function_targets.append(metabolite)
            
        
         
        
    fluxes = create_fluxes()
    glycolysis = Glycolysis(fluxes=fluxes, metabolites=metabolite_indices) #Initialize the model
    trainer = Trainer(glycolysis, combined_data, loss_function_targets) 
    start_time=time.time()
    loss_per_iteration=[]
    optimized_parameters=[]

    try:
        trainer.train()
        loss_per_iteration.append(trainer.get_loss_per_iteration)

        #the fuck am i doing here
        named_parameters=dict(trainer.ode.named_parameters())
        named_parameters={i:float(named_parameters[i]) for i in named_parameters}
        optimized_parameters.append(named_parameters)     
    except: 
        print("cannot solve ODEs, continue")
     

    
    # param_match=re.findall("_[a-z]*_\d*.csv",args.parameter_sets)    
    # param_match=param_match[0].strip(".csv")

    # output_filename_loss=output_dir+args.name+"_loss_per_iteration"+param_match+".csv"
    # output_filename_optim_params=output_dir+args.name+"_optim_param"+param_match+".csv"

    
    loss_per_iteration=pd.DataFrame(loss_per_iteration).T
    # loss_per_iteration.to_csv(output_filename_loss)
   
    optimized_parameters=pd.DataFrame(optimized_parameters).T

    # optimized_parameters.to_csv(output_filename_optim_params)
    end_time=time.time()
    print("Runtime: ", end_time - start_time)

if __name__ == "__main__":
    main()