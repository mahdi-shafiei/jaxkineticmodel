import os
import sys
import time
import re
from torchdiffeq import odeint,odeint_adjoint
sys.path.append("../")
from functions.symplectic_adjoint.torch_symplectic_adjoint import odeint_symplectic_adjoint
import argparse

from functions.kinetic_mechanisms.KineticMechanisms import *
from functions.kinetic_mechanisms.KineticModifiers import *
from functions.kinetic_mechanisms.KineticMechanismsCustom import *

from torch import nn
from scipy.interpolate import CubicSpline
import pandas as pd

from models.pCA_model_p20.pCA_fluxes_simpler import *
from trainer import Trainer




def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-n',"--name",type=str,required=True,help="Name of the training process that is used to save the file")
    parser.add_argument('-p',"--parameter_sets",type=str,required=True,help="Parameter set file")
    parser.add_argument('-d',"--data",type=str,required=True,help="time series data (NxT dataframe) used to fit")
    parser.add_argument('-o',"--output_dir",type=str,required=True,help="output directory for loss per iteration and the optimized parameters")

    args=parser.parse_args()


    data=pd.read_csv(args.data,index_col=0)
    
    #load the online data (required for fitting)
    online_data=pd.read_excel("../data/pCA_timeseries/Coumaric acid fermentation data for Paul van Lent.xlsx",sheet_name=3,header=2)

    metabolites_names=data.index.to_list()
    indices=np.arange(0,len(metabolites_names))
    metabolites=dict(zip(metabolites_names,indices))


    parameter_sets=pd.read_csv(args.parameter_sets,index_col=0)
    
    error_thresh=0.005
    max_iter=2000
    gpu=False
    lr=1e-3

    output_dir=args.output_dir
    loss_per_iteration=[]
    optimized_parameters=[]
    a=time.time()

    for i in range(np.shape(parameter_sets)[0]):
        parameter_dict=dict(parameter_sets.iloc[i,:])    
        fluxes=create_fluxes(parameter_dict)

        
        #this is specific to the pca model. The glucose uptake rate is a given
    
        glucose_feed=online_data['Rate feed C (carbon source) (g/h)'].values[:-1]
        t=online_data['Age (h)'].values[:-1]
        fluxes['vGluc_Feed']=glucose_rate_polynomial_approximation(t,glucose_feed,N=40)


        # oxygen=online_data['Dissolved oxygen (%)'].values[:2869]*0.28 #mmol/L
        # t=online_data['Age (h)'].values[:2869]
        # fluxes['vOx_uptake']=oxygen_uptake_polynomial_approximation(t,oxygen,3)


        model=pCAmodel(fluxes,metabolites)



        loss_function_metabolites=[1,2,5,6,7]
        loss_function_metabolites=[6]
        trainer=Trainer(model,data,loss_func_targets=loss_function_metabolites,
                            max_iter=max_iter,err_thresh=error_thresh,lr=lr,scaling=True,rtol=1e-3,atol=1e-7) #remove scaling here and add as additional step
        
        #make sure that scaling is performed for glucose (since in data yscale=0, but in first simulation it is almost 400. )
        # I will for now arbitrarily set this to 30, because this is an important property to be learned 



        try:
            trainer.train()
            loss_per_iteration.append(trainer.get_loss_per_iteration)

            #the fuck am i doing here
            named_parameters=dict(trainer.ode.named_parameters())
            named_parameters={i:float(named_parameters[i]) for i in named_parameters}
            optimized_parameters.append(named_parameters)

    
        except: 
            print("cannot solve ODEs, continue")
            continue
    

    param_match=re.findall("_[a-z]*_\d*.csv",args.parameter_sets)    
    param_match=param_match[0].strip(".csv")

    output_filename_loss=output_dir+args.name+"_loss_per_iteration"+param_match+".csv"
    output_filename_optim_params=output_dir+args.name+"_optim_param"+param_match+".csv"

    # loss_dictionary=dict(zip(index,loss_per_iteration))
    # # loss_per_iteration=np.array(loss_per_iteration,dtype=object).reshape(np.shape(loss_per_iteration)[0],-1)
    loss_per_iteration=pd.DataFrame(loss_per_iteration).T
    loss_per_iteration.to_csv(output_filename_loss)
    
    # names_parameters=list(parameter_sets.iloc[0,:].keys())
    # optimized_parameters=pd.DataFrame(torch.Tensor(optimized_parameters).detach().numpy(),columns=names_parameters)
    optimized_parameters=pd.DataFrame(optimized_parameters).T
    print(optimized_parameters)

    optimized_parameters.to_csv(output_filename_optim_params)
    b=time.time()
    print(b-a)

if __name__=="__main__":
    main()