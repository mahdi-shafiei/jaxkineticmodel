
import sys
sys.path.append('/tudelft.net/staff-bulk/ewi/insy/DBL/plent/NeuralODEs/jax_neural_odes')
sys.path.append('/home/plent/Documenten/Gitlab/NeuralODEs/jax_neural_odes')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import diffrax
import jax.numpy as jnp
import jax
import equinox as eqx
import optax
from models.manual_implementations.glycolysis.glycolysis_model import glycolysis,NeuralODE
import argparse
jax.config.update("jax_enable_x64", True)
from source.parameter_estimation.training import create_log_params_means_centered_loss_func,log_transform_parameters,exponentiate_parameters
from scripts.analysis_helper_functions.helper_function_glycolysis_analysis import overwrite_y0_dict,prepare_glycolysis_model,update_parameters_by_dilution_rate,divide_parameters_by_dilution_rate
from scripts.analysis_helper_functions.helper_function_glycolysis_analysis import load_model_glucose_pulse_FF_datasets
from datetime import date

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-n',"--iterations",type=int,required=True,help="Number of iterations for gradient descent")
    parser.add_argument('-d',"--dataset1",type=str,required=True,help="datasets to train on")
    parser.add_argument('-k',"--dataset2",type=str,required=True,help="datasets to train on")



    args=parser.parse_args()
    n_iter=args.iterations
    date_today=date.today()
    dataset1_name=args.dataset1
    dataset2_name=args.dataset2




    expression_data=pd.read_csv("datasets/VanHeerden_Glucose_Pulse/PvanHoekExpressionData.csv",index_col=0)

    column_pairs = [
        ('D_HXK', 'HXK'),('D_PGI', 'PGI'),
        ('D_PFK', 'PFK'),('D_FBA', 'FBA'),
        ('D_TPI', 'TPI'),('D_GAPDH', 'GAPDH'),
        ('D_PGK', 'PGK'),('D_PGM', 'PGM'),
        ('D_ENO', 'ENO'),('D_PYK', 'PYK'),
        ('D_PDC', 'PDC'),('D_ADH', 'ADH')]

    interpolation_expression_dict = {}

    Ds=jnp.linspace(0,0.375,100)

    for D_col, col in column_pairs:
        interp_key=f"expr_interpolated_{col}"
        interpolation_expression_dict[interp_key] = diffrax.LinearInterpolation(ts=jnp.array(expression_data[D_col]),ys=jnp.array(expression_data[col]))\
        

    y0_dict={'ICG1P':0.064568,
        "ICT6P":0.093705,
        "ICtreh":63.312040,
        'ICglucose':0.196003,
        'ICG6P':0.716385,
        'ICF6P':0.202293,
        "ICFBP":0.057001,
        "ICDHAP":0.048571,
        "ICG3P":0.020586,
        "ICglyc":0.1,
        "ICGAP":0.006213,
        "ICBPG":0.0001,
        "IC3PG":2.311074,
        "IC2PG":0.297534,
        "ICPEP":1.171415,
        "ICPYR":0.152195,
        "ICACE":0.04,
        "ICETOH":10.0,
        "ECETOH":0,
        "ECglycerol":0.0,
        "ICNADH":0.0106,
        "ICNAD":1.5794,
        "ICATP":3.730584,
        "ICADP":1.376832,
        "ICAMP":0.431427,
        "ICPHOS":10,
        "ICIMP":0.100,
        "ICINO":0.100,
        "ICHYP":1.5,}

    y0=jnp.array(list(y0_dict.values()))
    metabolite_names=list(y0_dict.keys())
    params=pd.read_csv("parameter_initializations/Glycolysis_model/parameter_initialization_glycolysis_literature_values.csv",index_col=0).to_dict()['0']

    glycolyse_GP1,time_points_GP1,y0_GP1,dataset_GP1=load_model_glucose_pulse_FF_datasets(f"{dataset1_name}.csv",dilution_rate="0.1",y0_dict=y0_dict)
    glycolyse_GP2,time_points_GP2,y0_GP2,dataset_GP2=load_model_glucose_pulse_FF_datasets(f"{dataset2_name}.csv",dilution_rate="0.1",y0_dict=y0_dict)

    log_loss_func_GP1=jax.jit(create_log_params_means_centered_loss_func(glycolyse_GP1))
    log_loss_func_GP2=jax.jit(create_log_params_means_centered_loss_func(glycolyse_GP2))



    grads_GP1=jax.jit(jax.grad(log_loss_func_GP1,0))
    grads_GP2=jax.jit(jax.grad(log_loss_func_GP2,0))


    datasets={"GP1":jnp.array(dataset_GP1),"GP2":jnp.array(dataset_GP2)}
    time_points={"GP1":jnp.array(time_points_GP1),"GP2":jnp.array(time_points_GP2)}

    loss_per_iter1=[]
    loss_per_iter2=[]
    loss_per_iter3=[]
    parameters_step_list=[]


    lr=1e-3
    optimizer = optax.adabelief(lr)
    clip_by_global=optax.clip_by_global_norm(np.log(4))
    optimizer = optax.chain(optimizer,clip_by_global)
    opt_state = optimizer.init(params)
    params_init=params

    ys=datasets
    ts=time_points

    for step in range(n_iter):
        grads={}

        
        log_params=log_transform_parameters(params_init)
        loss1=log_loss_func_GP1(log_params,ts['GP1'],ys['GP1'])
        grads1=grads_GP1(log_params,ts['GP1'],ys['GP1'])


        loss2=log_loss_func_GP2(log_params,ts['GP2'],ys['GP2'])
        grads2=grads_GP2(log_params,ts['GP2'],ys['GP2'])



        for key in grads1.keys():
            grads[key]=(grads1[key]+grads2[key])/2#(grads1[key]+grads2[key]+grads3[key])/3
        



        updates,opt_state=optimizer.update(grads,opt_state)
        #we perform updates in log space, but only return params in lin space
        log_params = optax.apply_updates(log_params, updates)
        lin_params = exponentiate_parameters(log_params) 
        params_init=lin_params

        loss_per_iter1.append(float(loss1))
        loss_per_iter2.append(float(loss2))


        if step% 50==0:
            print(f"Step {step}, Loss {loss1+loss2}")
            parameters_step_list.append(params_init)


    losses_per_iterations={"loss_per_iter1":loss_per_iter1,"loss_per_iter2":loss_per_iter2}
    losses_per_iterations_added=pd.DataFrame(losses_per_iterations)
    output_names=f"{date_today}_losses_{dataset1_name}_{dataset2_name}_datasets_niter={n_iter}.csv"
    losses_per_iterations_added.to_csv(f"results/EXP4_Glycolysis_Fitting_Datasets/{output_names}")


    parameters_step_list=pd.DataFrame(parameters_step_list).to_csv(f"results/EXP4_Glycolysis_Fitting_Datasets/{date_today}_parameters_step50_{dataset1_name}_{dataset2_name}_{n_iter}.csv")
    final_params=pd.DataFrame(pd.Series(params_init))
    final_params.to_csv(f"results/EXP4_Glycolysis_Fitting_Datasets/{date_today}_optimized_parameters_{dataset1_name}_{dataset2_name}_{n_iter}.csv")



if __name__=="__main__":
    main()