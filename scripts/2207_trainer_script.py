import matplotlib.pyplot as plt
import os
import sys, os
sys.path.append('/home/plent/Documenten/Gitlab/NeuralODEs/jax_neural_odes')
# sys.path.append('/home/plent/Documenten/Gitlab/NeuralODEs')
from source.load_sbml.sbml_load import *
from source.load_sbml.sbml_model import SBMLModel
jax.config.update("jax_enable_x64", True)
from source.utils import get_logger
from source.parameter_estimation.initialize_parameters import *
import optax
from source.parameter_estimation.training import *
import time
import argparse
from source.parameter_estimation.jacobian import *

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-n',"--name",type=str,required=True,help="Name of the training process that is used to save the file")
    parser.add_argument('-t_end',"--final_time_point",type=float,required=True,help="Final timepoint for evaluation")
    parser.add_argument('-s',"--n_parameters",type=int,required=True,help="number of parameters to initialize")
    parser.add_argument('-i',"--id_run",type=int,required=True,help="identifier for when doing different runs")
    # parser.add_argument('-d',"--data",type=str,required=True,help="time series data (NxT dataframe) used to fit")
    # parser.add_argument('-o',"--output_dir",type=str,required=True,help="output directory for loss per iteration and the optimized parameters")
    # model_name="Raia_CancerResearch.xml"
    
    args=parser.parse_args()

    model_name=args.name
    # model_name="Berzins2022 - C cohnii glucose and glycerol.xml"
    filepath="models/sbml_models/working_models/"+model_name
    lr=1e-3
    N=args.n_parameters
    lb=0.01
    ub=100
    # id="lhs_"+"N="+str(N)+"run_1"
    run_id="run_"+str(args.id_run)
    id="lhs_"+"N="+str(N)+run_id+"_log_update"
    loss_threshold=1e-3
    
    ts=jnp.linspace(0,args.final_time_point,10)

    dataset,params=generate_dataset(filepath,ts)
    


    bounds=generate_bounds(params,lower_bound=lb,upper_bound=ub)
    # uniform_parameter_initializations=uniform_sampling(bounds,N)
    lhs_parameter_initializations=latinhypercube_sampling(bounds,40000)

    #filter parameters based on stability
    model=SBMLModel(filepath)
    jacobian_object=Jacobian(model)
    logger.info("Jacobian Compiled")
    compiled_jacobian=jacobian_object.compile_jacobian()

    y_t=jnp.array(dataset.iloc[0,:])
    lhs_parameter_initializations=jacobian_object.filter_stable_parameter_sets(compiled_jacobian,y_t,lhs_parameter_initializations)

    print("number of feasible parameters: ", np.shape(lhs_parameter_initializations))
    
    if np.shape(lhs_parameter_initializations)[0]<args.n_parameters:
        choices=np.arange(0,np.shape(lhs_parameter_initializations)[0])
        indices=np.random.choice(a=choices,size=np.shape(lhs_parameter_initializations)[0])
    else:    ## downsampling
        choices=np.arange(0,np.shape(lhs_parameter_initializations)[0])

        indices=np.random.choice(a=choices,size=args.n_parameters)


    lhs_parameter_initializations=lhs_parameter_initializations.iloc[indices,:]

    print("lhs samples:  ",np.shape(lhs_parameter_initializations))

    save_dataset(model_name,dataset)
    save_parameter_initializations(model_name,lhs_parameter_initializations,id=id)


    ### If time
    
    JaxKmodel = model.get_kinetic_model()
    JaxKmodel = jax.jit(JaxKmodel)
    # #parameters are not yet defined
    params = get_global_parameters(model.model)
    params = {**model.local_params, **params}

    print("# params",len(params))

    # log_loss_func=jax.jit(create_log_params_log_loss_func(JaxKmodel))
    log_loss_func=jax.jit(create_log_params_means_centered_loss_func(JaxKmodel))
    loss_func=jax.jit(create_loss_func(JaxKmodel))





    @jax.jit
    def update(opt_state,params,ts,ys):
        """Update rule for the gradients for parameters"""
        loss=loss_func(params,ts,ys)
        grads=jax.jit(jax.grad(loss_func,0))(params,ts,ys) #loss w.r.t. parameters
        updates,opt_state=optimizer.update(grads,opt_state)
        params = optax.apply_updates(params, updates)
        return opt_state,params,loss,grads



    @jax.jit
    def update_log(opt_state,params,ts,ys):
        """Update rule for the gradients for log-transformed parameters. Can only be applied
        to nonnegative parameters"""
        log_params=log_transform_parameters(params)
        loss=log_loss_func(log_params,ts,ys)

        grads=jax.jit(jax.grad(log_loss_func,0))(log_params,ts,ys) #loss w.r.t. parameters
        updates,opt_state=optimizer.update(grads,opt_state)

        #we perform updates in log space, but only return params in lin space
        log_params = optax.apply_updates(log_params, updates)
        lin_params = exponentiate_parameters(log_params) 
        return opt_state,lin_params,loss,grads
            



    loss_per_iteration_dict={}
    optimized_parameters_dict={}
    global_norm_dict={}
    start=time.time()
    for init in range(np.shape(lhs_parameter_initializations)[0]):
        print(f"init {init}")
        
        params_init=lhs_parameter_initializations.iloc[init,:].to_dict()
        optimizer = optax.adabelief(lr)

        clip_by_global=optax.clip_by_global_norm(np.log(4))
        optimizer = optax.chain(optimizer,clip_by_global)
        opt_state = optimizer.init(params_init)


        loss_per_iter=[]
        gradient_norms=[]

        try:
            for step in range(3000):
                opt_state,params_init,loss,grads=update_log(opt_state,params_init,ts,jnp.array(dataset))
                # opt_state,params_init,loss,grads=update(opt_state,params_init,ts,jnp.array(dataset))

                gradient_norms.append(global_norm(grads))
                loss_per_iter.append(loss)

                if loss<loss_threshold:
                    #stop training because loss reached threshold
                    print("loss threshold reached")
                    loss_per_iteration_dict[init]=loss_per_iter
                    global_norm_dict[init]=gradient_norms
                    optimized_parameters_dict[init]=params_init
                    break


                if step% 10==0:
                    # print(f"global norm: {global_norm(grads)}")
                    print(f"Step {step}, Loss {loss}")

            
            loss_per_iteration_dict[init]=loss_per_iter
            optimized_parameters_dict[init]=params_init
            global_norm_dict[init]=gradient_norms

        except:
            print(f"init {init} could not be optimized")
            loss_per_iteration_dict[init]=loss_per_iter
            loss_per_iteration_dict[init].append(-1)



            global_norm_dict[init]=gradient_norms
            global_norm_dict[init].append(-1)
            continue
    end=time.time()
    print("time it took",end-start)


    losses = pd.DataFrame({ key:pd.Series(value) for key, value in loss_per_iteration_dict.items() })
    # losses= pd.DataFrame(loss_per_iteration_dict)

    optimized_parameters = pd.DataFrame({ key:pd.Series(value) for key, value in optimized_parameters_dict.items() })
    # optimized_parameters=pd.DataFrame(optimized_parameters_dict)
    # norms=pd.DataFrame(global_norm_dict)
    norms = pd.DataFrame({ key:pd.Series(value) for key, value in global_norm_dict.items() })



    save_losses(model_name,losses,id=id)
    save_optimized_params(model_name,optimized_parameters,id=id)
    save_norms(model_name,norms,id=id)

if __name__=="__main__":
    main()