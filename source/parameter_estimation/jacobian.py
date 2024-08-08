
# sys.path.append("/home/plent/Documenten/Gitlab/NeuralODEs/jax_neural_odes")

from source.load_sbml.sbml_load import *
from source.load_sbml.sbml_model import SBMLModel
import jax
import matplotlib.pyplot as plt


class Jacobian:
    def __init__(self, model):
        self.model=model
        self.JaxKmodel = jax.jit(self.model.get_kinetic_model())
        


    def make_jacobian(self, y0, params):
        global_params, local_params = separate_params(params)

        # Construct the global parameter dictionary
        global_params_dict = construct_param_point_dictionary(
            self.JaxKmodel.v_symbol_dictionaries,
            self.JaxKmodel.reaction_names,
            global_params
        )

        # Compute the function value
        y = self.JaxKmodel.func(t=0, y=y0, args=(global_params_dict, local_params, self.JaxKmodel.time_dict))
        return y

    def wrapped_make_jacobian(self, y0, global_params, local_params):
        params = {**global_params, **local_params}
        return self.make_jacobian(y0, params)

    def compile_jacobian(self):
        # Define the function for which the Jacobian is computed
        f = lambda y0, global_params, local_params: self.wrapped_make_jacobian(y0, global_params, local_params)
        
        # Compute the Jacobian
        J = jax.jacfwd(f, argnums=0)
        
        # JIT compile the Jacobian function
        J_compiled = jax.jit(J)
        return J_compiled
    

    def filter_stable_parameter_sets(self,compiled_jacobian,y_t,parameter_initializations):
        """Given an initial sampling, we filter out parameter sets that lead to unstable results for the initialiation. 
        This ensures higher initialization success for training and is cheap to evaluate.

        Input:
        y_i: the state at time=t. One note: initial values might not be good for a stability check. You might prefer to take the final point in a dataset!

        parameter_initializations: a pandas dataframe with parameter initializations from a latin hypercube sampling
        rule: type of filtering to perform. Right now we support three rules: 
        1) "stability"          all Re(λ_i)=<0+epsilon: (stable dynamics),
        """
        eigvals=[]
        for i in range(np.shape(parameter_initializations)[0]):
            init_param=parameter_initializations.iloc[i,:].to_dict()
            # print(init_param)
            init_global_params, init_local_params = separate_params_jac(init_param)
            # print(init_local_params)
            eigvals.append(jnp.linalg.eigvals(compiled_jacobian(y_t,init_global_params,init_local_params)))
        eigvals=np.array(eigvals)
        epsilon=0.001

        negative_eigenvalues=eigvals.real<=(0+epsilon)
        stable_parameters_indices=np.where(np.sum(negative_eigenvalues,axis=1)==len(negative_eigenvalues[0,:]))[0]
        filtered_parameters=parameter_initializations.iloc[stable_parameters_indices,:]
        return filtered_parameters
    
    def filter_oscillations(self,compiled_jacobian,y_t,parameter_initializations,period_bounds):
        """
        Filtering a LHS based on oscillatory behavior.
        Input:
        compiled jacobian
        y_t: values at time t 

        parameter_initializations: a pandas dataframe with parameter initializations from a latin hypercube sampling
        period bounds (list [lb,ub] in proper units): used to filter for dynamics where an estimate of the period of damped oscillations is available. Imaginary eigenvalus are check according to
        2pi/T_lb <=Im(λ_i)<= 2pi/Tub.
        """
        eigvals=[]
        for i in range(np.shape(parameter_initializations)[0]):
            init_param=parameter_initializations.iloc[i,:].to_dict()
            init_global_params, init_local_params = separate_params(init_param)
            eigvals.append(jnp.linalg.eigvals(compiled_jacobian(y_t,init_global_params,init_local_params)))
        eigvals=np.array(eigvals)
        epsilon=0.0001

        max_period=(2*np.pi)/np.max(np.abs(eigvals.imag)+epsilon,axis=1)


        oscillation_parameter_indices2=np.where(max_period>=period_bounds[0])[0]
    
        oscillation_parameter_indices1=np.where(max_period<=period_bounds[1])[0]
        oscillation_parameter_indices=np.intersect1d(oscillation_parameter_indices1,oscillation_parameter_indices2)
   
                                               
                                               #and max_imag_eigvals>=(2*np.pi/period_bounds[0]))[0]
        filtered_parameters=parameter_initializations.iloc[oscillation_parameter_indices,:]
        return filtered_parameters

            # elif rule == "stability_oscillation":
        #     stable_negative = eigvals.real < (0 + epsilon)
        #     positive_imaginary = eigvals.imag > 0

        #     stable_parameters_indices=np.where(np.sum(stable_negative,axis=1)==len(stable_negative[0,:]))[0]
        #     oscillation_parameter_indices=np.where(np.sum(positive_imaginary,axis=1)>=1)[0]
        #     indices=np.intersect1d(stable_parameters_indices,oscillation_parameter_indices)
        #     filtered_parameters=parameter_initializations.iloc[indices,:]
            


        # elif rule == "oscillation":
        #     logger.info("Periodicity not implemented yet")
        #     close_to_zero_real = np.abs(eigvals.real) <= epsilon
        #     positive_imaginary = eigvals.imag > 0
            
        #     stable_parameters_indices=np.where(np.sum(close_to_zero_real,axis=1)==len(close_to_zero_real[0,:]))[0]
        #     oscillation_parameter_indices=np.where(np.sum(positive_imaginary,axis=1)>=1)[0]
        #     indices=np.intersect1d(stable_parameters_indices,oscillation_parameter_indices)
        #     filtered_parameters=parameter_initializations.iloc[indices,:]
        #     # periodic_indices = np.where(np.sum(close_to_zero_real & positive_imaginary, axis=1) == eigvals.shape[1])[0]
        #     # filtered_parameters = parameter_initializations.iloc[periodic_indices, :]
        # else:
        #     raise ValueError("Unsupported filtering rule. Supported rules are: 'stability', 'stability_oscillation', 'oscillation'.")