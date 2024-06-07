import diffrax
from diffrax import diffeqsolve, ODETerm, Dopri5,SaveAt
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from sbml_load import construct_param_point_dictionary,separate_params
from sbml_load import time_dependency_symbols

jax.config.update("jax_enable_x64", True)


class JaxKineticModel():
    def __init__(self,v,
                 S,
                 flux_point_dict,
                 species_names,
                 reaction_names):#params,
        """Initialize given the following arguments:
        v: the flux functions given as lambidified jax functions,
        S: a stoichiometric matrix. For now only support dense matrices, but later perhaps add for sparse
        params: kinetic parameters
        flux_point_dict: a dictionary for each vi that tells what the corresponding metabolites should be in y. Should be matched to S.
        ##A pointer dictionary?
        """
        super().__init__()
        self.func=v
        self.stoichiometry=S
        # self.params=params
        self.flux_point_dict=flux_point_dict #this is ugly but wouldnt know how to do it another wa
        self.species_names=np.array(species_names)
        self.reaction_names=np.array(reaction_names)


    def __call__(self,t,y,args):
        """I explicitly add params to call for gradient calculations. Find out whether this is actually necessary"""
        params=args[0]
        local_params=args[1]
        time_dict=args[2](t)


  
        # @partial(jax.jit, static_argnums=2)
        
        
        def apply_func(i,y,flux_point_dict,local_params,time_dict):
            if len(flux_point_dict)!=0:
                y=y[flux_point_dict]
                species=self.species_names[flux_point_dict]
                y=dict(zip(species,y))
            else:
                y={}
                
            parameters=params[i]

            
            
            eval_dict={**y,**parameters,**local_params,**time_dict}

            vi=self.func[i](**eval_dict)
            return vi
        # Vectorize the application of the functions
        indices = np.arange(len(self.func))

        # v = jax.vmap(lambda i:apply_func(i,y,self.flux_point_dict[i]))(indices)

        v=jnp.stack([apply_func(i,y,self.flux_point_dict[i],
                                local_params[i],
                     time_dict[i]) for i in self.reaction_names]) #perhaps there is a way to vectorize this in a better way
        dY = jnp.matmul(self.stoichiometry, v) #dMdt=S*v(t)
        return dY        





class NeuralODE():
    func: JaxKineticModel
    def __init__(self,
                 v,
                 S,
                 met_point_dict,
                 v_symbol_dictionaries):
        super().__init__()


        self.func=JaxKineticModel(v,
                                  jnp.array(S),
                                  met_point_dict,
                                  list(S.index),
                                  list(S.columns))
        self.reaction_names=list(S.columns)
        self.v_symbol_dictionaries=v_symbol_dictionaries
        self.Stoichiometry=S
        
        def wrap_time_symbols(t):
            time_dependencies=time_dependency_symbols(v_symbol_dictionaries,t)
            return time_dependencies
        
        self.time_dict=jax.jit(wrap_time_symbols)

    def __call__(self,ts,y0,params):

        global_params,local_params=separate_params(params)



        #ensures that global params are loaded flux specific (necessary for jax)
        global_params=construct_param_point_dictionary(self.v_symbol_dictionaries,
                                                self.reaction_names,
                                                global_params) #this is required, 



        solution = diffrax.diffeqsolve(
        diffrax.ODETerm(self.func),
        diffrax.Kvaerno5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0,
        args=(global_params,local_params,self.time_dict),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=15000
        )
        return solution.ys