import diffrax
from diffrax import diffeqsolve, ODETerm, Dopri5,SaveAt
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial



class JaxKineticModel():
    def __init__(self,v,S,flux_point_dict,species_names):#params,
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

    def __call__(self,t,y,params):
        """I explicitly add params to call for gradient calculations. Find out whether this is actually necessary"""

        # @partial(jax.jit, static_argnums=2)

        def apply_func(i,y,flux_point_dict):
            y=y[flux_point_dict]
            species=self.species_names[flux_point_dict]
            parameters=params[i]
            y=dict(zip(species,y))
            eval_dict={**y,**parameters}
            vi=self.func[i](**eval_dict)
            return vi
        # Vectorize the application of the functions
        indices = np.arange(len(self.func))

        # v = jax.vmap(lambda i:apply_func(i,y,self.flux_point_dict[i]))(indices)
        v=jnp.stack([apply_func(i,y,self.flux_point_dict[i]) for i in indices]) #perhaps there is a way to vectorize this in a better way
        dY = jnp.matmul(self.stoichiometry, v) #dMdt=S*v(t)

        return dY        

class NeuralODE():
    func: JaxKineticModel
    def __init__(self,
                 v,
                 S,
                 flux_point_dict,
                 species_names):
        super().__init__()
        self.func=JaxKineticModel(v,S,flux_point_dict,species_names)

    def __call__(self,ts,y0,params):
        solution = diffrax.diffeqsolve(
        diffrax.ODETerm(self.func),
        diffrax.Kvaerno5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0,
        args=params,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys