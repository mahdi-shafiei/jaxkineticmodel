import jax.numpy as jnp


class Jax_Irrev_MM_Uni:
    """Irreversible Michaelis-Menten kinetics (uni-substrate), adapted to JAX."""
    def __init__(self,
                 substrate:str,
                 vmax: str, 
                 km_substrate: str):

        # Initialize the parameter names
        self.substrate=substrate
        self.vmax = vmax
        self.km_substrate = km_substrate

    def __call__(self, eval_dict):
        # Extract parameter values from the dictionary
        substrate=eval_dict[self.substrate]
        vmax = eval_dict[self.vmax]
        km_substrate = eval_dict[self.km_substrate]

        nominator = vmax * (substrate / km_substrate)
        denominator = 1 + (substrate / km_substrate)

        return nominator/denominator



class Jax_Facilitated_Diffusion():
    """facilitated diffusion formula, taken from 
    Lao-Martil, D., Schmitz, J. P., Teusink, B., & van Riel, N. A. (2023). Elucidating yeast glycolytic dynamics at steady state growth and glucose pulses through
     kinetic metabolic modeling. Metabolic engineering, 77, 128-142.
     
     Class works slightly different than in torch. We have to simply include the names of the reaction parameter so that call recognizes them"""
    def __init__(self,substrate_extracellular:str,
                 product_intracellular:str,
                 vmax:str,
                 km_internal:str,
                 km_external:str):
            super(Jax_Facilitated_Diffusion)
            self.substrate=substrate_extracellular
            self.product=product_intracellular
            self.vmax=vmax
            self.km_internal=km_internal
            self.km_external=km_external

    def __call__(self,eval_dict):

        vmax=eval_dict[self.vmax]
        
        km_external=eval_dict[self.km_external]
        km_internal=eval_dict[self.km_internal]
        substrate=eval_dict[self.substrate]
        product=eval_dict[self.product]

        numerator = vmax *(substrate - product)/km_external

        denominator = km_external * (1+ substrate/km_external + product/km_internal + 0.91 * substrate* product/km_external /km_internal)
        return numerator/denominator