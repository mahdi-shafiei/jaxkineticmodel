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
    

class Jax_Rev_UniUni_MM:
    """Reversible Michaelis-Menten"""
    def __init__(self,substrate:str,product:str, vmax: str, k_equilibrium: str, km_substrate: str, km_product: str):
        self.vmax = vmax
        self.k_equilibrium = k_equilibrium
        self.km_substrate = km_substrate
        self.km_product = km_product
        self.substrate=substrate
        self.product=product

    def __call__(self, eval_dict):
        vmax = eval_dict[self.vmax]
        k_equilibrium = eval_dict[self.k_equilibrium]
        km_substrate = eval_dict[self.km_substrate]
        km_product = eval_dict[self.km_product]

        substrate = eval_dict[self.substrate]
        product = eval_dict[self.product]

        nominator = vmax * (substrate / km_substrate) * (1 - (1 / k_equilibrium) * (product / substrate))
        denominator = 1 + (substrate / km_substrate) + (product / km_product)

        return nominator / denominator
    

class Jax_MM_Sink:
    """Just Irrev Michaelis-Menten, but objected as a specific class for sinks"""
    def __init__(self,substrate:str, v_sink: str, km_sink: str):
        self.v_sink = v_sink
        self.km_sink = km_sink
        self.substrate=substrate

    def __call__(self, eval_dict):
        v_sink = eval_dict[self.v_sink]
        km_sink = eval_dict[self.km_sink]
        substrate = eval_dict[self.substrate]

        return v_sink * substrate / (substrate + km_sink)
    


class Jax_Irrev_MM_Bi:
    def __init__(self, substrate1:str,
                 substrate2:str,
                 vmax: str,
                km_substrate1: str,
                km_substrate2: str):
                
        self.vmax = vmax
        self.km_substrate1 = km_substrate1
        self.km_substrate2 = km_substrate2
        self.substrate1=substrate1
        self.substrate2=substrate2

    def __call__(self, eval_dict):
        vmax = eval_dict[self.vmax]
        km_substrate1 = eval_dict[self.km_substrate1]
        km_substrate2 = eval_dict[self.km_substrate2]

        substrate1 = eval_dict[self.substrate1]
        substrate2 = eval_dict[self.substrate2]

        numerator = vmax * (substrate1 / km_substrate1) * (substrate2 / km_substrate2)
        denominator = (1 + (substrate1 / km_substrate1)) * (1 + (substrate2 / km_substrate2))

        return numerator / denominator