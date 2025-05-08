import inspect
import sympy



class Mechanism:
    """
    Abstract base class for a reaction mechansism
    """
    param_names: dict[str, str]
    modifiers: list['Mechanism']
    param_names_modifiers: dict[str, str]


    def __init__(self, **kwargs):
        # Initialize the parameter names
        try:
            inspect.signature(self.compute).bind(**kwargs)

        except TypeError:
            raise
        self.param_names = kwargs
        self.modifiers = []
        self.param_names_modifiers = {}


    def symbolic(self):
        symbol_dict = {k: sympy.Symbol(v) for k, v in self.param_names.items()}
        result = self.compute(**symbol_dict)
        for modifier in self.modifiers:
            result *= modifier.symbolic()
        return result

    def __str__(self):
        return str(self.symbolic())

    def __call__(self, eval_dict):
        kwargs = {k: eval_dict[v] for k, v in self.param_names.items()}
        result = self.compute(**kwargs)
        for modifier in self.modifiers:
            result *= modifier(eval_dict)
        return result

    def add_modifier(self, modifier: "Mechanism"):
        self.modifiers.append(modifier)
        self.param_names_modifiers.update(modifier.param_names)

    @staticmethod
    def compute(**kwargs):
        raise NotImplementedError()





class Jax_MM_Irrev_Uni(Mechanism):
    """Irreversible Michaelis-Menten kinetics (uni-substrate), adapted to JAX."""

    @staticmethod
    def compute(substrate, vmax, km_substrate):
        nominator = vmax * (substrate / km_substrate)
        denominator = 1 + (substrate / km_substrate)

        return nominator / denominator


class Jax_Facilitated_Diffusion(Mechanism):
    """facilitated diffusion formula, taken from
    Lao-Martil, D., Schmitz, J. P., Teusink, B., & van Riel, N. A. (2023). Elucidating yeast glycolytic dynamics at steady state growth and glucose pulses through
     kinetic metabolic modeling. Metabolic engineering, 77, 128-142.
     
     Class works slightly different than in torch. We have to simply include the names of the reaction parameter so that call recognizes them"""

    @staticmethod
    def compute(substrate,
                product,
                vmax,
                km_internal,
                km_external):
        numerator = vmax * (substrate - product) / km_external

        denominator = km_external * (
                1 + substrate / km_external + product / km_internal + 0.91 * substrate * product / km_external / km_internal
        )
        return numerator / denominator


class Jax_MM_Rev_UniUni(Mechanism):
    """Reversible Michaelis-Menten.

    Not sure about this particular equation. If there is a 0 substrate, you have division by zero. THats problematic
    I add epsilon for now to substrate """

    @staticmethod
    def compute(substrate, product, vmax,
                k_equilibrium, km_substrate, km_product):
        numerator = vmax * (substrate / km_substrate) * (1 - (1 / k_equilibrium) * (product / (substrate + 1e-8)))
        denominator = 1 + (substrate / km_substrate) + (product / km_product)
        return numerator / denominator


class Jax_MM(Mechanism):
    """Michaelis-Menten kinetic model."""

    @staticmethod
    def compute(substrate, vmax, km):
        return vmax * substrate / (substrate + km)


class Jax_MM_Sink(Mechanism):
    """Just Irrev Michaelis-Menten, but objected as a specific class for sinks"""

    @staticmethod
    def compute(substrate, v_sink, km_sink):
        return v_sink * substrate / (substrate + km_sink)


class Jax_MM_Irrev_Bi(Mechanism):
    @staticmethod
    def compute(substrate1, substrate2, vmax, km_substrate1, km_substrate2):
        numerator = vmax * (substrate1 / km_substrate1) * (substrate2 / km_substrate2)
        denominator = (1 + (substrate1 / km_substrate1)) * (1 + (substrate2 / km_substrate2))

        return numerator / denominator

class Jax_Constant_Flux(Mechanism):
    @staticmethod
    def compute(v):
        return v



class Jax_MM_Rev_UniBi(Mechanism):
    """Uni Bi reversible MM reaction of the form A-->B+C"""

    @staticmethod
    def compute(substrate, product1, product2, vmax, k_equilibrium, km_substrate, km_product1, km_product2):
        numerator = vmax / km_substrate * (substrate - product1 * product2 / k_equilibrium)
        denominator = substrate / km_substrate + (1 + product1 / km_product1) * (1 + product2 / km_product2)
        return numerator / denominator


class Jax_MM_Rev_BiBi(Mechanism):
    """Reversible BiBi Michaelis-Menten Kinetics

    Not sure about this particular equation. If there is a 0 substrate, you have division by zero. THats problematic
    I add epsilon for now to substrate (1e-8)
    """

    @staticmethod
    def compute(substrate1, substrate2, product1, product2, vmax, k_equilibrium, km_substrate1, km_substrate2,
                km_product1, km_product2):
        # Denominator calculation
        denominator = (1 + substrate1 / km_substrate1 + product1 / km_product1) * (
                1 + substrate2 / km_substrate2 + product2 / km_product2
        )

        # Numerator calculation
        numerator = (
                vmax
                * (substrate1 * substrate2 / (km_substrate1 * km_substrate2))
                * (1 - 1 / k_equilibrium * (product1 * product2 / ((substrate1 * substrate2) + 1e-8)))
        )

        # Rate equation
        return numerator / denominator


class Jax_Diffusion(Mechanism):
    """Diffusion model with a transport coefficient and enzyme."""

    @staticmethod
    def compute(substrate, enzyme, transport_coef):
        # Calculate diffusion rate
        diffusion_rate = transport_coef * (substrate - enzyme)
        return diffusion_rate


class Jax_MM_Ordered_Bi_Bi(Mechanism):
    """Ordered Bi-Bi Michaelis-Menten model with inhibitors."""

    @staticmethod
    def compute(substrate1, substrate2, product1, product2, vmax, k_equilibrium, km_substrate1, km_substrate2,
                km_product1, km_product2, ki_substrate1, ki_substrate2, ki_product1, ki_product2):
        s1 = substrate1  # NAD
        s2 = substrate2  # ETOH
        p1 = product1  # ACE
        p2 = product2  # NADH

        # Calculate numerator
        numerator = vmax * (s1 * s2 - p1 * p2 / k_equilibrium) / (ki_substrate1 * km_substrate2)

        # Calculate denominator
        denominator = (
                1
                + s1 / ki_substrate1
                + km_substrate1 * s2 / (ki_substrate1 * km_substrate2)
                + km_product2 * p1 / (km_product1 * ki_product2)
                + p2 / ki_product2
                + s1 * s2 / (ki_substrate1 * km_substrate2)
                + km_product2 * s1 * p1 / (km_product1 * ki_product2 * ki_substrate1)
                + km_substrate1 * s2 * p2 / (ki_substrate1 * km_substrate2 * ki_product2)
                + p1 * p2 / (km_product1 * ki_product2)
                + s1 * s2 * p1 / (ki_substrate1 * km_substrate2 * ki_product1)
                + s2 * p1 * p2 / (ki_substrate1 * km_substrate2 * ki_product2)
        )

        return numerator / denominator


class Jax_MA_Irrev(Mechanism):
    """Mass-action irreversible kinetic model."""

    @staticmethod
    def compute(substrate, k_fwd):
        return k_fwd * substrate
