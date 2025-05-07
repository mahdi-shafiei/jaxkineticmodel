from jaxkineticmodel.kinetic_mechanisms.JaxKineticMechanisms import Mechanism


class SimpleActivator(Mechanism):
    """ activation class modifier """

    @staticmethod
    def compute(activator, k_A):
        return 1 + activator / k_A


class SimpleInhibitor(Mechanism):
    """inhibition class modifier"""

    @staticmethod
    def compute(inhibitor, k_I):
        return 1 / (1 + inhibitor / k_I)


class BasicDivision(Mechanism):
    """Modifier to apply division based on a parameter/metaoblite"""
    @staticmethod
    def compute(symbol, scaling=1):
        return 1 / (symbol * scaling)


class BasicMultiplication(Mechanism):
    """Modifier to apply multiplication based on a parameter/metaoblite"""
    @staticmethod
    def compute(symbol, scaling=1):
        return symbol * scaling

class BiomassModifier(Mechanism):
    """Modifies expression based on Biomass. Used in the
    glycolysis model described in
    Lent, P. V., Bunkova, O., Planken, L., Schmitz, J., & Abeel, T. (2024).
    Neural Ordinary Differential Equations Inspired Parameterization of Kinetic Models.
    bioRxiv, 2024-12."""

    @staticmethod
    def compute(biomass):
        return biomass * 0.002
