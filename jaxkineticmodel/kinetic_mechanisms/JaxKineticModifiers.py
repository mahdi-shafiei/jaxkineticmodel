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
