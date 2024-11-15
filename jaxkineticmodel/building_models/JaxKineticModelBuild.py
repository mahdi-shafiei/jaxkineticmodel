import jax.numpy as jnp
import numpy as np
from collections import Counter
import diffrax
import pandas as pd
import sympy as sp
from jaxkineticmodel.utils import get_logger

logger = get_logger(__name__)


class BoundaryCondition:
    """Class to evaluate boundary conditions, similar in form to Diffrax. For most purposes the interpolation in diffrax is 
    perfectly fine.
    For now we only will consider boundary conditions that are dependent on t.

    #To do: think about how to expand this class to include metabolite dependencies in expression 
    # (could be easier to jax.jit)"""

    def __init__(self, expression: str):
        self.expression = sp.sympify(expression)

        self.expression = sp.lambdify(sp.Symbol("t"), self.expression, "jax")

    def evaluate(self, t):
        return self.expression(t)
        # return


class Reaction:
    """Base class that can be used for building kinetic models. The following things must be specified:
    species involved,
    name of reaction
    stoichiometry of the specific reaction,
    mechanism + named parameters, and compartment"""

    def __init__(self, name: str, species: list, stoichiometry: list, compartments: list, mechanism):
        self.name = name
        self.species = species
        self.stoichiometry = dict(zip(species, stoichiometry))
        self.mechanism = mechanism
        self.compartments = dict(zip(species, compartments))

        self.parameters = np.setdiff1d(
            list(vars(mechanism).values()), species
        )  # excludes species as parameters, but add as seperate variable
        self.species_in_mechanism = np.intersect1d(list(vars(mechanism).values()), species)


class JaxKineticModel_Build:
    def __init__(self, reactions: list, compartment_values: dict):
        """Kinetic model that is defined through it's reactions:
        Input:"""
        self.reactions = reactions

        self.stoichiometric_matrix = self._get_stoichiometry()
        self.S = jnp.array(self.stoichiometric_matrix)  # use for evaluation
        self.reaction_names = self.stoichiometric_matrix.columns.to_list()
        self.species_names = self.stoichiometric_matrix.index.to_list()

        self.species_compartments = self._get_compartments_species()
        self.compartment_values = jnp.array([compartment_values[self.species_compartments[i]] for i in self.species_names])

        # only retrieve the mechanisms from each reaction
        self.v = [reaction.mechanism for reaction in self.reactions]

        # retrieve parameter names
        self.parameter_names = self._flatten([reaction.parameters for reaction in self.reactions])
        self._check_parameter_uniqueness()

        self.boundary_conditions = {}

    def _get_stoichiometry(self):
        """Build stoichiometric matrix from reactions"""
        build_dict = {}
        for reaction in self.reactions:
            build_dict[reaction.name] = reaction.stoichiometry
        S = pd.DataFrame(build_dict).fillna(value=0)
        return S

    def _flatten(self, xss):
        return [x for xs in xss for x in xs]

    def _get_compartments_species(self):
        """Retrieve compartments for species and do a consistency check 
        that compartments are properly defined for each species"""
        comp_dict = {}
        for reaction in self.reactions:
            for species, values in reaction.compartments.items():
                if species not in comp_dict.keys():
                    comp_dict[species] = values
                else:
                    if comp_dict[species] != values:
                        logger.error(
                            f"Species {species} has ambiguous compartment values, please check consistency in the reaction definition"
                        )
        return comp_dict

    def _check_parameter_uniqueness(self):
        """Checks whether parameters are unique. Throws info if not, since some compounds might be governed by global (process) parameters"""
        count_list = Counter(self.parameter_names).values()
        for count, k in enumerate(count_list):
            if k != 1:
                logger.info(f"parameter {self.parameter_names[k]} is in multiple reactions.")

    def add_boundary(self, metabolite_name, boundary_condition):
        """Add a metabolite boundary condition
        input: metabolite name, boundary condition object or diffrax interpolation object"""

        self.boundary_conditions.update({metabolite_name: boundary_condition})
        index = self.species_names.index(metabolite_name)
        # self.S=jnp.delete(self.S,index)
        self.species_names.remove(metabolite_name)
        self.S = jnp.delete(self.S, index, axis=0)
        self.stoichiometric_matrix = self.stoichiometric_matrix.drop(labels=metabolite_name, axis=0)
        self.compartment_values = jnp.delete(self.compartment_values, index)

    def __call__(self, t, y, args):
        params, boundary_conditions = args

        y = dict(zip(self.species_names, y))
        if boundary_conditions:
            for key, value in boundary_conditions.items():
                boundary_conditions[key] = value.evaluate(t)

        # we construct this dictionary, and then overwrite

        # Think about how to vectorize the evaluation of mechanism.call
        eval_dict = {**y, **params, **boundary_conditions}
        v = jnp.stack([self.v[i](eval_dict) for i in range(len(self.reaction_names))])
        dY = jnp.matmul(self.S, v)
        dY = dY / self.compartment_values

        return dY


class NeuralODEBuild:
    def __init__(self, func):
        self.func = func
        self.parameter_names = func.parameter_names
        self.stoichiometric_matrix = func.stoichiometric_matrix
        self.reaction_names = list(func.stoichiometric_matrix.columns)
        self.species_names = list(func.stoichiometric_matrix.index)
        self.Stoichiometry = func.S
        self.boundary_conditions = {}

        self.max_steps = 200000
        self.rtol = 1e-8
        self.atol = 1e-11

        ## time dependencies: a function that return for all fluxes whether there is a time dependency

    def __call__(self, ts, y0, params):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Kvaerno5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=1e-8,
            y0=y0,
            args=(params, self.boundary_conditions),
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol, pcoeff=0.4, icoeff=0.3, dcoeff=0),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=self.max_steps,
        )

        return solution.ys
