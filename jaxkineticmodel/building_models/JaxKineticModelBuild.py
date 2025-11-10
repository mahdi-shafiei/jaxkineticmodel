from typing import List, Any

import jax.lax
import jax.numpy as jnp
import diffrax
import pandas as pd
import sympy as sp

from jaxkineticmodel.kinetic_mechanisms.JaxKineticMechanisms import Mechanism
from jaxkineticmodel.utils import get_logger

logger = get_logger(__name__)


class BoundaryCondition:
    """Class to evaluate boundary conditions, similar in form to Diffrax. For most purposes the interpolation in diffrax is
    perfectly fine. For now, we only will consider boundary conditions that are dependent on t.

    input:
    String expression of the boundary condition (which can be e.g., str(2))
    Boolean of whether expression is a constant boundary condition.
    This is required for consistency with exporting to sbml

    #To do: think about how to expand this class to include metabolite dependencies in expression
    """

    is_constant: bool
    sympified: sp.Basic
    lambdified: Any

    def __init__(self, string_expression: str):
        self.sympified = sp.sympify(string_expression)
        self.is_constant = isinstance(self.sympified, sp.Number)
        self.lambdified = sp.lambdify(sp.Symbol("t"), self.sympified, "jax")

    def evaluate(self, t):
        return self.lambdified(t)


class AssignmentRule:
    """
    Class to evaluate assignment rules in the NeuralODEBuild object.
    Input: string expression of the assignment rule
    """

    def __init__(self,
                 rule: str):
        self.rule = rule
        self.equation = sp.sympify(rule)
        self.free_symbols = list(self.equation.free_symbols)
        self.lambdified = sp.lambdify(self.free_symbols, self.rule)

    def __call__(self, eval_dict):
        eval_symbols = {str(k): eval_dict[str(k)] for k in self.free_symbols}
        return self.lambdified(**eval_symbols)


class BoundaryConditionDiffrax:
    """Helper function to load diffrax intepolations to the NeuralODEBuild object. It iss not compatible with SBML"""

    def __init__(self, boundary_condition: diffrax.AbstractPath):
        assert isinstance(boundary_condition, diffrax.AbstractPath)
        self.boundary_condition = boundary_condition

    def __call__(self, t):
        return self.boundary_condition.evaluate(t)


class Reaction:
    """Base class that can be used for building kinetic models. The following things must be specified:
    - species involved,
    - name of reaction
    - stoichiometry of the specific reaction,
    - mechanism + named parameters,
    - compartments"""

    def __init__(self, name: str, species: list, stoichiometry: list,
                 compartments: list, mechanism: Mechanism):
        assert len(stoichiometry) == len(species), "length stoichiometry should match number of species"
        assert len(stoichiometry) == len(compartments), "length stoichiometry should match number of compartments"

        self.name = name
        self.species = species
        self.stoichiometry = dict(zip(species, stoichiometry))
        self.mechanism = mechanism
        self.compartments = dict(zip(species, compartments))

        # exclude species as parameters, but add as seperate variable
        self.parameters = [x for x in mechanism.param_names.values()
                           if x not in species]
        self.parameters = self._add_modifier_parameters()
        self.species_in_mechanism = [x for x in mechanism.param_names.values()
                                     if x in species]

    def _add_modifier_parameters(self):
        """Adds parameter if a flux mechanism is modified (e.g., inhibitors,
        activators)"""
        if self.mechanism.param_names_modifiers:
            modifier_parameters = [x for x in self.mechanism.param_names_modifiers.values()
                                   if x not in self.species]
            parameters = [*self.parameters, *modifier_parameters]
        else:
            parameters = self.parameters

        return parameters


class JaxKineticModelBuild:
    reactions: List[Reaction]
    compartments: dict[str, int]
    boundary_conditions: dict[str, BoundaryCondition]
    stoichiometric_matrix: pd.DataFrame
    species_names: list[str]

    def __init__(self, reactions: List[Reaction], compartments: dict[str, int]):
        """A kinetic model that is defined through a list  Reactions:
        Input:
        reactions: list of reaction objects,
        compartments: list of compartments with corresponding value for evaluation
        """
        self.reactions = reactions
        self.stoichiometric_matrix = self._get_stoichiometry()
        self.S = jnp.array(self.stoichiometric_matrix)  # use for evaluation
        self.reaction_names = self.stoichiometric_matrix.columns.to_list()
        self.species_names = self.stoichiometric_matrix.index.to_list()

        self.species_compartments = self._get_compartments_species()
        self.compartments = compartments
        self.compartment_values = jnp.array([compartments[self.species_compartments[i]] for i in self.species_names])
        # only retrieve the mechanisms from each reaction
        self.v = [reaction.mechanism for reaction in self.reactions]

        # retrieve parameter names
        self.parameter_names = self._flatten([reaction.parameters for reaction in self.reactions])
        self.parameter_names = self._filter_parameters()

        self.boundary_conditions = {}

        self.assignment_rules = {}

    def _filter_parameters(self):
        """Filters out species from parameter list when a function has modifiers"""
        parameter_names = []
        for parameter in self.parameter_names:
            if parameter not in self.species_names:
                parameter_names.append(parameter)
        return parameter_names

    def _get_stoichiometry(self) -> pd.DataFrame:
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
                            (
                                f"Species {species} has ambiguous compartment values, "
                                f"please check consistency in the reaction definition"
                            )
                        )
        return comp_dict

    def add_boundary(self, metabolite_name: str, boundary_condition: BoundaryCondition):
        """Add a metabolite boundary condition
        input: metabolite name, boundary condition object or diffrax interpolation object"""

        # updates the list of boundary conditions
        self.boundary_conditions.update({metabolite_name: boundary_condition})
        index = self.species_names.index(metabolite_name)

        # since it is now a boundary condition, it will be removed
        # from species list
        self.species_names.remove(metabolite_name)

        # boundary conditions will not be evaluated in S*v(t)
        self.S = jnp.delete(self.S, index, axis=0)

        # remove it from the species_compartments list.
        # While a boundary could still lie inside/outside a compartment, from an evaluation
        # perspective it should not matter.

        # same here, but then for the pandas
        # (refactor this later)
        self.stoichiometric_matrix = self.stoichiometric_matrix.drop(labels=metabolite_name, axis=0)
        self.compartment_values = jnp.delete(self.compartment_values, index)

    def add_parameter_assignment_rule(self,
                                      variable: str,
                                      rule: AssignmentRule):
        """For a given parameter, create a rule that describes the value in symbolic terms.
        This is similar to add_boundary, but then for parameters.

        - Variable: can be a species, parameter or even a compartment (?)
        - Rule: string expression of the assignment rule


        NOTE: we need to handle the export and NeuralODEBuild simulation slightly differently.
        In the export: we want to include the rules in the sbml file, while for when
        you directly want to simulate the model, NeuralODEBuild has directly replaced the rules."""

        if variable in self.species_names:
            logger.warning(f"variable {variable} is a state variable, has not been properly tested yet")
        free_symbols = rule.free_symbols

        # check whether free_symbols in the assignment rule are actually defined in the model

        if variable not in self.parameter_names and variable not in self.species_names:
            logger.error(f"variable {variable} is not a valid assignment rule variable")
            raise AssertionError(f"variable: {variable} is not a valid assignment rule variable")

        # check whether all symbols are included in parameter_
        symbols_list = [i for i in free_symbols if str(i) in self.parameter_names or str(i) in self.species_names]


        if len(symbols_list) != len(free_symbols):
            missing = set(free_symbols) - set(symbols_list)
            logger.error(f"Invalid symbols: {missing}")
            raise AssertionError(f"Invalid symbols: {missing}")

        self.assignment_rules.update({variable: rule})
        return logger.info(f"added assignment rule {rule} for variable {variable}")

    def __call__(self, t, y, args):
        params, boundary_conditions, assignment_rules = args

        y = dict(zip(self.species_names, y))
        if boundary_conditions:
            for key, expression in boundary_conditions.items():
                boundary_conditions[key] = expression.evaluate(t)

        # we construct this dictionary, and then overwrite
        # Think about how to vectorize the evaluation of mechanism.call
        eval_dict = {**y, **params, **boundary_conditions}

        # while in principle one can do assignmnet rules
        if assignment_rules:
            for key, expression in assignment_rules.items():
                if key in y.keys():
                    eval_dict[key] = expression(eval_dict)
                elif key in params.keys():
                    eval_dict[key] = expression(eval_dict)

        v = jnp.stack([self.v[i](eval_dict) for i in range(len(self.reaction_names))])
        dY = jnp.matmul(self.S, v)
        dY /= self.compartment_values

        return dY


class NeuralODEBuild:
    """Class that wraps jaxkineticmodelbuild and can be used for parameterization"""
    def __init__(self,
                 func: JaxKineticModelBuild):
        self.func = func
        self.parameter_names = func.parameter_names
        self.stoichiometric_matrix = func.stoichiometric_matrix
        self.reaction_names = list(func.stoichiometric_matrix.columns)
        self.species_names = list(func.stoichiometric_matrix.index)
        self.Stoichiometry = func.S
        self.boundary_conditions = func.boundary_conditions
        self.assignment_rules = func.assignment_rules

        self.max_steps = 300000
        self.rtol = 1e-9
        self.atol = 1e-11
        self.dt0 = 1e-12
        self.solver = diffrax.Kvaerno5()
        self.stepsize_controller = diffrax.PIDController(rtol=self.rtol, atol=self.atol, pcoeff=0.4, icoeff=0.3,
                                                         dcoeff=0)
        self.adjoint = diffrax.RecursiveCheckpointAdjoint()

    def _change_solver(self, solver, **kwargs):
        """To change the ODE solver object to any solver class from diffrax
        Does not support multiterm objects yet."""

        if isinstance(solver, diffrax.AbstractAdaptiveSolver):
            # for what i recall, only the adaptive part is important to ensure
            # it can be loaded properly
            self.solver = solver
            step_size_control_parameters = {'rtol': self.rtol, 'atol': self.atol,
                                            "pcoeff": 0.4, "icoeff": 0.3, "dcoeff": 0}
            for key in kwargs:
                if key in step_size_control_parameters:
                    step_size_control_parameters[key] = kwargs[key]
            self.stepsize_controller = diffrax.PIDController(**step_size_control_parameters)
        elif not isinstance(solver, diffrax.AbstractAdaptiveSolver):
            self.solver = solver
            self.stepsize_controller = diffrax.ConstantStepSize()
        else:
            logger.error(f"solver {type(solver)} not supported yet")

        return logger.info(f"solver changed to {type(solver)}")

    def __call__(self, ts, y0, params):

        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self.func),
            solver=diffrax.Kvaerno5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=self.dt0,
            y0=y0,
            args=(params, self.boundary_conditions, self.assignment_rules),
            stepsize_controller=self.stepsize_controller,
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=self.max_steps,
            adjoint=self.adjoint
        )

        return solution.ys
