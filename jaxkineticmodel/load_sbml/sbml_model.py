
from typing import Union

import sympy as sp
import jax
import libsbml
import jax.numpy as jnp
import numpy as np
import pandas as pd
import re
import collections
import os
from jaxkineticmodel.utils import get_logger
from jaxkineticmodel.load_sbml.jax_kinetic_model import NeuralODE
from jaxkineticmodel.load_sbml.sympy_converter import SympyConverter, LibSBMLConverter

logger = get_logger(__name__)


class SBMLModel:
    S: Union[pd.DataFrame, None]

    def __init__(self, file_path):
        self.model = self._load_model(file_path)
        self.S = self._get_stoichiometric_matrix()
        # TODO the following two lines assume that self.S will never change
        self.reaction_names = list(self.S.columns)
        self.species_names = list(self.S.index)

        self.parameters = self._get_parameters()
        self.initial_assignments=self._get_initial_assignments()
        self.y0 = self._get_initial_conditions()


        self.y0,self.parameters=self._update_with_initial_assignments()
        self.y0 = jnp.array(list(self.y0.values()))
        self.compartments, self.species_compartment_values = self._get_compartments()

        self.v, self.v_symbols = self._get_fluxes()
        self.met_point_dict = self._construct_flux_pointer_dictionary()

    @staticmethod
    def _load_model(file_path):
        if not os.path.isfile(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError()

        reader = libsbml.SBMLReader()
        document = reader.readSBML(file_path)
        inconsistencies = document.checkInternalConsistency()
        if inconsistencies == 0:
            logger.info("No internal inconsistencies found")
        else:
            logger.warning(f"Number of internal inconsistencies: {document.checkInternalConsistency()}")

        model = document.getModel()
        logger.info("Model loaded.")
        logger.info(f" number of species: {model.getNumSpecies()}")
        logger.info(f" number of reactions: {model.getNumReactions()}")
        logger.info(f" number of global parameters: {model.getNumParameters()}")
        logger.info(f" number of constant boundary metabolites: {model.getNumSpeciesWithBoundaryCondition()}")
        logger.info(f" number of lambda function definitions: {len(model.function_definitions)}")
        logger.info(f" number of assignment rules: {model.getNumRules()}")
        logger.info(f" number of event rules: {model.getNumEvents()}")

        return model

    def _update_with_initial_assignments(self):
        """Update parameters and species with initial assignments."""

        #As far as I am aware, only parameters and species can be updated through initial assignments"""
        parameters=self.parameters
        y0=self.y0
        for key,value in self.initial_assignments.items():
            if key in self.parameters.keys():
                value=value.subs(self.parameters) #substitutes parameters
                value = value.subs(self.parameters)  # substitutes parameters
                parameters[key]=float(value) #force it to be a float
            elif key in self.y0.keys():
                value=value.subs(self.parameters)
                value=value.subs(self.y0)
                y0[key] = float(value)  # force it to be a float
            else:
                logger.warning(f"initial assignment rule not in species or parameters. Assignment for"
                               "{key} is ignored and simulation might be wrong")
        return y0, parameters


    def _get_initial_assignments(self):
        "retrieve all rules that are defined outside of species and parameters, we then use these rules to update y0 and parameters"
        libsbml_converter = LibSBMLConverter()
        initial_assignments = {}
        for assignment in self.model.getListOfInitialAssignments():
            expression = assignment.getMath()
            expression = libsbml_converter.libsbml2sympy(expression)
            initial_assignments[assignment.id] = expression
        return initial_assignments

    def _get_compartments(self):
        """retrieves compartments and the compartment values of species. The latter is necessary for proper scaling
        of rate laws."""

        compartments = self.model.getListOfCompartments()
        compartments = {cmp.id: cmp.size for cmp in compartments}
        compartment_list = self._get_compartments_initial_conditions(compartments)

        return compartments,compartment_list

    def _get_stoichiometric_matrix(self):
        """Retrieves the stoichiometric matrix from the model."""
        species_ids = []
        reduced_species_list = []
        for s in self.model.getListOfSpecies():
            # these conditions do not have a rate law
            if s.getConstant() and s.getBoundaryCondition():
                continue
            elif s.getBoundaryCondition() and not s.getConstant():
                continue
            elif s.getConstant():
                continue
            else:
                reduced_species_list.append(s)
                species_ids.append(s.getId())

        # species = [s.getName() for s in model.getListOfSpecies()]
        reactions = [r.getId() for r in self.model.getListOfReactions()]

        stoichiometry_matrix = np.zeros((len(species_ids), len(reactions)))
        for reaction_index, reaction in enumerate(self.model.getListOfReactions()):
            reactants = {r.getSpecies(): r.getStoichiometry() for r in reaction.getListOfReactants()}
            products = {p.getSpecies(): p.getStoichiometry() for p in reaction.getListOfProducts()}

            for species_index, species_node in enumerate(reduced_species_list):
                species_id = species_node.getId()

                net_stoichiometry = -int(reactants.get(species_id, 0)) + int(products.get(species_id, 0))
                # print(net_stoichiometry)
                stoichiometry_matrix[species_index, reaction_index] = net_stoichiometry

        species_names = [s.getId() for s in reduced_species_list]
        reaction_names = [r.getId() for r in self.model.getListOfReactions()]

        return pd.DataFrame(stoichiometry_matrix, index=species_names, columns=reaction_names)

    def _get_initial_conditions(self):
        """Retrieves the species initial concentrations
        from the SBML model. If a species is a constant boundary condition,
        then it should be passed as a parameter instead of an initial condition,
        since it does not have a rate law

        also"""
        species = self.model.getListOfSpecies()
        initial_concentration_dict = {}
        for specimen in species:
            if specimen.isSetConstant() and specimen.isSetBoundaryCondition():
                # there are also non-stationary boundary conditions, deal with this later.
                if specimen.getConstant() and specimen.getBoundaryCondition():
                    logger.info(f"Constant Boundary Specimen {specimen.id}")
                    continue
                elif specimen.getBoundaryCondition() and not specimen.getConstant():
                    continue
                elif not specimen.getBoundaryCondition() and specimen.getConstant():
                    continue  # not a boundary, but still a constant
                elif not specimen.getBoundaryCondition() and not specimen.getConstant():
                    if specimen.isSetInitialConcentration():

                        initial_concentration_dict[specimen.id] = specimen.initial_concentration
                    elif specimen.isSetInitialAmount():
                        initial_concentration_dict[specimen.id] = specimen.initial_amount
                    else:
                        logger.error(f"specimen {specimen.id} has no initial amount or concentration set")
            else:
                logger.warn(f"{specimen.id} constant/boundary attribute not set. Assume that boundary is constant")
                continue

        return initial_concentration_dict

    def _get_parameters(self):
        """Retrieves the parameters from the SBML model. Both local (with a identifier lp.{reaction.id}.) and global."""

        parameters={}
        #retrieve global parameters
        global_params = self.model.getListOfParameters()
        global_parameters = {param.id: param.value for param in global_params}
        parameters.update(global_parameters)

        #retrieve local parameters
        for reaction in self.model.reactions:
            r = reaction.getKineticLaw()
            local_parameters = {"lp."+str(reaction.id)+"."+param.id: param.value for param in r.getListOfParameters()}
            parameters.update(local_parameters)
        return parameters

    def _get_compartments_initial_conditions(self, compartments):
        """Returns a list of the compartment values of
        the initial conditions. This is necessary in the dMdt to properly scale."""
        species = self.model.getListOfSpecies()
        compartment_list = []

        for specimen in species:
            if specimen.isSetConstant() and specimen.isSetBoundaryCondition():
                # there are also non-stationary boundary conditions, deal with this later.
                if specimen.getConstant() and specimen.getBoundaryCondition():
                    logger.info(f"Constant Boundary Specimen {specimen.id}")
                    continue
                elif specimen.getBoundaryCondition() and not specimen.getConstant():
                    continue
                elif not specimen.getBoundaryCondition() and specimen.getConstant():
                    continue  # not a boundary, but still a constant
                elif not specimen.getBoundaryCondition() and not specimen.getConstant():
                    compartment_list.append(compartments[specimen.compartment])

        compartment_list = jnp.array(compartment_list)
        return compartment_list

    def _get_fluxes(self):
        """Retrieves flux functions from the SBML model for simulation,
        It already replaces some values that are constant."""

        libsbml_converter = LibSBMLConverter()
        species_ic = self._get_initial_conditions()
        constant_boundaries=get_constant_boundary_species(self.model)
        lambda_functions=get_lambda_function_dictionary(self.model)
        assignments_rules=get_assignment_rules_dictionary(self.model)
        event_rules = get_events_dictionary(self.model)

        v = {}
        v_symbol_dict = {}  # all symbols that are used in the equation.

        for reaction in self.model.reactions:

            astnode_reaction=reaction.getKineticLaw().math
            equation=libsbml_converter.libsbml2sympy(astnode_reaction) #sympy type

            # arguments from the lambda expression are mapped to their respective symbols.
            for func in equation.atoms(sp.Function):
                if hasattr(func, 'name'):
                    variables=lambda_functions[func.name].variables
                    variable_substitution = dict(zip(variables, func.args))
                    expression=lambda_functions[func.name].expr
                    expression=expression.subs(variable_substitution)
                    equation=equation.subs({func:expression})



            equation=equation.subs(self.compartments)
            equation=equation.subs(assignments_rules)
            equation=equation.subs(constant_boundaries)


            free_symbols=list(equation.free_symbols)

            equation=sp.lambdify(free_symbols, equation,"jax")

            filtered_dict=dict(zip([str(i) for i in free_symbols],free_symbols))

            # v[reaction.id] = vi  # the jitted equation
            v[reaction.id]=equation
            v_symbol_dict[reaction.id] = filtered_dict

        return v, v_symbol_dict

    def _construct_flux_pointer_dictionary(self):
        """In jax, the values that are used need to be pointed directly in y0."""
        flux_point_dict = {}
        for k, reaction in enumerate(self.reaction_names):
            v_dict = self.v_symbols[reaction]
            filtered_dict = [self.species_names.index(key) for key in v_dict.keys() if key in self.species_names]
            filtered_dict = jnp.array(filtered_dict)
            flux_point_dict[reaction] = filtered_dict
        return flux_point_dict

    def get_kinetic_model(self):
        return NeuralODE(
            fluxes=self.v,
            stoichiometric_matrix=self.S,
            met_point_dict=self.met_point_dict,
            v_symbols=self.v_symbols,
            compartment_values=self.species_compartment_values,
        )


def get_lambda_function_dictionary(model):
    """Stop giving these functions confusing names...
    it returns a dictionary with all lambda functions"""
    functional_dict = {}
    libsbml_converter = LibSBMLConverter()

    for function in model.function_definitions:
        id = function.getId()
        math = function.getMath()
        equation=libsbml_converter.libsbml2sympy(math)


        functional_dict[id] = equation
    return functional_dict


def get_global_parameters(model):
    """Most sbml models have their parameters defined globally,
    this function retrieves them"""
    params = model.getListOfParameters()
    global_parameter_dict = {param.id: param.value for param in params}
    return global_parameter_dict


def get_compartments(model):
    """Some sbml models have compartments, retrieves them"""
    compartments = model.getListOfCompartments()
    compartment_dict = {cmp.id: cmp.size for cmp in compartments}
    return compartment_dict


# We do not deal yet with non-constant boundaries
def get_string_expression(reaction):
    """retrieves the kinetic rate law from the reaction"""
    kinetic_law = reaction.getKineticLaw()
    # print(kinetic_law.name)
    klaw_math = kinetic_law.math
    string_rate_law = libsbml.formulaToString(klaw_math)
    # here we sometimes need to add exceptions. For example, to evaluate tanh, we need to replace it with torch.Tanh
    string_rate_law = string_rate_law.replace("^", "**")
    return string_rate_law


def get_constant_boundary_species(model):
    """Species that are boundary conditions should be fed as fixed, non-learnable parameters
    https://synonym.caltech.edu/software/libsbml/5.18.0/docs/formatted/python-api/classlibsbml_1_1_species.html"""
    constant_boundary_dict = {}
    species = model.getListOfSpecies()
    for specimen in species:
        if specimen.getBoundaryCondition():
            if model.getLevel() == 2:
                logger.info(f"Assume that boundary {specimen.id} is constant for level 2")
                constant_boundary_dict[specimen.id] = specimen.initial_concentration

            constant_boundary_dict[specimen.id] = specimen.initial_concentration
    return constant_boundary_dict


def get_local_parameters(reaction):
    """Some sbml models also have local parameters (locally defined for reactions),
    this function retrieves them for an individual reaction, removing the chance
    similarly named parameters are overwritten"""
    r = reaction.getKineticLaw()
    local_parameter_dict = {param.id: param.value for param in r.getListOfParameters()}
    return local_parameter_dict


def get_reaction_species(reaction):
    """Retrieves the substrates, products, and modifiers from sbml format.
    These will be passed to the Torch Kinetic Model class."""
    sub = reaction.getListOfReactants()
    prod = reaction.getListOfProducts()
    mod = reaction.getListOfModifiers()

    substrates = [s.species for s in sub]
    products = [p.species for p in prod]
    modifiers = [m.species for m in mod]

    species = substrates + products + modifiers
    return species


def get_reaction_symbols_dict(eval_dict):
    """This functions works on the local_dictionary passed in the sympify function
    It ensures that sympy symbols for parameters and y-values are properly passed,
    while the rest is simply substituted in the expression."""
    symbol_dict = {i: sp.Symbol(i) for i in eval_dict.keys() if not callable(eval_dict[i])}  # skip functions symbols
    return symbol_dict


# we want the output to be a list v, which contains jitted function
def sympify_lambidify_and_jit_equation(equation, nested_local_dict):
    """Sympifies, lambdifies, and then jits a string rate law
    equation: the string rate law equation
    nested_local_dict: a dictionary having dictionaries of
      global parameters,local parameters, compartments, and boundary conditions

      #returns
      the jitted equation
      a filtered dictionary. This will be used to construct the flux_pointer_dictionary.

    """
    # unpacking the nested_local_dictionary, with global and local parameters symbols
    globals = get_reaction_symbols_dict(nested_local_dict["globals"])
    locals = get_reaction_symbols_dict(nested_local_dict["locals"])
    species = get_reaction_symbols_dict(nested_local_dict["species"])
    boundaries = nested_local_dict["boundary"]
    lambda_funcs = nested_local_dict["lambda_functions"]

    assignment_rules = nested_local_dict["boundary_assignments"]

    compartments = nested_local_dict["compartments"]  # learnable
    local_dict = {**species, **globals, **locals}

    equation = sp.sympify(equation, locals={**local_dict, **assignment_rules, **lambda_funcs})

    # these are filled in before compiling

    equation = equation.subs(assignment_rules)
    equation = equation.subs(compartments)
    equation = equation.subs(boundaries)
    equation = equation.subs({"pi": sp.pi})

    # free symbols are used for lambdifying
    free_symbols = list(equation.free_symbols)

    filtered_dict = {key: value for key, value in local_dict.items() if value in free_symbols or key in locals}

    # perhaps a bit hacky, but some sbml models have symbols that are
    # predefined
    for symbol in free_symbols:
        if str(symbol) == "time":
            logger.info("time")
            filtered_dict["time"] = sp.Symbol("time")

    equation = sp.lambdify((filtered_dict.values()), equation, "jax")
    equation = jax.jit(equation)

    return equation, filtered_dict


def species_match_to_S(initial_conditions, species_names):
    """Small helper function ensures that y0 is properly matched to rows of S
    Input: initial conditions is a dictionary of initial conditions
    species_names: is the order of the rows in S"""
    y0 = []
    for species in species_names:
        if species in initial_conditions.keys():
            y0.append(initial_conditions[species])
    y0 = jnp.array(y0)
    return y0


def reaction_match_to_S(flux_funcs, reaction_names):
    """Small helper function ensures that y0 is properly matched to rows of S
    Input: initial conditions is a dictionary of initial conditions
    species_names: is the order of the rows in S"""
    v = []
    for reaction in reaction_names:
        v.append(flux_funcs[reaction])
    return v


def construct_param_point_dictionary(v_symbol_dictionaries, reaction_names, parameters):
    """In jax, the values that are used need to be pointed directly in y."""
    flux_point_dict = {}
    for k, reaction in enumerate(reaction_names):
        v_dict = v_symbol_dictionaries[reaction]
        filtered_dict = {}
        for key, value in v_dict.items():
            if key in parameters.keys():
                filtered_dict[key] = parameters[key]
        # params_point_dict=[parameters.index(key) for key in v_dict.keys() if key in parameters]
        # print(params_point_dict)
        # filtered_dict=jnp.array(params_point_dict)
        flux_point_dict[reaction] = filtered_dict
    return flux_point_dict


def get_leaf_nodes(node, leaf_nodes):
    """Finds the leaf nodes of the mathml expression."""
    if node.getNumChildren() == 0:
        name = node.getName()
        if name is not None:
            leaf_nodes.append(name)
        # print(node.getName())
    else:
        for i in range(node.getNumChildren()):
            get_leaf_nodes(node.getChild(i), leaf_nodes)
    leaf_nodes = np.array(leaf_nodes)
    leaf_nodes = np.unique(leaf_nodes)
    leaf_nodes = leaf_nodes.tolist()
    return leaf_nodes




def replace_piecewise(formula):
    """Replace libsbml piecewise with sympy piecewise."""
    # Code taken from: https://github.com/matthiaskoenig/sbmlsim/blob/develop/src/sbmlsim/combine/mathml.py
    # FIXME This approach is not robust (or very Pythonic). Rewrite with
    #  regular expressions, or by iterating through the AST.
    while True:
        index = formula.find("piecewise(")
        if index == -1:
            break

        # process piecewise
        search_idx = index + 9

        # init counters
        bracket_open = 0
        pieces = []
        piece_chars = []

        while search_idx < len(formula):
            c = formula[search_idx]
            if c == ",":
                if bracket_open == 1:
                    pieces.append("".join(piece_chars).strip())
                    piece_chars = []
            else:
                if c == "(":
                    if bracket_open != 0:
                        piece_chars.append(c)
                    bracket_open += 1
                elif c == ")":
                    if bracket_open != 1:
                        piece_chars.append(c)
                    bracket_open -= 1
                else:
                    piece_chars.append(c)

            if bracket_open == 0:
                pieces.append("".join(piece_chars).strip())
                break

            # next character
            search_idx += 1

        # find end index
        if (len(pieces) % 2) == 1:
            pieces.append("True")  # last condition is True
        sympy_pieces = []
        for k in range(0, int(len(pieces) / 2)):
            sympy_pieces.append(f"({pieces[2*k]}, {pieces[2*k+1]})")
        new_str = f"Piecewise({','.join(sympy_pieces)})"
        formula = formula.replace(formula[index : search_idx + 1], new_str)

    return formula




def separate_params(params):
    global_params = {}
    local_params = collections.defaultdict(dict)

    for key in params.keys():
        if re.match("lp.*.", key):
            fkey = key.removeprefix("lp.")
            list = fkey.split(".")
            value = params[key]
            newkey = list[1]
            local_params[list[0]][newkey] = value
        else:
            global_params[key] = params[key]
    return global_params, local_params


# def wrap_time_symbols(t):
def time_dependency_symbols(v_symbol_dictionaries, t):
    time_dependencies = {}
    for key, values in v_symbol_dictionaries.items():
        time_dependencies[key] = {}
        for value in values.keys():
            if value == "time":
                time_dependencies[key] = {value: t}
    return time_dependencies


#   time_dependencies=time_dependency_symbols(v_symbol_dictionaries,t)
#   return time_dependencies



def get_assignment_rules_dictionary(model):
    """Get rules that assign to variables. I did not lambdify here"""
    libsbml_converter = LibSBMLConverter()
    assignment_dict = {}
    for rule in model.rules:
        id = rule.getId()
        expr = rule.getMath()
        expr=libsbml_converter.libsbml2sympy(expr)


        assignment_dict[id] = expr

    return assignment_dict




def get_events_dictionary(model):
    """There are many type of events. For now, I only add a few and the rest will throw a warning"""
    num_events = model.getNumEvents()
    events_dict = {}
    if model.getLevel() == 2 and num_events != 0:
        for i in range(num_events):
            event = model.getEvent(i)

            # Print the trigger
            trigger = event.getTrigger()
            if trigger is not None:
                logger.info(f"Trigger: {libsbml.formulaToString(trigger.getMath())}")
                trigger_event = libsbml.formulaToL3String(trigger.getMath())

                # need to replace && for proper reading in sympy
                trigger_event = trigger_event.replace("&&", "&")
                leaf_nodes = []
                leaf_nodes = get_leaf_nodes(trigger.getMath(), leaf_nodes)
                symbols = {leaf_node: sp.Symbol(leaf_node) for leaf_node in leaf_nodes}

                expr = sp.sympify(trigger_event, locals=symbols)
                if event.isSetId():
                    events_dict[event.id] = expr
                if event.isSetName():
                    events_dict[event.name] = expr

            # Print the delay (if any)
            delay = event.getDelay()
            if delay is not None:
                logger.warn("sbml model has delay event, but this is not supported yet, output might be different")

            # Print the priority (if any)
            priority = event.getPriority()
            if priority is not None:
                logger.warn("sbml model has priority event,but this is not supported yet, output might be different")
    return events_dict
