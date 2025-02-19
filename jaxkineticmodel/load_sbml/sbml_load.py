import sympy as sp
import libsbml
import jax.numpy as jnp
import numpy as np
import pandas as pd
import re
import collections
import os

# Suddenly this jaxkineticmodel.utils doesnt load anymore without adding path directly. Make an init file?
import sys



from jaxkineticmodel.utils import get_logger


logger = get_logger(__name__)


def load_sbml_model(file_path):
    """loading sbml model from file_path"""
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError()

    reader = libsbml.SBMLReader()
    document = reader.readSBML(file_path)
    logger.info(f"Number of internal inconsistencies: {document.checkInternalConsistency()}")

    model = document.getModel()
    print("Number of species:", model.getNumSpecies())
    print("Number of reactions:", model.getNumReactions())
    print("Number of global parameters", model.getNumParameters())
    print("Number of constant boundary metabolites: ", model.getNumSpeciesWithBoundaryCondition())
    print("Number of lambda function definitions: ", len(model.function_definitions))
    print("Number of assignment rules", model.getNumRules())
    print("Number of events", model.getNumEvents())
    return model


def get_initial_conditions(model):
    """Retrieves the species initial concentrations
    from the SBML model. If a species is a constant boundary condition,
    then it should be passed as a parameter instead of an initial condition,
    since it does not have a rate law"""
    species = model.getListOfSpecies()
    initial_concentration_dict = {}
    for specimen in species:
        # there are also non-stationary boundary conditions, deal with this later.
        if specimen.isSetConstant() and specimen.isSetBoundaryCondition():
            if specimen.getConstant() and specimen.getBoundaryCondition():
                print("Constant Boundary Specimen ", specimen.id)
                continue

            elif specimen.getBoundaryCondition() and not specimen.getConstant():
                ## these will be passed to the sympy expression by get_boundaries
                # initial_concentration_dict[specimen.id] = specimen.initial_concentration
                continue

            else:
                initial_concentration_dict[specimen.id] = specimen.initial_concentration

        else:
            logger.warn(f"Constant and Boundary condition boolean are not set for {specimen}")
    return initial_concentration_dict


def get_global_parameters(model):
    """Most sbml models have their parameters defined globally,
    this function retrieves them"""
    params = model.getListOfParameters()
    global_parameter_dict = {}
    for param in params:
        if param.isSetConstant():
            if param.getConstant() is True:
                global_parameter_dict[param.id] = param.value
            if param.getConstant() is False:
                global_parameter_dict[param.id] = param.value
    return global_parameter_dict


def get_initial_assignments(model, global_parameters, assignment_rules, y0):
    """Some sbml assign values through the list of initial assignments. This should be used
    to overwrite y0 where necessary. This can be done outside the model structure"""

    initial_assignments = {}
    for init_assign in model.getListOfInitialAssignments():
        if init_assign.id in y0.keys():
            math = init_assign.getMath()
            math_string = libsbml.formulaToL3String(math)
            sympy_expr = sp.sympify(math_string, locals={**assignment_rules, **global_parameters})
            # sympy_expr=sympy_expr.subs(global_parameters)
            if type(sympy_expr) is not float:
                sympy_expr = sympy_expr.subs(global_parameters)
                sympy_expr = sympy_expr.subs(y0)
                sympy_expr = np.float64(sympy_expr)
            initial_assignments[init_assign.id] = sympy_expr
    return initial_assignments


def overwrite_init_conditions_with_init_assignments(model, global_parameters, assignment_rules, y0):
    """y0 values are initialized in sbml, but some models also define initial assignments
    These should be leading and be passed to y0"""
    initial_assignments = get_initial_assignments(model, global_parameters, assignment_rules, y0)
    # initial_assignments=get_initial_assignments(model,params)
    for key in initial_assignments.keys():
        if key in y0.keys():
            y0[key] = initial_assignments[key]
    return y0


def get_compartments(model):
    """Some sbml models have compartments, retrieves them"""
    compartments = model.getListOfCompartments()
    compartment_dict = {cmp.id: cmp.size for cmp in compartments}
    return compartment_dict


def get_events_dictionary(model):
    """There are many type of events. For now I only add a few and the rest will throw a warning"""
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
        if specimen.isSetConstant() and specimen.isSetBoundaryCondition():
            if specimen.getBoundaryCondition() and specimen.getConstant():
                # if it is both a boundary condition and a constant,
                # then we pass it as a constant boundary that will be filled into the sympy equation
                print("constant boundary", specimen.id)
                constant_boundary_dict[specimen.id] = specimen.initial_concentration

            elif specimen.getConstant() and not specimen.getBoundaryCondition():
                print("constant non-boundary", specimen.id)
                # if it is not a boundary condition but it is a constant,
                # then we pass it as a constant boundary that will be filled into the sympy equation

                constant_boundary_dict[specimen.id] = specimen.initial_concentration
            elif (
                specimen.getBoundaryCondition() and not specimen.getConstant()
            ):  # values are either defined by a rule/event or rate law
                logger.info(f"{specimen.id} is boundary but not constant ")
                if specimen.isSetInitialConcentration():
                    constant_boundary_dict[specimen.id] = specimen.initial_concentration

                # print(specimen.get)
                continue
        else:
            logger.warn(("Constant and Boundary conditions were not set for level 2 we assume that boundary is constant"))
            #     print(specimen)
            if model.getLevel() == 2:
                constant_boundary_dict[specimen.id] = specimen.initial_concentration

    logger.info(constant_boundary_dict)
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


# # we want the output to be a list v, which contains jitted function
# def sympify_lambidify_and_jit_equation(equation, nested_local_dict):
#     """Sympifies, lambdifies, and then jits a string rate law
#     "equation: the string rate law equation
#     nested_local_dict: a dictionary having dictionaries of
#       global parameters,local parameters, compartments, and boundary conditions

#       #returns
#       the jitted equation
#       a filtered dictionary. This will be used to construct the flux_pointer_dictionary.

#     """
#     # unpacking the nested_local_dictionary, with global and local parameters symbols
#     globals = get_reaction_symbols_dict(nested_local_dict['globals'])
#     locals = get_reaction_symbols_dict(nested_local_dict['locals'])
#     species = get_reaction_symbols_dict(nested_local_dict['species'])
#     boundaries = nested_local_dict['boundary']
#     lambda_funcs = nested_local_dict['lambda_functions']

#     assignment_rules = nested_local_dict['boundary_assignments']
#     rate_rules=nested_local_dict['rate_rules']
#     event_rules=nested_local_dict['event_rules']


#     compartments = nested_local_dict['compartments']  # learnable
#     local_dict = {**species, **globals, **locals}


#     # print("input equation:",equation)
#     equation = sp.sympify(equation, locals={**local_dict,
#                                             **assignment_rules,
#                                             **lambda_funcs,**rate_rules,**event_rules})
#     # print("after sympifying",equation)
#     # these are filled in before compiling

#     # print("assignment",assignment_rules)


#     # print("assignment eq",equation)
#     equation = equation.subs(compartments)
#     equation = equation.subs(boundaries)

#     # print("c",equation)
#     equation=equation.subs(event_rules)
#     equation = equation.subs(assignment_rules)
#     equation=equation.subs({"pi":3.14159265359})
#     print(equation)
#     # print("output after substitution",equation)

#     # free symbols are used for lambdifying
#     free_symbols = list(equation.free_symbols)
#     # print(free_symbols)
#     filtered_dict = {key: value for key, value in local_dict.items() if value in free_symbols or key in locals}

#     # perhaps a bit hacky, but some sbml models have symbols that are
#     # predefined
#     for symbol in free_symbols:
#         if str(symbol) == "time":
#             filtered_dict['time'] = sp.Symbol('time')

#     equation = sp.lambdify((filtered_dict.values()), equation, "jax")
#     equation = jax.jit(equation)

#     return equation, filtered_dict


def get_stoichiometric_matrix(model):
    """Retrieves the stoichiometric matrix from the model."""

    species_ids = []
    reduced_species_list = []
    for s in model.getListOfSpecies():
        # these conditions do not have a rate law
        if s.isSetConstant() and s.isSetBoundaryCondition():
            if s.getConstant() and s.getBoundaryCondition():
                # is a boundary and a constant, does not have stoichiometry
                continue
            elif s.getBoundaryCondition() and not s.getConstant():
                # is a boundary and but not a constant. has a stoichiometry?
                # reduced_species_list.append(s)
                # species_ids.append(s.getId())
                continue

            elif not s.getBoundaryCondition() and s.getConstant():
                continue

            elif not s.getBoundaryCondition() and not s.getConstant():
                reduced_species_list.append(s)
                species_ids.append(s.getId())
        else:
            logger.warn("Constant and Boundary booleans are not set")

    # species = [s.getName() for s in model.getListOfSpecies()]
    reactions = [r.getId() for r in model.getListOfReactions()]

    stoichiometry_matrix = np.zeros((len(species_ids), len(reactions)))
    for reaction_index, reaction in enumerate(model.getListOfReactions()):
        reactants = {r.getSpecies(): r.getStoichiometry() for r in reaction.getListOfReactants()}
        products = {p.getSpecies(): p.getStoichiometry() for p in reaction.getListOfProducts()}

        for species_index, species_node in enumerate(reduced_species_list):
            species_id = species_node.getId()

            net_stoichiometry = -int(reactants.get(species_id, 0)) + int(products.get(species_id, 0))
            # print(net_stoichiometry)
            stoichiometry_matrix[species_index, reaction_index] = net_stoichiometry

    species_names = [s.getId() for s in reduced_species_list]
    reaction_names = [r.getId() for r in model.getListOfReactions()]

    stoichiometry_matrix = pd.DataFrame(stoichiometry_matrix, index=species_names, columns=reaction_names)

    return stoichiometry_matrix


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


def construct_flux_pointer_dictionary(v_symbol_dictionaries, reaction_names, species_names):
    """In jax, the values that are used need to be pointed directly in y."""
    flux_point_dict = {}
    for k, reaction in enumerate(reaction_names):
        v_dict = v_symbol_dictionaries[reaction]
        filtered_dict = [species_names.index(key) for key in v_dict.keys() if key in species_names]
        filtered_dict = jnp.array(filtered_dict)
        flux_point_dict[reaction] = filtered_dict
    return flux_point_dict


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


def get_ordered_symbols(expr):
    """Orders based on appearance: thanks chatGPT"""
    ordered_symbols = []

    def traverse(e):
        if isinstance(e, sp.Symbol) and e not in ordered_symbols:
            ordered_symbols.append(e)
        for arg in e.args:
            traverse(arg)

    traverse(expr)
    return ordered_symbols


def get_lambda_function_dictionary(model):
    """Stop giving these functions confusing names...
    it returns a dictionary with all lambda functions"""
    functional_dict = {}

    for function in model.function_definitions:
        id = function.getId()
        math = function.getMath()
        n_nodes = math.getNumChildren()
        string_math = libsbml.formulaToL3String(math.getChild(n_nodes - 1))


        math_nodes = []
        for i in range(function.getNumArguments()):
            math_node_name = function.getArgument(i).getName()
            math_nodes.append(math_node_name)

        sp_symbols = {}
        for node in math_nodes:
            sp_symbols[node] = sp.Symbol(node)
        expr = sp.sympify(string_math, locals=sp_symbols)
        # substitute some operations using sympy expressions. This is a bit ugly

        func_x = sp.lambdify(math_nodes, expr, "jax")

        functional_dict[id] = func_x

    return functional_dict


def get_assignment_rules_dictionary(model):
    """Get all rules that are an assignment rule, not algebraic or rate"""
    assignment_dict = {}
    for rule in model.rules:
        if rule.isAssignment():
            id = rule.getId()
            math = rule.getMath()
            leaf_nodes = []
            string_math = libsbml.formulaToL3String(math)

            math_nodes = get_leaf_nodes(math, leaf_nodes=leaf_nodes)
            sp_symbols = {node: sp.Symbol(node) for node in math_nodes}
            expr = sp.sympify(string_math, locals=sp_symbols)
            # rule_x = sp.lambdify(math_nodes, expr, "jax")

            assignment_dict[id] = expr
        # print("the expression: ",id,expr)
    return assignment_dict


def get_rate_rules_dictionary(model):
    """Retrieve all rate_rules, and replace reactions with their respective kinetic laws"""
    rate_rules_dict = {}
    for rule in model.rules:
        if rule.isRate():
            id = rule.getId()
            math = rule.getMath()
            leaf_nodes = []
            string_math = libsbml.formulaToL3String(math)

            math_nodes = get_leaf_nodes(math, leaf_nodes=leaf_nodes)
            sp_symbols = {node: sp.Symbol(node) for node in math_nodes}
            expr = sp.sympify(string_math, locals=sp_symbols)

            # this replaces reactions with the actual kinetic law, to ensure proper
            # sympyfying and jitting
            temp_dict = {}
            for math_node in math_nodes:
                if model.getReaction(math_node) is not None:
                    reaction = model.getReaction(math_node)
                    math = reaction.getKineticLaw()
                    string_math = libsbml.formulaToL3String(math.getMath())

                    temp_dict[math_node] = string_math

            expr = expr.subs(temp_dict)
            # rule_x=sp.lambdify(math_nodes,expr, "jax")
            ## here we should substitute any reaction name with its actual equation

            rate_rules_dict[id] = expr

    return rate_rules_dict


def separate_params(params):
    """Seperates the global from local parameters using a identifier (lp.[Enz].)"""
    global_params = {}
    local_params = collections.defaultdict(dict)

    for key in params.keys():
        if re.match("lp_*_", key):
            fkey = key.removeprefix("lp_")
            list = fkey.split("_")
            value = params[key]
            newkey = list[1]
            local_params[list[0]][newkey] = value
        else:
            global_params[key] = params[key]
    return global_params, local_params


def separate_params_jac(params):
    """Only used to pass parameters locally and globally to the jacobian (see if this is better?)"""
    global_params = {}
    local_params = collections.defaultdict(dict)

    for key in params.keys():
        if re.match("lp.*.", key):
            # fkey = key.removeprefix("lp.")
            # list = fkey.split(".")
            value = params[key]
            # newkey = list[1]
            local_params[key] = value
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
