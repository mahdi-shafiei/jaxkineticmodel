import sympy as sp
import jax
import libsbml
import jax.numpy as jnp
import numpy as np
import pandas as pd
import re
import collections

def load_sbml_model(file_path):
    """loading sbml model from file_path"""
    reader=libsbml.SBMLReader()
    document=reader.readSBML(file_path)
    print("Number of internal inconsistencies",document.checkInternalConsistency())

    model=document.getModel()
    print("Number of species:",model.getNumSpecies())
    print("Number of reactions:",model.getNumReactions())
    print("Number of global parameters", model.getNumParameters())
    print("Number of constant boundary metabolites: ",model.getNumSpeciesWithBoundaryCondition())
    print("Number of lambda function definitions: ", len(model.function_definitions))
    print("Number of assignment rules",model.getNumRules())
    return model

def get_initial_conditions(model):
    """Retrieves the species initial concentrations 
    from the SBML model. If a species is a constant boundary condition, 
    then it should be passed as a parameter instead of an initial condition,
    since it does not have a rate law"""
    species=model.getListOfSpecies()
    initial_concentration_dict={}
    for i in range(len(species)):
        specimen=species[i]

        #there are also non-stationary boundary conditions, deal with this later. 
        if specimen.getConstant() and specimen.getBoundaryCondition():
            print("Constant Boundary Specimen ",specimen.id)
            continue
        
        elif specimen.getBoundaryCondition() and not specimen.getConstant():
            if model.getLevel()==2:
                print("Level 2 sbml, assume that boundary condition is constant for",specimen.id)
            
            if model.getLevel()==3:
                print("Non-Constant Boundary Specimen", specimen.id)
            continue

        else:   
            initial_concentration_dict[species[i].id]=specimen.initial_concentration
    return initial_concentration_dict

def get_global_parameters(model):
    """Most sbml models have their parameters defined globally, 
    this function retrieves them"""
    global_parameter_dict={}
    params=model.getListOfParameters()
    for i in range(len(params)):
        global_parameter_dict[params[i].id]=params[i].value
    return global_parameter_dict

def get_compartments(model):
    """Some sbml models have compartments, retrieves them"""
    compartment_dict={}
    compartments=model.getListOfCompartments()
    for i in range(len(compartments)):
        compartment_dict[compartments[i].id]=compartments[i].size

    return compartment_dict



## We do not deal yet with non-constant boundaries
def get_string_expression(reaction):
    """retrieves the kinetic rate law from the reaction"""
    kinetic_law=reaction.getKineticLaw()  
    # print(kinetic_law.name)
    klaw_math=kinetic_law.math
    string_rate_law=libsbml.formulaToString(klaw_math)
    # here we sometimes need to add exceptions. For example, to evaluate tanh, we need to replace it with torch.Tanh
    string_rate_law=string_rate_law.replace("^","**")
    return string_rate_law

def get_constant_boundary_species(model):
    """Species that are boundary conditions should be fed as fixed, non-learnable parameters
    https://synonym.caltech.edu/software/libsbml/5.18.0/docs/formatted/python-api/classlibsbml_1_1_species.html"""
    constant_boundary_dict={}
    species=model.getListOfSpecies()
    for i in range(len(species)):
        specimen=species[i]
        if specimen.getBoundaryCondition():
            if model.getLevel()==2:

                print("Assume that boundary is constant for level 2")
                constant_boundary_dict[specimen.id]=specimen.initial_concentration





            constant_boundary_dict[specimen.id]=specimen.initial_concentration
    return constant_boundary_dict


def get_local_parameters(reaction):
    """Some sbml models also have local parameters (locally defined for reactions), this function retrieves them for an individual reaction, removing the chance 
    similarly named parameters are overwritten"""
    local_parameter_dict={}
    id=reaction.id
    r=reaction.getKineticLaw()
    for i in range(len(r.getListOfParameters())):
        
        local_keys=r.getListOfParameters()[i].id
        value=r.getListOfParameters()[i].value
        local_parameter_dict[local_keys]=value
    return local_parameter_dict

def get_reaction_species(reaction):
    """Retrieves the substrates, products, and modifiers from sbml format. 
     These will be passed to the Torch Kinetic Model class. """
    sub=reaction.getListOfReactants()
    prod=reaction.getListOfProducts()
    mod=reaction.getListOfModifiers()
    
    substrates=[]
    products=[]
    modifiers=[]
    for i in range(len(sub)):
        substrates.append(sub[i].species)
    for i in range(len(prod)):
        products.append(prod[i].species)
    for i in range(len(mod)):
        modifiers.append(mod[i].species)

    species=substrates+products+modifiers
    return species


def get_reaction_symbols_dict(eval_dict):
    """This functions works on the local_dictionary passed in the sympify function
    It ensures that sympy symbols for parameters and y-values are properly passed,
    while the rest is simply substituted in the expression."""
    symbol_dict={}
    for i in eval_dict.keys():
        if callable(eval_dict[i]): #skip functions symbols
            continue
        else:
            symbol_dict[i]=sp.Symbol(i)
    return symbol_dict

#we want the output to be a list v, which contains jitted function
def sympify_lambidify_and_jit_equation(equation,nested_local_dict):
    """Sympifies, lambdifies, and then jits a string rate law
    "equation: the string rate law equation
    nested_local_dict: a dictionary having dictionaries of
      global parameters,local parameters, compartments, and boundary conditions
    
      #returns
      the jitted equation
      a filtered dictionary. This will be used to construct the flux_pointer_dictionary. 
    
    """  
    #unpacking the nested_local_dictionary, with global and local parameters symbols
    globals=get_reaction_symbols_dict(nested_local_dict['globals'])
    locals=get_reaction_symbols_dict(nested_local_dict['locals'])
    species=get_reaction_symbols_dict(nested_local_dict['species'])
    boundaries=nested_local_dict['boundary']
    lambda_funcs=nested_local_dict['lambda_functions'] 

    assignment_rules=nested_local_dict['boundary_assignments']
    
    compartments=nested_local_dict['compartments'] #learnable
    local_dict={**species,**globals,**locals}


    equation=sp.sympify(equation,locals={**local_dict,
                                         **assignment_rules,
                                         **lambda_funcs})

    #these are filled in before compiling


    equation=equation.subs(assignment_rules)
    equation=equation.subs(compartments)
    equation=equation.subs(boundaries)


    #free symbols are used for lambdifying
    free_symbols = list(equation.free_symbols)


    filtered_dict = {key: value for key, value in local_dict.items() if value in free_symbols}

    #perhaps a bit hacky, but some sbml models have symbols that are 
    #predefined
    for symbol in free_symbols:
        if str(symbol)=="time":
            filtered_dict['time']=sp.Symbol('time')
    

    equation=sp.lambdify((filtered_dict.values()),equation,"jax")
    equation=jax.jit(equation)


    return equation,filtered_dict


def create_fluxes_v(model):
    """This function defines the jax jitted equations that are used in TorchKinModel
    class
    """
    #retrieve whatever is important
    nreactions=model.getNumReactions()

    species_ic=get_initial_conditions(model)
    global_parameters=get_global_parameters(model)
    compartments=get_compartments(model)
    constant_boundaries=get_constant_boundary_species(model)


    lambda_functions=get_lambda_function_dictionary(model)
    assignments_rules=get_assignment_rules_dictionary(model)


    v={} 

    v_symbol_dict={} #all symbols that are used in the equation.

    local_param_dict={} #local parameters with the reaction it belongs to as a new parameter
    for i in range(nreactions):
        reaction=model.reactions[i] #will be looped by nreactions

        
        local_parameters=get_local_parameters(reaction)
        # print(local_parameters)

        
    # reaction_species=get_reaction_species(reaction)


        nested_dictionary_vi={'species':species_ic,
                            'globals':global_parameters,
                            "locals":local_parameters,
                            "compartments":compartments,
                            "boundary":constant_boundaries,
                            "lambda_functions":lambda_functions,
                            "boundary_assignments":assignments_rules} #add functionality

        
        vi_rate_law=get_string_expression(reaction)
        vi,filtered_dict=sympify_lambidify_and_jit_equation(vi_rate_law,nested_dictionary_vi)

        v[model.reactions[i].id]=vi #the jitted equation
        v_symbol_dict[model.reactions[i].id]=filtered_dict

        # here
        for key in local_parameters.keys():

            newkey="lp."+str(model.reactions[i].id)+"."+key
            local_param_dict[newkey]=local_parameters[key]
    return v,v_symbol_dict,local_param_dict


def get_stoichiometric_matrix(model):
    """Retrieves the stoichiometric matrix from the model. """

    species_ids=[]
    reduced_species_list=[]
    for s in model.getListOfSpecies():
    #these conditions do not have a rate law
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
    reactions = [r.getId() for r in model.getListOfReactions()]

    stoichiometry_matrix = np.zeros((len(species_ids),len(reactions)))
    for reaction_index, reaction in enumerate(model.getListOfReactions()):

        reactants = {r.getSpecies(): r.getStoichiometry() for r in reaction.getListOfReactants()}
        products = {p.getSpecies(): p.getStoichiometry() for p in reaction.getListOfProducts()}

        for species_index, species_node in enumerate(reduced_species_list):
            species_id = species_node.getId()


            net_stoichiometry = -int(reactants.get(species_id, 0)) + int(products.get(species_id, 0))
            # print(net_stoichiometry)
            stoichiometry_matrix[species_index, reaction_index] = net_stoichiometry

    species_names = [s.getId() for s in reduced_species_list]
    reaction_names=[r.getId() for r in model.getListOfReactions()]

    stoichiometry_matrix=pd.DataFrame(stoichiometry_matrix,index=species_names,columns=reaction_names)

    return stoichiometry_matrix

def species_match_to_S(initial_conditions,species_names):
    """Small helper function ensures that y0 is properly matched to rows of S
    Input: initial conditions is a dictionary of initial conditions
    species_names: is the order of the rows in S"""
    y0=[]
    for species in species_names:
        if species in initial_conditions.keys():
            y0.append(initial_conditions[species])
    y0=jnp.array(y0)
    return y0


def reaction_match_to_S(flux_funcs,reaction_names):
    """Small helper function ensures that y0 is properly matched to rows of S
    Input: initial conditions is a dictionary of initial conditions
    species_names: is the order of the rows in S"""
    v=[]
    for reaction in reaction_names:
        v.append(flux_funcs[reaction])
    return v

def construct_flux_pointer_dictionary(v_symbol_dictionaries,reaction_names,species_names):
    """In jax, the values that are used need to be pointed directly in y."""
    flux_point_dict={}
    for k,reaction in enumerate(reaction_names):
        v_dict=v_symbol_dictionaries[reaction]
        filtered_dict=[species_names.index(key) for key in v_dict.keys() if key in species_names]
        filtered_dict=jnp.array(filtered_dict)
        flux_point_dict[reaction]=filtered_dict
    return flux_point_dict

def construct_param_point_dictionary(v_symbol_dictionaries,reaction_names,parameters):
    """In jax, the values that are used need to be pointed directly in y."""
    flux_point_dict={}
    for k,reaction in enumerate(reaction_names):

        v_dict=v_symbol_dictionaries[reaction]
        filtered_dict={}
        for key,value in v_dict.items():
            if key in parameters.keys():
                filtered_dict[key]=parameters[key]
        # params_point_dict=[parameters.index(key) for key in v_dict.keys() if key in parameters]
        # print(params_point_dict)
        # filtered_dict=jnp.array(params_point_dict)
        flux_point_dict[reaction]=filtered_dict
    return flux_point_dict

def get_leaf_nodes(node, leaf_nodes):
    """Finds the leaf nodes of the mathml expression."""
    if node.getNumChildren() == 0:
        name=node.getName()
        if name!=None:
            leaf_nodes.append(name)
        # print(node.getName())
    else:
        for i in range(node.getNumChildren()):
            get_leaf_nodes(node.getChild(i), leaf_nodes)
    leaf_nodes=np.array(leaf_nodes)
    leaf_nodes=np.unique(leaf_nodes)
    leaf_nodes=leaf_nodes.tolist()
    return leaf_nodes


def get_lambda_function_dictionary(model):
    """Stop giving these functions confusing names...
    it returns a dictionary with all lambda functions"""
    functional_dict={}

    for fnc in range(model.getNumFunctionDefinitions()): #loop over function definitions
        function=model.function_definitions[fnc]
        id=function.getId()
        math=function.getMath()
        n_nodes=math.getNumChildren()
        string_math=libsbml.formulaToL3String(math.getChild(n_nodes-1))
        symbols=[]
        leaf_nodes=[]
        sp_symbols={}
        math_nodes=get_leaf_nodes(math,leaf_nodes=leaf_nodes)
        sp_symbols={}
        for node in math_nodes:
            sp_symbols[node]=sp.Symbol(node)
        expr=sp.sympify(string_math,locals=sp_symbols)

        func_x=sp.lambdify(math_nodes,expr, "jax")
        
        functional_dict[id]=func_x
    return functional_dict

def get_assignment_rules_dictionary(model):
    """Get rules that assign to variables. I did not lambdify here"""
    assignment_dict={}
    for fnc in range(model.getNumRules()):
        rule=model.rules[fnc]
        id=rule.getId()
        math=rule.getMath()
        leaf_nodes=[]
        string_math=libsbml.formulaToL3String(math)

        math_nodes=get_leaf_nodes(math,leaf_nodes=leaf_nodes)
        sp_symbols={}
        for node in math_nodes:
            sp_symbols[node]=sp.Symbol(node)
        expr=sp.sympify(string_math,locals=sp_symbols)
        # rule_x=sp.lambdify(math_nodes,expr, "jax")
        
        assignment_dict[id]=expr
        # print("the expression: ",id,expr)
    return assignment_dict


def separate_params(params):
    global_params={}
    local_params=collections.defaultdict(dict)

    for key in params.keys():

        if re.match("lp.*.",key):

            fkey=key.removeprefix("lp.")
            list=fkey.split(".")
            value=params[key]
            newkey=list[1]
            local_params[list[0]][newkey]=value
            
            
        else:
            global_params[key]=params[key]
    return global_params,local_params


# def wrap_time_symbols(t):
def time_dependency_symbols(v_symbol_dictionaries,t):
    time_dependencies={}
    for key,values in v_symbol_dictionaries.items():
        for value in values.keys():
            if value=="time":
                time_dependencies[key]={value:t}
            else:
                time_dependencies[key]={}
    return time_dependencies
#   time_dependencies=time_dependency_symbols(v_symbol_dictionaries,t)
#   return time_dependencies