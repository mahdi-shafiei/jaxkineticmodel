import sympy as sp
import jax
import libsbml
import jax.numpy as jnp
import numpy as np



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
            continue
            # print("Constant Boundary Specimen ",specimen.id)
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
        if specimen.getConstant():
            constant_boundary_dict[specimen.id]=specimen.initial_concentration
    return constant_boundary_dict


def get_local_parameters(reaction):
    """Some sbml models also have local parameters (locally defined for reactions), this function retrieves them for an individual reaction, removing the chance 
    similarly named parameters are overwritten"""
    local_parameter_dict={}
    id=reaction.id
    r=reaction.getKineticLaw()
    local_keys=[]
    for i in range(len(r.getListOfParameters())):
        
        global_key=id+"_"+r.getListOfParameters()[i].id
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
    lambda_funcs=nested_local_dict['lambda_functions'] 
    boundary=nested_local_dict['boundary'] # not learnable
    # print("boundary",boundary)

    compartments=nested_local_dict['compartments'] #learnable
    local_dict={**species,**globals,**locals}
    # print(local_dict)
    equation=sp.sympify(equation,locals={**local_dict,
                                         **compartments,**boundary,**lambda_funcs})
    #free symbols are used for lambdifying
    free_symbols = equation.free_symbols
    # print(free_symbols)
    filtered_dict = {key: value for key, value in local_dict.items() if value in free_symbols}
    # print(filtered_dict)
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
    
    lambda_functions=get_function_dfn_names(model)

    v={} 
    v_filtered_dict={} #all symbols that are used in the equation.
    for i in range(nreactions):
        reaction=model.reactions[i] #will be looped by nreactions
        local_parameters=get_local_parameters(reaction)

        
    # reaction_species=get_reaction_species(reaction)


        nested_dictionary_vi={'species':species_ic,
                            'globals':global_parameters,
                            "locals":local_parameters,
                            "compartments":compartments,
                            "boundary":constant_boundaries,
                            "lambda_functions":lambda_functions} #add functionality


        vi_rate_law=get_string_expression(reaction)
        vi,filtered_dict=sympify_lambidify_and_jit_equation(vi_rate_law,nested_dictionary_vi)

        v[model.reactions[i].id]=vi
        v_filtered_dict[model.reactions[i].id]=filtered_dict
    return v,v_filtered_dict


def get_stoichiometric_matrix(model):
    """Retrieves the stoichiometric matrix from the model. This code was taken from
    https://gist.github.com/lukauskas/d1e30bdccc5b801d341d. A minor mistake was found in that the
    stoichiometric coefficients were reversed"""
    species = [s.getName() for s in model.getListOfSpecies()]
    reactions = [r.getId() for r in model.getListOfReactions()]

    stoichiometry_matrix = np.zeros((len(species),len(reactions)))
    for reaction_index, reaction in enumerate(model.getListOfReactions()):

        reactants = {r.getSpecies(): r.getStoichiometry() for r in reaction.getListOfReactants()}
        products = {p.getSpecies(): p.getStoichiometry() for p in reaction.getListOfProducts()}

        for species_index, species_node in enumerate(model.getListOfSpecies()):
            species_id = species_node.getId()


            net_stoichiometry = -int(reactants.get(species_id, 0)) + int(products.get(species_id, 0))
            # print(net_stoichiometry)
            stoichiometry_matrix[species_index, reaction_index] = net_stoichiometry

    species_names = [s.getId() for s in model.getListOfSpecies()]
    reaction_names=[r.getId() for r in model.getListOfReactions()]
    return stoichiometry_matrix,species_names,reaction_names

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
        flux_point_dict[k]=filtered_dict
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
        flux_point_dict[k]=filtered_dict
    return flux_point_dict


def get_function_dfn_names(model):
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
        sp_symbols={}
        for i in range(n_nodes-1):
            symbol=math.getChild(i).getName()
            symbols.append(symbol)
            sp_symbols[symbol]=sp.Symbol(symbol)
        expr=sp.sympify(string_math,locals=sp_symbols)
        print("get_function_dfn_names fnc, ", expr)
        func_x=sp.lambdify(symbols,expr, "jax")
        
        functional_dict[id]=func_x
    return functional_dict