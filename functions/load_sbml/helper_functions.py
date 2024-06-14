import libsbml
import torch
import numpy as np
from torch import nn
import time
import sympy as sp



def load_sbml_model(file_path):
    """Should belong in load_sbml_model.py"""

    """loading model from file_path"""
    reader=libsbml.SBMLReader()
    document=reader.readSBML(file_path)
    print("Number of internal inconsistencies",document.checkInternalConsistency())

    model=document.getModel()
    print("Number of species:",model.getNumSpecies())
    print("Number of reactions:",model.getNumReactions())
    print("Number of global parameters", model.getNumParameters())
    print("Number of constant boundary metabolites: ",model.getNumSpeciesWithBoundaryCondition())
    print("Number of lambda function defitions: ", len(model.function_definitions))

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
    """Most sbml models have their parameters defined globally, this function retrieves them"""
    global_parameter_dict={}
    params=model.getListOfParameters()
    for i in range(len(params)):
        global_parameter_dict[params[i].id]=params[i].value
    return global_parameter_dict

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



def get_compartments(model):
    """Some sbml models have compartments, retrieves them"""
    compartment_dict={}
    compartments=model.getListOfCompartments()
    for i in range(len(compartments)):
        compartment_dict[compartments[i].id]=compartments[i].size

    return compartment_dict


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
        # print(expr)
        func_x=sp.lambdify(args=symbols,expr=expr)
        
        functional_dict[id]=func_x
    return functional_dict

def get_reaction_symbols_dict(eval_dict):
    ### Ensures that sp.simpify properly works
    symbol_dict={}
    for i in eval_dict.keys():
        if callable(eval_dict[i]): #skip functions symbols
            continue
        else:
            symbol_dict[i]=sp.Symbol(i)
    return symbol_dict

def get_stoichiometry_for_species(model, species_id):
    #Thanks chatGPT
    # Get the species by ID
    species = model.getSpecies(species_id)

    if species is None:
        print(f"Species with ID '{species_id}' not found.")
        return
    
    # Dictionary to store the reactions and their stoichiometries
    reactions_info = {}

    # Iterate through all reactions to find the specified species
    for i in range(model.getNumReactions()):
        reaction = model.getReaction(i)

        # Check for reactants
        for j in range(reaction.getNumReactants()):
            reactant = reaction.getReactant(j)
            if species.getId() == reactant.getSpecies():
                stoichiometry = -1 * reactant.getStoichiometry()
                reactions_info[reaction.getId()] = stoichiometry

        # Check for products
        for j in range(reaction.getNumProducts()):
            product = reaction.getProduct(j)
            if species.getId() == product.getSpecies():
                stoichiometry = product.getStoichiometry()
                reactions_info[reaction.getId()] = stoichiometry
    return reactions_info