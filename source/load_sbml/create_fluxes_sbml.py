import torch
import libsbml
from torch import nn
import time
import sympy as sp
import re as re

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

def get_symbolic_function_expr(model):
    """Some sbml models have seperate functions defined. This helper function
    gathers them. 
    
    INPUT:
    1. model """
    functional_dict={}
    for i in range(model.getNumFunctionDefinitions()):
        func=model.function_definitions[i]
        symbol_keys=[]
        func_math=func.getMath()
        for j in range(func_math.getNumChildren()):
            Child=func_math.getChild(j)
            if Child.getName()!=None:
                symbol_keys.append(Child.getName())
        #this is the last term and the expression
        rate_law=libsbml.formulaToL3String(Child)
        expr=sp.sympify(rate_law)
        func_x=sp.lambdify(symbol_keys,expr)
        functional_dict[model.function_definitions[i].getId()]=func_x
    return functional_dict

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

def get_string_expression(reaction):
    #retrieves the kinetic rate law from the sbml model
    kinetic_law=reaction.getKineticLaw()  
    klaw_math=kinetic_law.math
    string_rate_law=libsbml.formulaToString(klaw_math)
    # here we sometimes need to add exceptions. For example, to evaluate tanh, we need to replace it with torch.Tanh
    string_rate_law=string_rate_law.replace("^","**")
    return string_rate_law


def get_parameters_for_evaluation(reaction,parameter_dict):
    """retrieve parameters that are used for evaluating the expression"""
    #retrieve parameters
    id=reaction.id
    kinetic_law=reaction.getKineticLaw()  
    klaw_math=kinetic_law.math
    string_rate_law=libsbml.formulaToString(klaw_math)

    #parse string
    operators_to_remove = ["+","-" ,"*","/","^","(",")",","]
    temp=string_rate_law
    for i in operators_to_remove:
        temp=temp.replace(i,"~") #replace everywhere in between with an extra space, because if you remove ( then tan and n will be together.
    temp=temp.replace(" ","")
    splitted_rate_law=temp.split("~")     
    keys=[]
    for i in splitted_rate_law:
        i=i.strip()
        
        if i in parameter_dict:
            keys.append(i)
        k=reaction.id+"_"+i
        if k in parameter_dict:
            keys.append(k)
        #perhaps we need to add substrates, products, and modifiers later    
    #add local parameters that are necessary for flux calculations
    local_parameter_dict={key: parameter_dict[key] for key in keys}
    #replace key value with a different key
    local_parameter_dict={key.replace(reaction.id+"_",""): value for key, value in local_parameter_dict.items()}
    return local_parameter_dict

def get_symbolic_function_expr(model):
    """Some sbml models have seperate functions defined. This helper function
    gathers them. 
    
    INPUT:
    1. model """
    functional_dict={}
    for i in range(model.getNumFunctionDefinitions()):
        func=model.function_definitions[i]
        symbol_keys=[]
        func_math=func.getMath()
        for j in range(func_math.getNumChildren()):
            Child=func_math.getChild(j)
            if Child.getName()!=None:
                symbol_keys.append(Child.getName())
        #this is the last term and the expression
        rate_law=libsbml.formulaToL3String(Child)
        expr=sp.sympify(rate_law)
        func_x=sp.lambdify(symbol_keys,expr)
        functional_dict[model.function_definitions[i].getId()]=func_x
    return functional_dict


def symbolic_equation(string_rate_law,eval_dict):
    expr=sp.sympify(string_rate_law)
    func=sp.lambdify(eval_dict.keys(),expr)
    return func


class torch_SBML_rate_law(torch.nn.Module):  
    def __init__(self,
            sbml_reaction, #the reaction in model
            parameter_dict, #gathered parameters
            boundary_dict, #boundary conditions
            compartment_dict, #compartments that are included in model
            functional_dict, #gathered functions that are required for evaluation
            metabolite_names):
        super(torch_SBML_rate_law, self).__init__()

        self.species=get_reaction_species(sbml_reaction)
        self.string_rate_law=get_string_expression(sbml_reaction)
        local_parameters=get_parameters_for_evaluation(reaction=sbml_reaction,
                                                      parameter_dict=parameter_dict)
        self.local_parameters={key:torch.nn.Parameter(torch.Tensor([value])) for key,value in local_parameters.items()}
        self.local_parameters=torch.nn.ParameterDict(self.local_parameters)
        
        self.boundary_dict=boundary_dict
        self.compartment_dict=compartment_dict
        self.functional_dict=functional_dict
        
        self.metabolite_names=metabolite_names
        
        # for the sympy object, we gather all parameters/compartments, metabolites, and other required objects 
        #into one evaluation_dictionary. 
        temp_dict=dict(zip(self.metabolite_names,torch.zeros(len(self.metabolite_names)))) #required for initialization, will be written over in calculate
        self.eval_dict={**self.local_parameters,**temp_dict,**self.boundary_dict,**self.compartment_dict}

        self.eval_dict.update(functional_dict)
        #for 
        self.func=symbolic_equation(self.string_rate_law,self.eval_dict)

        

    def calculate(self,concentrations):
        ## This will be done slightly different. Instead of subsetting substrates, products, modifiers
        #Add all of them and let the eval function sort it out. This makes it simpler to code
        # However, then we have to pass everything in the forward pass
    
        temp_dict=dict(zip(self.metabolite_names,concentrations))
        for key,value in temp_dict.items():
            self.eval_dict[key]=value
        v=self.func(**self.eval_dict)
        return v


def create_fluxes(parameter_dict, boundaries, compartments, model):
    # Get initial concentrations for all metabolites in the model
    initial_concentration_dict = get_initial_conditions(model)
    functional_dict=get_symbolic_function_expr(model)
    # Dictionary to store computed flux values for each reaction
    v = {}

    # Iterate over all reactions in the model
    for i in range(model.getNumReactions()):
        # Get the torch_rate_law class
        v_i = torch_SBML_rate_law(sbml_reaction=model.reactions[i],
                                  parameter_dict=parameter_dict,
                                  boundary_dict=boundaries,
                                  compartment_dict=compartments,
                                  functional_dict=functional_dict,
                                  metabolite_names=initial_concentration_dict.keys())
        
        # Store the computed flux value for the current reaction
        v[model.reactions[i].id] = v_i
    
    # Return the dictionary containing computed flux values for all reactions
    return v
