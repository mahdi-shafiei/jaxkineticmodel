from torchdiffeq import odeint_adjoint,odeint
import libsbml
import torch

def load_sbml_model(file_path):
    """loading model from file_path"""
    reader=libsbml.SBMLReader()
    document=reader.readSBML(file_path)
    print("Number of internal inconsistencies",document.checkInternalConsistency())

    model=document.getModel()
    print("Number of species:",model.getNumSpecies())
    print("Number of reactions:",model.getNumReactions())
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
            print("Specimen ",specimen.id, " is a constant boundary condition")
        else:   
            initial_concentration_dict[species[i].id]=specimen.initial_concentration
    return initial_concentration_dict

def get_model_parameters(model):
    """retrieves parameters from the loaded sbml model. 
    These will be passed to the Torch Kinetic mechanism class.
    
    Values of dictionary keys that are nan are later checked and filled in the get_parameters_for_evaluation function()
    """
    parameter_dict={}
    params=model.getListOfParameters()
    for i in range(len(params)):
        parameter_dict[params[i].id]=torch.Tensor([params[i].value])
    compartments=model.getListOfCompartments()
    #compartments are added to the parameter_dict because they are need in the evaluation
    for i in range(len(compartments)):
        parameter_dict[compartments[i].id]=torch.Tensor([compartments[i].size])
    
    #If species is a boundary, this should be added to the parameter_dict, since it does
    #not have a rate law
    species=model.getListOfSpecies()
    for i in range(len(species)):
        specimen=species[i]
        if specimen.getConstant() and specimen.getBoundaryCondition():
            parameter_dict[specimen.id]=torch.Tensor([specimen.initial_concentration])
    return parameter_dict

#this function might not be required.
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

def get_parameters_for_evaluation(string_rate_law,parameter_dict):
    """retrieve parameters that are used for evaluating the expression"""
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
        #perhaps we need to add substrates, products, and modifiers later
    return keys

def evaluate_piecewise_expression(expression,t):
    """Evaluate a piecewise expression. This function is very specific to
    the yeast model ss model, might be not important."""
    expression=expression.replace("piecewise(","")
    # splitted_expression=expression.split("), ")
    expression=expression.split(", lt")
    values=[]
    condition_values=[]
    for i in expression:
        i=i.replace("(","")
        i=i.replace(")","")
        i=i.split(",")
        if len(i)==1:
            values.append(float(i[0]))
        else:
            values.append(float(i[2]))
            condition_values.append(float(i[1]))
    condition_values.insert(0,0)
    for i in range(len(condition_values)-1):
        if t>=condition_values[i] and t<condition_values[i+1]:
            expression_eval=values[i]
        elif t>=condition_values[i+1]:
            expression_eval=values[i+1]
    return torch.Tensor([expression_eval])

def fill_in_assignment_rules(subset_parameters):
    """for the dictionary specific for a flux, fill in any value that is defined by some assignment rule
    Now this function only takes piecewise functions, but for other expressions it might be enough to do an evaluation of the rule. Worries for later.
    """
    t=subset_parameters['t']
    for i in subset_parameters.keys():
        rule=model.getRuleByVariable(i)
        if rule!=None:
            string_assignment_law=rule.getFormula()
            value=evaluate_piecewise_expression(string_assignment_law,t)
            subset_parameters[i]=value
    return subset_parameters