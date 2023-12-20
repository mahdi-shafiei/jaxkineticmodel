import torch
import libsbml
from torch import nn

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


class torch_SBML_rate_law(torch.nn.Module):  
    def __init__(self,
            sbml_reaction,
            parameter_dict,
            boundary_dict,
            compartment_dict,
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
        
        self.metabolite_names=metabolite_names



        self.evaluation=compile(self.string_rate_law,"<string>","eval")

    def calculate(self,concentrations):
        ## This will be done slightly different. Instead of subsetting substrates, products, modifiers
        #Add all of them and let the eval function sort it out. This makes it simpler to code
        # However, then we have to pass everything in the forward pass
        temp_dict=dict(zip(self.metabolite_names,concentrations))
        m={i:torch.Tensor([temp_dict[i]]) for i in self.species if i in self.metabolite_names} #this is buggy?
        eval_dict=self.local_parameters
        eval_dict={**eval_dict,**m,**self.boundary_dict,**self.compartment_dict}
        v=eval(self.evaluation,eval_dict)
        return v
        
def create_fluxes(parameter_dict,boundaries,compartments,model):
    initial_concentration_dict=get_initial_conditions(model)
    v={}
    for i in range(model.getNumReactions()):
        v_i=torch_SBML_rate_law(sbml_reaction=model.reactions[i],
                                parameter_dict=parameter_dict,
                                boundary_dict=boundaries,
                                compartment_dict=compartments,
                                metabolite_names=initial_concentration_dict.keys())
        v[model.reactions[i].id]=v_i
    return v
