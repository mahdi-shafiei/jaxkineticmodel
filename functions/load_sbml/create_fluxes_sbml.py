

from helper_functions import *
import sympy as sp


class torch_SBML_rate_law(torch.nn.Module):
    def __init__(self,
        model,
        sbml_reaction_id, #the reaction in model
        global_parameters :dict, #gathered parameters
        constant_boundaries: dict, #boundary conditions
        func_dict:dict, # lambda functionals
        compartments:dict, #compartments that are included in model
        metabolite_names):  
        super(torch_SBML_rate_law, self).__init__()

        reaction=model.reactions[sbml_reaction_id]
        self.species=get_reaction_species(reaction)
        self.string_rate_law=get_string_expression(reaction)
        print("X", self.string_rate_law)
        self.func_dict=func_dict
        

            
        ### global and local parameters
        self.global_parameters={key:torch.nn.Parameter(torch.Tensor([value])) for key,value in global_parameters.items()}
        self.global_parameters=torch.nn.ParameterDict(self.global_parameters)


        local_parameters=get_local_parameters(reaction)

        self.local_parameters={key:torch.nn.Parameter(torch.Tensor([value])) for key,value in local_parameters.items()}
        self.local_parameters=torch.nn.ParameterDict(self.local_parameters)
        self.num_local_parameters=len(self.local_parameters)

        self.boundary_dict=constant_boundaries
        self.compartment_dict=compartments
        self.metabolite_names=metabolite_names

        temp_dict=dict(zip(self.metabolite_names,torch.zeros(len(self.metabolite_names)))) #required for initialization, will be written over in calculate
        self.eval_dict={**self.local_parameters,**temp_dict,**self.boundary_dict,**self.compartment_dict,**self.global_parameters,**self.func_dict}


        #sympify the string rate law
        sp_symbols_dict=get_reaction_symbols_dict(self.eval_dict)

        self.expr=sp.sympify(self.string_rate_law,locals=sp_symbols_dict)
        print(self.expr)
        # print(expr)
        self.func=sp.lambdify(list(self.eval_dict.keys()),self.expr)


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
    # Dictionary to store computed flux values for each reaction
    v = {}
    func_dict=get_function_dfn_names(model)

    # Iterate over all reactions in the model
    for i in range(model.getNumReactions()):

        # Get the torch_rate_law class
        v_i = torch_SBML_rate_law(model=model,
                                  sbml_reaction_id=i,
                                  global_parameters=parameter_dict,
                                  constant_boundaries=boundaries,
                                  func_dict=func_dict,
                                  compartments=compartments,
                                  metabolite_names=initial_concentration_dict.keys())
        
        # Store the computed flux value for the current reaction
        v[model.reactions[i].id] = v_i
    
    # Return the dictionary containing computed flux values for all reactions
    return v



class torch_SBML_kinetic_model(torch.nn.Module):
    def __init__(self,
                 model,
                 fluxes): #metabolites might not be necessary.
        super(torch_SBML_kinetic_model,self).__init__()
        self.fluxes=nn.ParameterDict(fluxes)
        self.metabolite_names=list(get_initial_conditions(model).keys())
        n_parameters=len(list(self.fluxes.parameters()))
        print("global and local parameters", n_parameters)

        #get stoichiometric info
        self.stoich={}
        for i in self.metabolite_names:
            temp=get_stoichiometry_for_species(model,i)
            self.stoich[i]=temp
        if len(self.stoich)!=len(self.metabolite_names):
            print("mismatch between metabolites and rows (metabolites) of stoichiometry")


    def calculate_fluxes(self,_,concentrations):
        for i in self.fluxes:
            self.fluxes[i].value=self.fluxes[i].calculate(concentrations) 
            # print(self.fluxes[i].value)
        
    def forward(self,_,conc_in):
        self.calculate_fluxes(_,conc_in)
        dXdt=torch.Tensor([])
        for k,i in enumerate(self.metabolite_names):
            if len(self.stoich[i])!=0: #check whether stoichiometry is not empty (e.g. it is a modifier)
                x=sum([self.fluxes[j].value*self.stoich[i][j] for j in self.stoich[i]])
            else:
                x=torch.Tensor([0])
            dXdt=torch.cat([dXdt,x],dim=0)
        return dXdt

