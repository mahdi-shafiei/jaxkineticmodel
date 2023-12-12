
import torch
from sbml_load_functions import *
from torch_sbml_rate_law import torch_SBML_rate_law
import numpy as np

class torch_SBML_kinetic_model(torch.nn.Module):
    def __init__(self,
                 model,
                 parameter_dict):
        super(torch_SBML_kinetic_model,self).__init__()
        self.metabolite_names=list(get_initial_conditions(model).keys())

        #sanity check: model parameter keys should all be unique
        if len(parameter_dict)!=len(np.unique(list(parameter_dict.keys()))):
            print("number of parameters not unique")
        self.parameter_dict=torch.nn.ParameterDict(parameter_dict)
        
        #set up fluxes
        fluxes={}
        for i in range(model.getNumReactions()):
            v=torch_SBML_rate_law(model.reactions[i],parameter_dict=self.parameter_dict,metabolite_names=self.metabolite_names)
            # print(v.string_rate_law)
            fluxes[model.reactions[i].id]=v
        self.fluxes=fluxes

        #get stoichiometric info
        self.stoich={}
        for i in self.metabolite_names:
            temp=get_stoichiometry_for_species(model,i)
            self.stoich[i]=temp
        if len(self.stoich)!=len(self.metabolite_names):
            print("mismatch between metabolites and rows (metabolites) of stoichiometry")

    def calculate_fluxes(self,_,concentrations):
        for i in self.fluxes:
            self.fluxes[i].parameter_dict["t"]=torch.Tensor([_])
            # self.fluxes[i].evaluation_dictionary["n"]=torch.Tensor([485])
            self.fluxes[i].value=self.fluxes[i].calculate(concentrations)
            
            


    def forward(self,_,conc_in):
        # print(_)
        self.calculate_fluxes(_,conc_in)
        dXdt=torch.Tensor([])
        for k,i in enumerate(self.metabolite_names):
            if len(self.stoich[i])!=0: #check whether stoichiometry is not empty (e.g. it is a modifier)
                x=sum([self.fluxes[j].value*self.stoich[i][j] for j in self.stoich[i]])
            else:
                x=torch.Tensor([0])
            dXdt=torch.cat((dXdt,x),dim=0)
        return dXdt