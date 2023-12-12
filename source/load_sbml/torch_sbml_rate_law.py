from sbml_load_functions import *

class torch_SBML_rate_law(torch.nn.Module):  
    def __init__(self,
            sbml_reaction,
            parameter_dict,
            metabolite_names):
        super(torch_SBML_rate_law, self).__init__()

        self.species=get_reaction_species(sbml_reaction)
        self.string_rate_law=get_string_expression(sbml_reaction)
        local_parameters=get_parameters_for_evaluation(reaction=sbml_reaction,
                                                      parameter_dict=parameter_dict)
        
        
        # subset_parameters = {key: parameter_dict[key] for key in subset_keys}
        local_parameters['t']=torch.Tensor([0])#required because we need to add t in the beginning
        # subset_parameters=fill_in_assignment_rules(subset_parameters)
        self.parameter_dict=local_parameters
        # self.parameter_dict=subset_parameters
        # self.parameter_dict['tanh']=torch.tanh #this is super ugly but it works
        self.metabolite_names=metabolite_names
        self.evaluation=compile(self.string_rate_law,"<string>","eval")

    def calculate(self,concentrations):
        ## This will be done slightly different. Instead of subsetting substrates, products, modifiers
        #Add all of them and let the eval function sort it out. This makes it simpler to code
        # However, then we have to pass everything in the forward pass
        temp_dict=dict(zip(self.metabolite_names,concentrations))
        m={i:torch.Tensor([temp_dict[i]]) for i in self.species if i in self.metabolite_names} #this is buggy?
        eval_dict=self.parameter_dict
        eval_dict={**eval_dict,**m}
        v=eval(self.evaluation,eval_dict)

        return torch.Tensor([v])