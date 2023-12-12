import torch
from torch import nn

class MAPK_Signalling(torch.nn.Module):
    def __init__(self,fluxes,metabolites):
        super(MAPK_Signalling,self).__init__()
        self.fluxes=nn.ParameterDict(fluxes)
        self.metabolites=metabolites


    def calculate_fluxes(self,concentrations):
        
        # Two fluxes are calculated a bit different here. The vmax is dependent on the enzyme concentration, and this can differ. This is something that we might like to encounter for in later stages
        # But for now, we will effectively change the parameter by hand.
        self.fluxes['mapkkk'].vmax=0.025*concentrations[self.metabolites['MAPKKKP']]
        self.fluxes['mapkk'].vmax=0.025*concentrations[self.metabolites['MAPKKP']]

        #### Add the concentrations as a dictionary below
        self.fluxes['receptor'].value=self.fluxes['receptor'].calculate(concentrations[self.metabolites['MAPKKK']],concentrations[self.metabolites['MAPKP']])
        self.fluxes['phosphatase1'].value=self.fluxes['phosphatase1'].calculate(concentrations[self.metabolites['MAPKKKP']])
        self.fluxes['mapkkk'].value=self.fluxes['mapkkk'].calculate(concentrations[self.metabolites['MAPKK']])
        self.fluxes['phosphatase2'].value=self.fluxes['phosphatase2'].calculate(concentrations[self.metabolites['MAPKKP']])
        self.fluxes['mapkk'].value=self.fluxes['mapkk'].calculate(concentrations[self.metabolites['MAPK']])
        self.fluxes['phosphatase3'].value=self.fluxes['phosphatase3'].calculate(concentrations[self.metabolites['MAPKP']])

    def forward(self, _, conc_in):
        self.calculate_fluxes(conc_in)
        MAPKKK= -self.fluxes['receptor'].value + self.fluxes['phosphatase1'].value

        MAPKP=- self.fluxes['phosphatase3'].value + self.fluxes['mapkk'].value

        MAPKKKP= + self.fluxes['receptor'].value - self.fluxes['phosphatase1'].value

        MAPKK=-self.fluxes['mapkkk'].value + self.fluxes['phosphatase2'].value
        
        MAPKKP=self.fluxes['mapkkk'].value -self.fluxes['phosphatase2'].value

        MAPK= -self.fluxes['mapkk'].value + self.fluxes['phosphatase3'].value

        return torch.cat([MAPKKK,MAPKP,MAPKKKP,MAPKK,MAPKKP,MAPK],dim=0)
