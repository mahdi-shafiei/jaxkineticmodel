class Monod_Model(torch.nn.Module):
    def __init__(self,fluxes,metabolites):
        super(Monod_Model,self).__init__()
        self.fluxes=nn.ParameterDict(fluxes)
        self.metabolites=metabolites

    def calculate_fluxes(self,concentrations):
        self.fluxes['mu_L'].value=self.fluxes['mu_L'].calculate(concentrations[self.metabolites['LACT']])
        self.fluxes['mu_A'].value=self.fluxes['mu_A'].calculate(concentrations[self.metabolites['ACT']])
        self.fluxes['mu_P'].value=self.fluxes['mu_P'].calculate(concentrations[self.metabolites['PYR']])
        self.fluxes['r_AL'].value=self.fluxes['r_AL'].calculate(concentrations[self.metabolites['LACT']]*concentrations[self.metabolites['X']])
        self.fluxes['r_PL'].value=self.fluxes['r_PL'].calculate(concentrations[self.metabolites['LACT']]*concentrations[self.metabolites['X']])
        self.fluxes['r_AP'].value=self.fluxes['r_AP'].calculate(concentrations[self.metabolites['PYR']]*concentrations[self.metabolites['X']])
    
    def forward(self,_,conc_in):
        self.calculate_fluxes(conc_in)

        # Additional parameters needed
        Yxl=torch.Tensor([17.0])
        Yxa=torch.Tensor([11.1])
        Yxp=torch.Tensor([16.7])
        K_e=torch.Tensor([0.013])

        # if _<7.1: #Enforce lag-time
        #     LACT=torch.Tensor([0.0])
        #     ACT=torch.Tensor([0.0])
        #     PYR=torch.Tensor([0.0])
        #     X=torch.Tensor([0.0])
        # elif _>=7.1:
        LACT=((-(conc_in[self.metabolites['X']] * self.fluxes['mu_L'].value) /Yxl)-
                self.fluxes['r_PL'].value-self.fluxes['r_AL'].value)
        
        ACT=(self.fluxes['r_AL'].value+ self.fluxes['r_AP'].value - (conc_in[self.metabolites['X']]*self.fluxes['mu_A'].value)/Yxa)
        
        PYR=self.fluxes['r_PL'].value - ((conc_in[self.metabolites['X']]*self.fluxes['mu_P'].value)/Yxp)-self.fluxes['r_AP'].value

        X=conc_in[self.metabolites['X']]*(self.fluxes['mu_A'].value+self.fluxes['mu_P'].value+self.fluxes['mu_L'].value-K_e)
        return torch.cat([LACT,ACT,PYR,X],dim=0)