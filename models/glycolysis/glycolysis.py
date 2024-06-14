import torch
from torch import nn

class Glycolysis(torch.nn.Module):
    def __init__(self,
                 fluxes,
                 metabolites):
        super(Glycolysis, self).__init__()
        self.fluxes = nn.ParameterDict(fluxes) # dict with fluxes
        self.metabolites = metabolites # dict with indices of each metabolite
        
       
    def calculate_fluxes(self, concentrations):
        """
        Calculates the flux values of the glycolysis reactions based on given metabolite concentrations.

        """

        for flux_name, flux in self.fluxes.items():
            if hasattr(flux, 'calculate'):
                required_metabolites_substrate = [concentrations[self.metabolites[m]] for m in flux.substrate_names]
                required_metabolites_product = [concentrations[self.metabolites[m]] for m in flux.product_names]
                required_metabolites_modifiers = [concentrations[self.metabolites[m]] for m in flux.modifiers_names]

                flux_value = flux.calculate(required_metabolites_substrate, required_metabolites_product, required_metabolites_modifiers)
                self.fluxes[flux_name].value = flux_value
            

    #GLYC3P != Glyceral3P != GAP
    def forward(self, _ , conc_in):
        self.calculate_fluxes(conc_in)
        GLCo = 0#conc_in[self.metabolites['ECglucose']]
        TREHo = 0#conc_in[self.metabolites['ECtreh']]
        UDPG = 0
        GLCi = -self.fluxes['v_GLK'].value + self.fluxes['v_GLT'].value + 2.*self.fluxes['v_NTH1'].value
        G6P = + self.fluxes['v_GLK'].value - self.fluxes['v_PGI'].value + self.fluxes['vsinkG6P'].value + self.fluxes['v_PGM1'].value - self.fluxes['v_TPS1'].value
        G1P = - self.fluxes['v_PGM1'].value - 3.1000e-04#self.fluxes['v_UGP'].value
        T6P = + self.fluxes['v_TPS1'].value -self.fluxes['v_TPS2'].value
        TRE = + self.fluxes['v_TPS2'].value - self.fluxes['v_NTH1'].value
        F6P = - self.fluxes['v_PFK'].value + self.fluxes['v_PGI'].value + self.fluxes['vsinkF6P'].value
        F16BP = + self.fluxes['v_PFK'].value - self.fluxes['v_ALD'].value
     
        GAP = + self.fluxes['v_ALD'].value - self.fluxes['v_GAPDH'].value + self.fluxes['v_TPI'].value + self.fluxes['vsinkGAP'].value #GLYCERAL3P
        DHAP = + self.fluxes['v_ALD'].value - self.fluxes['v_TPI'].value - self.fluxes['v_G3PDH'].value
        G3P = + self.fluxes['v_G3PDH'].value - self.fluxes['v_HOR2'].value #GLYC3P
        GLYCEROL = + self.fluxes['v_HOR2'].value - self.fluxes['v_GlycT'].value
        BPG = + self.fluxes['v_GAPDH'].value - self.fluxes['v_PGK'].value
        P3G = + self.fluxes['v_PGK'].value - self.fluxes['v_PGM'].value + self.fluxes['vsinkP3G'].value
        P2G = + self.fluxes['v_PGM'].value - self.fluxes['v_ENO'].value
        PEP = + self.fluxes['v_ENO'].value - self.fluxes['v_PYK'].value + self.fluxes['vsinkPEP'].value
        PYR = + self.fluxes['v_PYK'].value - self.fluxes['v_PDC'].value + self.fluxes['vsinkPYR'].value
        ACE = + self.fluxes['v_PDC'].value - self.fluxes['v_ADH'].value + self.fluxes['vsinkACE'].value
        ETOH = + self.fluxes['v_ADH'].value - self.fluxes['v_EtohT'].value
        
        ATP = + self.fluxes['v_ADK1'].value - self.fluxes['v_GLK'].value - self.fluxes['v_ATPase'].value - self.fluxes['v_PFK'].value + \
           self.fluxes['v_PGK'].value + self.fluxes['v_PYK'].value - self.fluxes['v_TPS1'].value + self.fluxes['v_ATPmito'].value
        
        ADP = - 2.*self.fluxes['v_ADK1'].value + self.fluxes['v_GLK'].value + self.fluxes['v_ATPase'].value + \
            self.fluxes['v_PFK'].value - self.fluxes['v_PGK'].value - self.fluxes['v_PYK'].value + self.fluxes['v_TPS1'].value -self.fluxes['v_ATPmito'].value
            
        AMP = + self.fluxes['v_ADK1'].value - self.fluxes['v_Amd1'].value + self.fluxes['v_Ade1312'].value
        #self.fluxes['v_HOR2'].value + self.fluxes['v_RHR2'].value 
        PI = - self.fluxes['v_GAPDH'].value + self.fluxes['v_ATPase'].value + self.fluxes['v_HOR2'].value+0.000 + 2*\
           self.fluxes['v_TPS1'].value + self.fluxes['v_TPS2'].value - self.fluxes['v_ATPmito'].value + self.fluxes['v_Isn1'].value \
        - self.fluxes['v_Pnp1'].value + self.fluxes['v_vacPi'].value - self.fluxes['vsinkG6P'].value -self.fluxes['vsinkF6P'].value - \
            self.fluxes['vsinkGAP'].value - self.fluxes['vsinkP3G'].value - self.fluxes['vsinkPEP'].value
        IMP = + self.fluxes['v_Amd1'].value - self.fluxes['v_Ade1312'].value + self.fluxes['v_Hpt1'].value - self.fluxes['v_Isn1'].value
        INO = + self.fluxes['v_Isn1'].value - self.fluxes['v_Pnp1'].value 
        HYP = + self.fluxes['v_Pnp1'].value  - self.fluxes['v_Hpt1'].value 
        NAD = + self.fluxes['v_G3PDH'].value  - self.fluxes['v_GAPDH'].value  + self.fluxes['v_ADH'].value  + self.fluxes['v_NADHmito'].value 
        NADH = - self.fluxes['v_G3PDH'].value + self.fluxes['v_GAPDH'].value  - self.fluxes['v_ADH'].value  - self.fluxes['v_NADHmito'].value 
        
        conc_out = torch.Tensor([GLCo, TREHo, GLCi, TRE, T6P, UDPG, G1P, G6P, F6P, F16BP, GAP, DHAP, P3G, P2G, PEP, PYR, G3P, ATP, ADP, AMP, PI, NAD, NADH, ETOH, ACE, BPG, IMP, INO, HYP, GLYCEROL])

        
        return conc_out

    

                

