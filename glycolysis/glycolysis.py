import torch
from torch import nn

class Glycolysis(torch.nn.Module):
    def __init__(self,
                 fluxes,
                 metabolites):
        super(Glycolysis, self).__init__()
        self.fluxes = nn.ParameterDict(fluxes) # dict with fluxes
        self.metabolites = metabolites # dict with indices of each metabolite
        
       
    #GLYCERAL3P = GLYC3P = G3P ???
    
    def calculate_fluxes(self, concentrations):
        """Calculate fluxes and assign to value within flux class"""
        self.fluxes['v_GLT'].value=self.fluxes['v_GLT'].calculate(concentrations[self.metabolites['Glci']],concentrations[self.metabolites['Glco']]) 
        
        self.fluxes['v_GLK'].value=self.fluxes['v_GLK'].calculate([concentrations[self.metabolites['ATP']], concentrations[self.metabolites['Glci']]],
                                       [concentrations[self.metabolites['ADP']], concentrations[self.metabolites['G6P']]],
                                       concentrations[self.metabolites['T6P']])
                                       
        self.fluxes['v_PGM1'].value=self.fluxes['v_PGM1'].calculate(concentrations[self.metabolites['G1P']], concentrations[self.metabolites['G6P']])
        
        self.fluxes['v_TPS1'].value=self.fluxes['v_TPS1'].calculate([concentrations[self.metabolites['G6P']],0.07], #concentrations[self.metabolites['UDPG']]
                                        [concentrations[self.metabolites['F6P']], concentrations[self.metabolites['PI']]])
        self.fluxes['v_TPS2'].value=self.fluxes['v_TPS2'].calculate(concentrations[self.metabolites['T6P']], concentrations[self.metabolites['PI']])
        
        self.fluxes['v_NTH1'].value=self.fluxes['v_NTH1'].calculate(concentrations[self.metabolites['TRE']])

        self.fluxes['v_PGI'].value=self.fluxes['v_PGI'].calculate(concentrations[self.metabolites['F6P']], concentrations[self.metabolites['ATP']])
        
        self.fluxes['v_PFK'].value=self.fluxes['v_PFK'].calculate([concentrations[self.metabolites['ATP']], concentrations[self.metabolites['Glci']]],
                                       concentrations[self.metabolites['F16BP']], 
                                       concentrations[self.metabolites['AMP']])
        
        self.fluxes['v_ALD'].value=self.fluxes['v_ALD'].calculate(concentrations[self.metabolites['F16BP']],
                                       [concentrations[self.metabolites['F6P']], concentrations[self.metabolites['DHAP']]])
        
        self.fluxes['v_TPI'].value=self.fluxes['v_TPI'].calculate(concentrations[self.metabolites['G3P']], concentrations[self.metabolites['G3P']])
        
        self.fluxes['v_G3PDH'].value=self.fluxes['v_G3PDH'].calculate([concentrations[self.metabolites['DHAP']], concentrations[self.metabolites['NAD']]],
                                         [concentrations[self.metabolites['G3P']], concentrations[self.metabolites['G3P']]],
                                         [concentrations[self.metabolites['F16BP']], concentrations[self.metabolites['ATP']], concentrations[self.metabolites['ADP']]])
        
        self.fluxes['v_HOR2'].value=self.fluxes['v_HOR2'].calculate(concentrations[self.metabolites['G3P']], concentrations[self.metabolites['PI']])
        
        self.fluxes['v_GlycT'].value=self.fluxes['v_GlycT'].calculate(concentrations[self.metabolites['GLYCEROL']])
        
        self.fluxes['v_GAPDH'].value=self.fluxes['v_GAPDH'].calculate([concentrations[self.metabolites['G3P']], concentrations[self.metabolites['NAD']], 
                                          concentrations[self.metabolites['PI']]],
                                       [concentrations[self.metabolites['BPG']], concentrations[self.metabolites['NADH']]])
        
        self.fluxes['v_PGK'].value=self.fluxes['v_PGK'].calculate([concentrations[self.metabolites['BPG']], concentrations[self.metabolites['ADP']]],
                                       [concentrations[self.metabolites['P3G']], concentrations[self.metabolites['ATP']]])
        
        self.fluxes['v_PGM'].value=self.fluxes['v_PGM'].calculate(concentrations[self.metabolites['P3G']], concentrations[self.metabolites['P2G']])
        
        self.fluxes['v_ENO'].value=self.fluxes['v_ENO'].calculate(concentrations[self.metabolites['P2G']], concentrations[self.metabolites['PEP']])
        
        self.fluxes['v_PYK'].value =self.fluxes['v_PYK'].calculate([concentrations[self.metabolites['PEP']], concentrations[self.metabolites['ADP']]],
                                      concentrations[self.metabolites['ATP']], 
                                      concentrations[self.metabolites['F16BP']])
        
        self.fluxes['v_PDC'].value=self.fluxes['v_PDC'].calculate(concentrations[self.metabolites['PYR']], concentrations[self.metabolites['PI']])
        
        self.fluxes['v_ADH'].value=self.fluxes['v_ADH'].calculate([concentrations[self.metabolites['NAD']], concentrations[self.metabolites['ETOH']]],
                                       [concentrations[self.metabolites['ACE']], concentrations[self.metabolites['NADH']]])
        
        self.fluxes['v_EtohT'].value=self.fluxes['v_EtohT'].calculate(concentrations[self.metabolites['ETOH']])
        
        self.fluxes['v_ATPmito'].value=self.fluxes['v_ATPmito'].calculate([concentrations[self.metabolites['PI']],  concentrations[self.metabolites['ADP']]])
        
        self.fluxes['v_ATPase'].value=self.fluxes['v_ATPase'].calculate(concentrations[self.metabolites['ATP']],  concentrations[self.metabolites['ADP']])
        
        self.fluxes['v_ADK1'].value=self.fluxes['v_ADK1'].calculate(concentrations[self.metabolites['ADP']], [concentrations[self.metabolites['ATP']],
                                                                                  concentrations[self.metabolites['AMP']]])
        
        self.fluxes['v_vacPi'].value=self.fluxes['v_vacPi'].calculate(concentrations[self.metabolites['PI']])
        
        self.fluxes['v_Amd1'].value=self.fluxes['v_Amd1'].calculate(concentrations[self.metabolites['AMP']], concentrations[self.metabolites['ATP']],
                                        concentrations[self.metabolites['PI']])
        
        self.fluxes['v_Ade1312'].value=self.fluxes['v_Ade1312'].calculate(concentrations[self.metabolites['IMP']])
        
        self.fluxes['v_Isn1'].value=self.fluxes['v_Isn1'].calculate(concentrations[self.metabolites['IMP']])
        
        self.fluxes['v_Pnp1'].value=self.fluxes['v_Pnp1'].calculate(concentrations[self.metabolites['INO']])
        
        self.fluxes['v_Hpt1'].value=self.fluxes['v_Hpt1'].calculate(concentrations[self.metabolites['HYP']])
        
        self.fluxes['v_NADHmito'].value=self.fluxes['v_NADHmito'].value=self.fluxes['v_NADHmito'].calculate(concentrations[self.metabolites['NADH']])
        
        self.fluxes['vsinkG6P'].value=self.fluxes['vsinkG6P'].calculate(concentrations[self.metabolites['G6P']])
        
        self.fluxes['vsinkF6P'].value=self.fluxes['vsinkF6P'].calculate(concentrations[self.metabolites['F6P']])
        
        self.fluxes['vsinkGAP'].value=self.fluxes['vsinkGAP'].calculate(concentrations[self.metabolites['GAP']])
        
        self.fluxes['vsinkP3G'].value=self.fluxes['vsinkP3G'].calculate(concentrations[self.metabolites['P3G']])
        
        self.fluxes['vsinkPEP'].value=self.fluxes['vsinkPEP'].calculate(concentrations[self.metabolites['PEP']])
        
        self.fluxes['vsinkPYR'].value=self.fluxes['vsinkPYR'].calculate(concentrations[self.metabolites['PYR']])
        self.fluxes['vsinkACE'].value=self.fluxes['vsinkACE'].calculate(concentrations[self.metabolites['ACE']])
   


        


    def forward(self, _ , conc_in):
        self.calculate_fluxes(conc_in)

        GLCi = -self.fluxes['v_GLK'].value + self.fluxes['v_GLT'].value + 2.*self.fluxes['v_NTH1'].value
        G6P = + self.fluxes['v_GLK'].value - self.fluxes['v_PGI'].value + self.fluxes['vsinkG6P'].value + self.fluxes['v_PGM1'].value - self.fluxes['v_TPS1'].value
        G1P = - self.fluxes['v_PGM1'].value - 3.1000e-04#self.fluxes['v_UGP'].value
        T6P = + self.fluxes['v_TPS1'].value -self.fluxes['v_TPS2'].value
        TRE = + self.fluxes['v_TPS2'].value - self.fluxes['v_NTH1'].value
        F6P = - self.fluxes['v_PFK'].value + self.fluxes['v_PGI'].value + self.fluxes['vsinkF6P'].value
        F16BP = + self.fluxes['v_PFK'].value - self.fluxes['v_ALD'].value
        GAP = + self.fluxes['v_ALD'].value - self.fluxes['v_GAPDH'].value + self.fluxes['v_TPI'].value + self.fluxes['vsinkGAP'].value
        DHAP = + self.fluxes['v_ALD'].value - self.fluxes['v_TPI'].value - self.fluxes['v_G3PDH'].value
        G3P = + self.fluxes['v_G3PDH'].value - self.fluxes['v_HOR2'].value
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
        
        return torch.Tensor([GLCi, G6P, G1P,T6P, TRE, F6P, F16BP, GAP, DHAP, G3P, GLYCEROL, BPG, P3G, P2G, PEP, PYR, ACE, ETOH, ATP, ADP, AMP, PI, IMP, INO, HYP, NAD, NADH, GLCo])

    



# def forward(self, _, conc_in):
#        # !!!!!!!fluxes.calculate_fluxes()!!!!!!



#         GLCi = -self.v_HXK + self.v_GLT + 2.*self.v_NTH1
#         G6P = + self.v_GLK - self.v_PGI + self.v_sinkG6P + self.v_PGM1 - self.v_TPS1
#         G1P = - self.v_PGM1 - self.v_UGP
#         T6P = + self.v_TPS1 - self.v_TPS2
#         TRE = + self.v_TPS2 - self.v_NTH1
#         F6P = - self.v_PFK + self.v_PGI + self.v_sinkF6P
#         F16BP = + self.v_PFK - self.v_ALD
#         GAP = + self.v_ALD - self.v_GAPDH + self.v_TPI1 + self.v_sinkGAP
#         DHAP = + self.v_ALD - self.v_TPI1 - self.v_G3PDH
#         G3P = + self.v_G3PDH - self.v_HOR2
#         GLYCEROL = + self.v_HOR2 - self.v_GLYCEROLtransport
#         BPG = + self.v_GAPDH - self.v_PGK
#         P3G = + self.v_PGK - self.v_PGM + self.v_sinkP3G
#         P2G = + self.v_PGM - self.v_ENO
#         PEP = + self.v_ENO - self.v_PYK + self.v_sinkPEP
#         PYR = + self.v_PYK - self.v_PDC + self.v_sinkPYR
#         ACE = + self.v_PDC - self.v_ADH + self.v_sinkACE
#         ETOH = + self.v_ADH - self.v_ETOHtransport
#         ATP = + self.v_ADK1 - self.v_GLK - self.v_ATPase - self.v_PFK + \
#             self.v_PGK + self.v_PYK - self.v_TPS1 + self.v_mito
#         ADP = - 2.*self.v_ADK1 + self.v_GLK + self.v_ATPase + \
#             self.v_PFK - self.v_PGK - self.v_PYK + self.v_TPS1 - self.v_mito
#         AMP = + self.v_ADK1 - self.v_Amd1 + self.v_Ade13_v_Ade12
#         PI = - self.v_GAPDH + self.v_ATPase + self.v_HOR2 + self.v_RHR2 + 2 * \
#             self.v_TPS1 + self.v_TPS2 - self.v_mito + self.v_Isn1 - self.v_Pnp1
#         + self.v_vacuolePi - self.v_sinkG6P - self.v_sinkF6P - \
#             self.v_sinkGAP - self.v_sinkP3G - self.v_sinkPEP
#         IMP = + self.v_Amd1 - self.v_Ade13_v_Ade12 + self.v_Hpt1 - self.self.v_Isn1
#         INO = + self.v_Isn1 - self.v_Pnp1
#         HYP = + self.v_Pnp1 - self.v_Hpt1
#         NAD = + self.v_G3PDH - self.v_GAPDH + self.v_ADH + self.v_mitoNADH
#         NADH = - self.v_G3PDH + self.v_GAPDH - self.v_ADH - self.v_mitoNADH
#         return [GLCi, G6P, G1P,	T6P, TRE, F6P, F16BP, GAP, DHAP, G3P, GLYCEROL,	BPG, P3G, P2G, PEP,	PYR, ACE, ETOH,	ATP, ADP, AMP, PI, IMP, INO, HYP, NAD, NADH]

