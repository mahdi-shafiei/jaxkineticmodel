import jax.numpy as jnp

class SimpleActivator:
    "activation class modifier"
    def __init__(self, k_A: str):
        self.k_A = k_A

    def add_modifier(self, activator, eval_dict):
        k_A = eval_dict[self.k_A]
        return 1 + activator / k_A
    

class SimpleInhibitor:
    """inhibition class modifier"""
    def __init__(self,
                 k_I:str):
        super(SimpleInhibitor, self).__init__()
        self.k_I=k_I

    def add_modifier(self, inhibitor,eval_dict):
        k_I=eval_dict[self.k_I]
        return 1/ (1+ inhibitor/k_I) 



class Jax_Irrev_MM_Bi_w_Modifiers:
    """Michaelis-Menten equation with modifiers: can be activator or inhibition
    the names of the modifiers concentrations should be added as strings as well as classes, in the same order as the classes"""
    def __init__(self,substrate1:str,
                 substrate2:str,
                 modifiers_list:list, 
                 vmax: str, 
                 km_substrate1: str, 
                 km_substrate2: str,
                   modifiers):
        
        
        self.substrate1=substrate1
        self.substrate2=substrate2
        self.vmax = vmax
        self.km_substrate1 = km_substrate1
        self.km_substrate2 = km_substrate2
        self.modifiers = modifiers
        self.modifiers_list=modifiers_list

    def __call__(self, eval_dict):
        vmax = eval_dict[self.vmax]
        km_substrate1 = eval_dict[self.km_substrate1]
        km_substrate2 = eval_dict[self.km_substrate2]

        substrate1 = eval_dict[self.substrate1]
        substrate2 = eval_dict[self.substrate2]
        
        v = vmax * (substrate1 / km_substrate1) * (substrate2 / km_substrate2) \
            / ((1 + (substrate1 / km_substrate1)) * (1 + (substrate2 / km_substrate2)))

        for i, modifier in enumerate(self.modifiers):
            modifier_conc = eval_dict[self.modifiers_list[i]]
            v *= modifier.add_modifier(modifier_conc, eval_dict)

        return v
    

class Jax_Specific:
    """Specifically designed for PFK (for which the functional expression we retrieved from:
    Metabolic Engineering 77 (2023) 128–142
    Available online 23 March 2023
    1096-7176/© 2023 The Authors. Published by Elsevier Inc. on behalf of International Metabolic Engineering Society. This is an open access article under the CC
    BY license (http://creativecommons.org/licenses/by/4.0/).Elucidating yeast glycolytic dynamics at steady state growth and glucose
    pulses through kinetic metabolic modeling
    
    Think of reducing the equation by assuming AMP, ATP, ADP are constant
    """
    def __init__(self, 
                 substrate1:str,
                 substrate2:str,
                 product1:str,
                 modifier:str,
                 vmax: str, kr_F6P: str, kr_ATP: str, gr: str, c_ATP: str, ci_ATP: str, 
                 ci_AMP: str, ci_F26BP: str, ci_F16BP: str, l: str, kATP: str, kAMP: str, 
                 F26BP: str, kF26BP: str, kF16BP: str):
        self.vmax = vmax
        self.kr_F6P = kr_F6P
        self.kr_ATP = kr_ATP
        self.gr = gr
        self.c_ATP = c_ATP
        self.ci_ATP = ci_ATP
        self.ci_AMP = ci_AMP
        self.ci_F26BP = ci_F26BP
        self.ci_F16BP = ci_F16BP
        self.l = l
        self.kATP = kATP
        self.kAMP = kAMP
        self.F26BP = F26BP
        self.kF26BP = kF26BP
        self.kF16BP = kF16BP
        self.substrate1=substrate1
        self.substrate2=substrate2
        self.product1=product1
        self.modifier=modifier

    def __call__(self, eval_dict):
        vmax = eval_dict[self.vmax]
        kr_F6P = eval_dict[self.kr_F6P]
        kr_ATP = eval_dict[self.kr_ATP]
        gr = eval_dict[self.gr]
        c_ATP = eval_dict[self.c_ATP]
        ci_ATP = eval_dict[self.ci_ATP]
        ci_AMP = eval_dict[self.ci_AMP]
        ci_F26BP = eval_dict[self.ci_F26BP]
        ci_F16BP = eval_dict[self.ci_F16BP]
        l = eval_dict[self.l]
        kATP = eval_dict[self.kATP]
        kAMP = eval_dict[self.kAMP]
        F26BP = eval_dict[self.F26BP]
        kF26BP = eval_dict[self.kF26BP]
        kF16BP = eval_dict[self.kF16BP]

        # substrate and product are assumed to be provided in eval_dict
        substrate1 = eval_dict[self.substrate1]
        substrate2 = eval_dict[self.substrate2]
        product = eval_dict[self.product1]
        modifiers = eval_dict[self.modifier]

        lambda1 = substrate1 / kr_F6P
        lambda2 = substrate2 / kr_ATP
        R = 1 + lambda1 * lambda2 + gr * lambda1 * lambda2
        T = 1 + c_ATP * lambda2
        L = l * ((1 + ci_ATP * substrate2 / kATP) / (1 + substrate2 / kATP)) \
              * ((1 + ci_AMP * modifiers / kAMP) / (modifiers / kAMP)) \
              * ((1 + ci_F26BP * F26BP / kF26BP + ci_F16BP * product / kF16BP) \
                 / (1 + F26BP / kF26BP + product / kF16BP))
        
        return vmax * gr * lambda1 * lambda2 * R / (R**2 + L * T**2)





