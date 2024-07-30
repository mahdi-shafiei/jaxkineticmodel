import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim



class Torch_ATPase(torch.nn.Module):
    def __init__(self,
                ATPase_ratio:float,
                to_be_learned):
            super(Torch_ATPase, self).__init__()
            if to_be_learned[0]:
                self.ATPase_ratio = torch.nn.Parameter(torch.Tensor([ATPase_ratio]))
            else:
                self.ATPase_ratio = ATPase_ratio
            

    def calculate(self, substrate, product):
        return self.ATPase_ratio * substrate / product
    

#v_GLT
class Torch_Facilitated_Diffusion(torch.nn.Module):
    def __init__(self,
                vmax: float,
                k_equilibrium: float,
                km_internal: float,
                km_external: float,
                to_be_learned):
            super(Torch_Facilitated_Diffusion, self).__init__()

            if to_be_learned[0]:
                self.vmax = torch.nn.Parameter(torch.tensor(vmax))
            else:
                self.vmax = vmax


            if to_be_learned[2]:
                self.k_e = torch.nn.Parameter(torch.Tensor([k_equilibrium]))
            else:
                self.km_internal = km_internal

            if to_be_learned[2]:
                self.km_internal = torch.nn.Parameter(torch.Tensor([km_internal]))
            else:
                self.km_internal = km_internal
            
            if to_be_learned[3]:
                self.km_external = torch.nn.Parameter(torch.Tensor([km_external]))
            else:
                self.km_external = km_external
            

    def calculate(self, substrate_i, substrate_e):
        numerator = self.vmax *(substrate_e - substrate_i)/self.km_external
        denominator = self.km_external * (1+ substrate_e/self.km_external + substrate_i/self.km_internal + 0.91 * substrate_e* substrate_i/self.km_external /self.km_internal)
        return numerator/denominator


class Torch_Amd1(torch.nn.Module):
    def __init__(self,
                vmax:float,
                k50:float,
                ki:float,
                k_atp:float,
                to_be_learned):
            super(Torch_Amd1, self).__init__()
            
            if to_be_learned[0]:
                self.vmax = torch.nn.Parameter(torch.Tensor([vmax]))
            else:
                self.vmax = vmax
            
            if to_be_learned[1]:
                self.k50 = torch.nn.Parameter(torch.Tensor([k50]))
            else:
                self.k50  = k50 

            if to_be_learned[2]:
                self.ki = torch.nn.Parameter(torch.Tensor([ki]))
            else:
                self.ki = ki

            if to_be_learned[3]:
                self.k_atp = torch.nn.Parameter(torch.Tensor([k_atp]))
            else:
                self.k_atp = k_atp

    # s AMP # p ATP # m PI
    def calculate(self, substrate, product, modifier):
        #(p_Amd1_Vmax * AMP) / (p_Amd1_K50 * (1+ PHOS / p_Amd1_Kpi) / (ATP / p_Amd1_Katp + 1) + AMP);
        return self.vmax * substrate / (self.k50 * (1 + modifier/self.ki) / (product/self.k_atp) + substrate)



class Torch_MA_Rev(torch.nn.Module):
    def __init__(self,
                k:float,
                steady_state_substrate:float,
                to_be_learned):
            super(Torch_MA_Rev, self).__init__()
            if to_be_learned[0]:
                self.k = torch.nn.Parameter(torch.Tensor([k]))
            else:
                self.k = k
            self.steady_state_substrate = steady_state_substrate
            

    def calculate(self, substrate):
        return self.k * (self.steady_state_substrate - substrate)




class Torch_MA_Rev_Bi(torch.nn.Module):
    def __init__(self,
                k_equilibrium:float,
                k_fwd:float,
                to_be_learned):
            super(Torch_MA_Rev_Bi, self).__init__()


            if to_be_learned[0]:
                self.k_equilibrium = torch.nn.Parameter(torch.Tensor([k_equilibrium]))
            else:
                self.k_equilibrium = k_equilibrium
            
            if to_be_learned[1]:
                self.k_fwd = torch.nn.Parameter(torch.Tensor([k_fwd]))
            else:
                self.k_fwd = k_fwd

          

    def calculate(self, substrate, product):
        return self.k_fwd * (substrate*substrate - product[0]*product[1]/self.k_equilibrium)


# v_ADH
class Torch_MM_Ordered_Bi_Bi(torch.nn.Module):
    #co-factor binding first
    def __init__(self,
                vmax:float,
                k_equilibrium:float,
                km_substrate1:float,
                km_substrate2:float,
                km_product1:float,
                km_product2:float,
                ki_substrate1:float,
                ki_substrate2:float,
                ki_product1:float,
                ki_product2:float,
                to_be_learned):
            super(Torch_MM_Ordered_Bi_Bi, self).__init__()

            # dictionary with all parameters
            params = {
                vmax: 'vmax',
                k_equilibrium: 'k_equilibrium',
                km_substrate1: 'km_substrate1',
                km_substrate2: 'km_substrate2',
                km_product1: 'km_product1',
                km_product2: 'km_product2',
                ki_substrate1: 'ki_substrate1',
                ki_substrate2: 'ki_substrate2',
                ki_product1: 'ki_product1',
                ki_product2: 'ki_product2'
            }

            # make parameters learnable/treat as a given
            for idx, (value, param_name) in enumerate(params.items()):
                if to_be_learned[idx]:
                    self.__setattr__(
                        param_name, torch.nn.Parameter(torch.Tensor([value])))
                else:
                    self.__setattr__(param_name, value)
          
    #vADH: ACE + NADH -> ETOH + NAD;  
    

#NAD = s1
#ethanol = p1
    def calculate(self, substrate, product):
        s1 = substrate[0] #NAD
        s2 = substrate[1] #ETOH
        p1 = product[0] #ACE
        p2 = product[1] #NADH
        
        numerator = self.vmax * (s1* s2 - p1*p2/self.k_equilibrium) / self.ki_substrate1/ self.km_substrate2
        denominator = 1 + s1/self.ki_substrate1 
        + self.km_substrate1 * s2 / self.ki_substrate1 /  self.km_substrate2 
        + self.km_product2 * p1 / self.km_product1/ self.ki_product2 
        + p2/self.ki_product2 
        + s1 * s2 / self.ki_substrate1 /  self.km_substrate2   
        + self.km_product2 * s1 * p1 / self.km_product1/ self.ki_product2 / self.ki_substrate1
        + self.km_substrate1 * s2 * p2 / self.ki_substrate1/ self.km_substrate2 / self.ki_product2 
        + p1 * p2 / self.km_product1/ self.ki_product2 
        + s1 * s2 * p1 / self.ki_substrate1/ self.km_substrate2 / self.ki_product1 
        + s2 * p1 * p2 / self.ki_substrate1/ self.km_substrate2 / self.ki_product2 
        return numerator/denominator




# v_GAPDH
class Torch_MM_Ordered_Bi_Tri(torch.nn.Module):
    #co-factor binding first
    def __init__(self,
                vmax:float,
                k_equilibrium:float,
                km_substrate1:float,
                km_substrate2:float,
                ki:float,
                km_product1:float,
                km_product2:float,
                to_be_learned):
            super(Torch_MM_Ordered_Bi_Tri, self).__init__()

            # dictionary with all parameters
            params = {
                vmax: 'vmax',
                k_equilibrium: 'k_equilibrium',
                km_substrate1: 'km_substrate1',
                km_substrate2: 'km_substrate2',
                ki: 'ki',
                km_product1: 'km_product1',
                km_product2: 'km_product2',
            }

            # make parameters learnable/treat as a given
            for idx, (value, param_name) in enumerate(params.items()):
                if to_be_learned[idx]:
                    self.__setattr__(
                        param_name, torch.nn.Parameter(torch.Tensor([value])))
                else:
                    self.__setattr__(param_name, value)
          

    def calculate(self, substrate, product):
        s1 = substrate[0] #glyc3p
        s2 = substrate[1] #nad
        s3 = substrate[2] #pi
        p1 = product[0] #bpg
        p2 = product[1] #nadh
        
        numerator = self.vmax * (s1* s2 *s3  - p1*p2/self.k_equilibrium) / self.km_substrate1 / self.km_substrate2 /self.ki
        denominator = (1 + s1/self.km_substrate1) * (1 + s2/self.km_substrate2)*\
        (1 + s3/self.ki) + (1 + p1/self.km_product1) * (p2/self.km_product2) - 1
        return numerator/denominator


# v_PDC
class Torch_Hill_Irreversible_Inhibition(torch.nn.Module):
    def __init__(self,
                 vmax: float,
                 hill: float,
                 k_half_substrate: float,
                 ki:float,
                 to_be_learned):
        super(Torch_Hill_Irreversible_Inhibition, self).__init__()

        # self.substrate=substrate
        if to_be_learned[0]:
            self.vmax = torch.nn.Parameter(torch.Tensor([vmax]))
        else:
            self.vmax = vmax

        if to_be_learned[1]:
            self.hill = torch.nn.Parameter(torch.Tensor([hill]))
        else:
            self.hill = hill

        if to_be_learned[2]:
            self.k_half_substrate = torch.nn.Parameter(
                torch.Tensor([k_half_substrate]))
        else:
            self.k_half_substrate = k_half_substrate

        if to_be_learned[3]:
            self.ki = torch.nn.Parameter(
                torch.Tensor([ki]))
        else:
            self.ki = ki



    def calculate(self, substrate, inhibitor):
        numerator = self.vmax * ((substrate/self.k_half_substrate) ** self.hill)
        denominator = 1 + ((substrate/self.k_half_substrate) ** self.hill) + inhibitor/self.ki
        return numerator/denominator


# v_PYK
class Torch_Hill_Bi_Irreversible_Activation(torch.nn.Module):
    def __init__(self,
                 vmax: float,
                 hill: float,
                 k_substrate1: float,
                 k_substrate2: float,
                 k_product: float,
                 k_activator:float,
                 l:float,
                 to_be_learned):
        super(Torch_Hill_Bi_Irreversible_Activation, self).__init__()

        # self.substrate=substrate
        if to_be_learned[0]:
            self.vmax = torch.nn.Parameter(torch.Tensor([vmax]))
        else:
            self.vmax = vmax

        if to_be_learned[1]:
            self.hill = torch.nn.Parameter(torch.Tensor([hill]))
        else:
            self.hill = hill

        if to_be_learned[2]:
            self.k_substrate1 = torch.nn.Parameter(
                torch.Tensor([k_substrate1]))
        else:
            self.k_substrate1 = k_substrate1

        if to_be_learned[3]:
            self.k_substrate2 = torch.nn.Parameter(
                torch.Tensor([k_substrate2]))
        else:
            self.k_substrate2 = k_substrate2

        if to_be_learned[4]:
            self.l = torch.nn.Parameter(
                torch.Tensor([l]))
        else:
            self.l = l


        if to_be_learned[5]:
            self.k_product = torch.nn.Parameter(
                torch.Tensor([k_product]))
        else:
            self.k_product = k_product


        if to_be_learned[6]:
            self.k_activator = torch.nn.Parameter(
                torch.Tensor([k_activator]))
        else:
            self.k_activator = k_activator

    #pep = substarte1 
    #adp = substrate2
    #atp =product
    #FBP=activator
    # vPYK:  PEP + ADP -> PYR + ATP;   
    def calculate(self, substrate, product, activator):
        return self.vmax * substrate[0] * substrate[1] / (self.k_substrate1 * self.k_substrate2) /((1+substrate[0] / self.k_substrate1) * (1+substrate[1] / self.k_substrate2)) \
            * ((substrate[0] / self.k_substrate1 + 1) ** (self.hill - 1)) \
                / (self.l * (((product/self.k_product + 1)/ (activator/self.k_activator + 1))**self.hill) + ((substrate[0] / self.k_substrate1 + 1) ** self.hill))



##includes simple (non-competitive) inhibition/activation
class Torch_Irrev_MM_Bi_w_Modifiers(torch.nn.Module):
    def __init__(self,
                 vmax: float,
                 km_substrate1: float,
                 km_substrate2: float,
                 modifiers,
                 to_be_learned):
        super(Torch_Irrev_MM_Bi_w_Modifiers, self).__init__()
        # self.substrate1=substrate1
        # self.substrate2=substrate2
        self.modifiers = modifiers
        if to_be_learned[0]:
            self.vmax = torch.nn.Parameter(torch.Tensor([vmax]))
        else:
            self.vmax = vmax

        if to_be_learned[1]:
            self.km_substrate1 = torch.nn.Parameter(
                torch.Tensor([km_substrate1]))
        else:
            self.km_substrate1 = km_substrate1

        if to_be_learned[2]:
            self.km_substrate2 = torch.nn.Parameter(
                torch.Tensor([km_substrate2]))
        else:
            self.km_substrate2 = km_substrate2
        

    def calculate(self, substrate, modifier_conc):
        substrate1 = substrate[0]
        substrate2 = substrate[1]
        v = self.vmax*(substrate1/self.km_substrate1) * \
            (substrate2/self.km_substrate2) \
                  / (1+(substrate1/self.km_substrate1)) * \
            (1+(substrate2/self.km_substrate2))
        
        for i, modifier in enumerate(self.modifiers):
            v *= modifier.add_modifier(modifier_conc[i])
        return v
    

# v_TPS2
class Torch_Irrev_MM_Bi_w_Inhibition(torch.nn.Module):
    def __init__(self,
                 vmax: float,
                 km_substrate1: float,
                 ki: float,
                 to_be_learned):
        super(Torch_Irrev_MM_Bi_w_Inhibition, self).__init__()
     
        if to_be_learned[0]:
            self.vmax = torch.nn.Parameter(torch.Tensor([vmax]))
        else:
            self.vmax = vmax

        if to_be_learned[1]:
            self.km_substrate1 = torch.nn.Parameter(
                torch.Tensor([km_substrate1]))
        else:
            self.km_substrate1 = km_substrate1

        if to_be_learned[2]:
            self.ki = torch.nn.Parameter(
                torch.Tensor([ki]))
        else:
            self.ki = ki
        
    def calculate(self, substrate, product):
    
        return (self.vmax * substrate * product) /((self.km_substrate1 * self.ki) + (self.km_substrate1 * substrate)*product)
    

class Torch_Irrev_MM_Uni_w_Modifiers(torch.nn.Module):
    def __init__(self,
                 vmax: float,
                 km_substrate: float,
                 modifiers,
                 to_be_learned):
        super(Torch_Irrev_MM_Uni_w_Modifiers, self).__init__()
        self.modifiers = modifiers
        if to_be_learned[0]:
            # make mu a learnable parameter
            self.vmax = torch.nn.Parameter(torch.tensor([vmax]))
        else:
            self.vmax = vmax

        if to_be_learned[1]:
            self.km_substrate = torch.nn.Parameter(torch.Tensor([km_substrate]))
        else:
            self.km_substrate = km_substrate

    def calculate(self, substrate, modifier_conc):
        v = (self.vmax)*(substrate/self.km_substrate)/(1+(substrate/self.km_substrate))
       
        for i, modifier in enumerate(self.modifiers):
            v *= modifier.add_modifier(modifier_conc) 
        return v


     
# v_PFK    
class Torch_Specific(torch.nn.Module):
    def __init__(self,
                vmax: float,
                kr_F6P:float,
                kr_ATP:float,
                gr:float,
                c_ATP:float,
                ci_ATP:float,
                ci_AMP:float,
                ci_F26BP:float,
                ci_F16BP:float,
                l:float,
                kATP:float,
                kAMP:float,
                F26BP:float,
                kF26BP:float,
                kF16BP:float,
                to_be_learned):
            super(Torch_Specific, self).__init__()

            if to_be_learned[0]:
                self.vmax = torch.nn.Parameter(torch.tensor(vmax))
            else:
                self.vmax = vmax
            
            if to_be_learned[1]:
                self.kr_F6P = torch.nn.Parameter(torch.Tensor([kr_F6P]))
            else:
                self.kr_F6P = kr_F6P

            if to_be_learned[2]:
                self.kr_ATP = torch.nn.Parameter(torch.Tensor([kr_ATP]))
            else:
                self.kr_ATP = kr_ATP

            if to_be_learned[3]:
                self.gr = torch.nn.Parameter(torch.Tensor([gr]))
            else:
                self.gr = gr

            if to_be_learned[4]:
                self.c_ATP = torch.nn.Parameter(torch.Tensor([c_ATP]))
            else:
                self.c_ATP  = c_ATP 

            if to_be_learned[5]:
                self.ci_ATP = torch.nn.Parameter(torch.Tensor([ci_ATP]))
            else:
                self.ci_ATP  = ci_ATP 

            if to_be_learned[6]:
                self.ci_AMP = torch.nn.Parameter(torch.Tensor([ci_AMP]))
            else:
                self.ci_AMP  = ci_AMP 

            if to_be_learned[7]:
                self.ci_F26BP = torch.nn.Parameter(torch.Tensor([ci_F26BP]))
            else:
                self.ci_F26BP  = ci_F26BP 
            
            if to_be_learned[8]:
                self.ci_F16BP = torch.nn.Parameter(torch.Tensor([ci_F16BP]))
            else:
                self.ci_F16BP  = ci_F16BP 

            if to_be_learned[9]:
                self.l = torch.nn.Parameter(torch.Tensor([l]))
            else:
                self.l  = l

            if to_be_learned[10]:
                self.kATP = torch.nn.Parameter(torch.Tensor([kATP]))
            else:
                self.kATP  = kATP


            if to_be_learned[11]:
                self.kAMP = torch.nn.Parameter(torch.Tensor([kAMP]))
            else:
                self.kAMP  = kAMP


            if to_be_learned[12]:
                self.F26BP = torch.nn.Parameter(torch.Tensor([F26BP]))
            else:
                self.F26BP  = F26BP

            if to_be_learned[13]:
                self.kF26BP = torch.nn.Parameter(torch.Tensor([kF26BP]))
            else:
                self.kF26BP  = kF26BP
            if to_be_learned[14]:
                self.kF16BP = torch.nn.Parameter(torch.Tensor([kF16BP]))
            else:
                self.kF16BP  = kF16BP
    
    # vPFK: F6P + ATP -> F16BP + ADP      
    # modifiers amp
    # product f16bp
  
    def calculate(self, substrate, product, modifiers):

        lambda1 = substrate[0] / self.kr_F6P
        lambda2 = substrate[1] / self.kr_ATP 
        R = 1 + lambda1 * lambda2 + self.gr *  lambda1 * lambda2
        T = 1 + self.c_ATP * lambda2
        L = self.l * ((1+ self.ci_ATP * substrate[1]/self.kATP)/(1+  substrate[1]/self.kATP)) \
            * ((1+ self.ci_AMP * modifiers/self.kAMP)/( modifiers/self.kAMP)) \
            * ((1+ self.ci_F26BP * self.F26BP / self.kF26BP + self.ci_F16BP * product/self.kF16BP) \
               /(1+ self.F26BP/self.kF26BP + product/self.kF16BP))
        
        return self.vmax * self.gr * lambda1 * lambda2 * R /(R**2 +L*T**2)
    

# v_ALD
class Torch_MM_unibi(torch.nn.Module):
    def __init__(self,
                 vmax: float,
                 k_equilibrium: float,
                 km_substrate: float,
                 km_product1: float,
                 km_product2: float,
                 to_be_learned):
        super(Torch_MM_unibi, self).__init__()

        # self.substrate=substrate
        if to_be_learned[0]:
            self.vmax = torch.nn.Parameter(torch.Tensor([vmax]))
        else:
            self.vmax = vmax

        if to_be_learned[1]:
            self.k_equilibrium = torch.nn.Parameter(torch.Tensor([k_equilibrium]))
        else:
            self.k_equilibrium = k_equilibrium

        if to_be_learned[2]:
            self.km_substrate = torch.nn.Parameter(
                torch.Tensor([km_substrate]))
        else:
            self.k_half_substrate = km_substrate

        if to_be_learned[3]:
            self.km_product1 = torch.nn.Parameter(
                torch.Tensor([km_product1]))
        else:
            self.km_product1 = km_product1

        if to_be_learned[4]:
            self.km_product2 = torch.nn.Parameter(
                torch.Tensor([km_product2]))
        else:
            self.km_product2 = km_product2

    def calculate(self, substrate, product):
        numerator = self.vmax / self.km_substrate * (substrate - product[0] * product[1] / self.k_equilibrium)
        denominator = (substrate/self.km_substrate + (1 + product[0]/self.km_product1) * (1 + product[1]/self.km_product2))
        return numerator/denominator

#s1 ATP
#s2 GLCi
#p1 ADP
#p2 G6P
class Torch_Rev_BiBi_MM_w_Inhibition(torch.nn.Module):
    def __init__(self,
                 vmax: float,
                 k_equilibrium: float,
                 km_substrate1: float,
                 km_substrate2: float,
                 km_product1: float,
                 km_product2: float,
                 ki_inhibitor:float,
                 to_be_learned
                 ):
        super(Torch_Rev_BiBi_MM_w_Inhibition, self).__init__()
       

        # dictionary with all parameters
        params = {
            vmax: 'vmax',
            k_equilibrium: 'k_equilibrium',
            km_substrate1: 'km_substrate1',
            km_substrate2: 'km_substrate2',
            km_product1: 'km_product1',
            km_product2: 'km_product2',
            ki_inhibitor: 'ki_inhibitor'
        }

        # make parameters learnable/treat as a given
        for idx, (value, param_name) in enumerate(params.items()):
            if to_be_learned[idx]:
                self.__setattr__(
                    param_name, torch.nn.Parameter(torch.Tensor([value])))
            else:
                self.__setattr__(param_name, value)

    def calculate(self, substrate, product, modifier):

        denominator = (1 + substrate[0]/ self.km_substrate1 + product[0]/ self.km_product1)*\
            (1 + substrate[1]/ self.km_substrate2 + product[1]/ self.km_product2 + modifier/self.ki_inhibitor)
        
        numerator = self.vmax*(substrate[0]*substrate[1]/self.km_substrate1/self.km_substrate2)*(
            1-1/self.k_equilibrium*(product[0]*product[1]/substrate[0]/substrate[1]))
        v = numerator/denominator

        return v


# v_G3PDH
class Torch_Rev_BiBi_MM_w_Activation(torch.nn.Module):
    def __init__(self,
                 vmax: float,
                 k_equilibrium: float,
                 km_substrate1: float,
                 km_substrate2: float,
                 km_product1: float,
                 km_product2: float,
                 ka1: float,
                 ka2: float,
                 ka3: float,
                 to_be_learned
                 ):
        super(Torch_Rev_BiBi_MM_w_Activation, self).__init__()
       

        # dictionary with all parameters
        params = {
            vmax: 'vmax',
            k_equilibrium: 'k_equilibrium',
            km_substrate1: 'km_substrate1',
            km_substrate2: 'km_substrate2',
            km_product1: 'km_product1',
            km_product2: 'km_product2',
            ka1: 'ka1',
            ka2: 'ka2',
            ka3: 'ka3'
        }

        # make parameters learnable/treat as a given
        for idx, (value, param_name) in enumerate(params.items()):
            if to_be_learned[idx]:
                self.__setattr__(
                    param_name, torch.nn.Parameter(torch.Tensor([value])))
            else:
                self.__setattr__(param_name, value)

    def calculate(self, substrate, product, modifier):

        denominator = (1 + substrate[0]/ self.km_substrate1 + product[0]/ self.km_product1)*\
            (1 + substrate[1]/ self.km_substrate2 + product[1]/ self.km_product2)*\
            (1 + modifier[0]/self.ka1 + modifier[1]/self.ka2 + modifier[2]/self.ka2)
        
        numerator = self.vmax*(substrate[0]*substrate[1]/self.km_substrate1/self.km_substrate2)*(
            1-1/self.k_equilibrium*(product[0]*product[1]/substrate[0]/substrate[1]))
        v = numerator/denominator

        return v
    

# PGK is defined using reverse flux!
class Torch_Rev_BiBi_MM_Vr(torch.nn.Module):
    def __init__(self,
                 vmax: float,
                 k_equilibrium: float,
                 km_substrate1: float,
                 km_substrate2: float,
                 km_product1: float,
                 km_product2: float,
                 to_be_learned
                 ):
        super(Torch_Rev_BiBi_MM_Vr, self).__init__()
       

        # dictionary with all parameters
        params = {
            vmax: 'vmax',
            k_equilibrium: 'k_equilibrium',
            km_substrate1: 'km_substrate1',
            km_substrate2: 'km_substrate2',
            km_product1: 'km_product1',
            km_product2: 'km_product2',
        }

        # make parameters learnable/treat as a given
        for idx, (value, param_name) in enumerate(params.items()):
            if to_be_learned[idx]:
                self.__setattr__(
                    param_name, torch.nn.Parameter(torch.Tensor([value])))
            else:
                self.__setattr__(param_name, value)

    # s1 BPG s2 ADP
    # p1 P3G p2 ATP
    def calculate(self, substrate, product):
        denominator = (1 + substrate[0]/ self.km_substrate1 + product[0]/ self.km_product1)*\
            (1 + substrate[1]/ self.km_substrate2 + product[1]/ self.km_product2) * (self.km_product1 * self.km_product2)
        
        numerator = self.vmax*(substrate[0]*substrate[1] * self.k_equilibrium - product[0]*product[1])
        return numerator/denominator
# if __name__ == '__main__':
#     p_ATPase_ratio = 1
#     v_ATPase = Torch_ATPase(ATPase_ratio=p_ATPase_ratio, to_be_learned=[True])
#     print(v_ATPase.calculate(2,3 ))


class Torch_Irrev_Biomass(torch.nn.Module):
    def __init__(self,
                 vmax: float,
                 km_substrate: float,
                 a:float,
                 ms: float,
                 to_be_learned):
        super(Torch_Irrev_Biomass, self).__init__()

        if to_be_learned[0]:
            # make mu a learnable parameter
            self.vmax = torch.nn.Parameter(torch.tensor([vmax]))
        else:
            self.vmax = vmax
        if to_be_learned[1]:
            self.km_substrate = torch.nn.Parameter(torch.Tensor([km_substrate]))
        else:
            self.km_substrate = km_substrate
        if to_be_learned[2]:
            # make mu a learnable parameter
            self.a = torch.nn.Parameter(torch.tensor([a]))
        else:
            self.vmax = a
        if to_be_learned[3]:
            self.ms = torch.nn.Parameter(torch.Tensor([ms]))
        else:
            self.ms = ms

    def calculate(self, substrate):
        nominator = (self.vmax)*(substrate/self.km_substrate)
        denominator = (1+(substrate/self.km_substrate))
        qs=nominator/denominator
        qx=(qs-self.ms)/self.a
        return qs,qx
