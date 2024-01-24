import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
from torchdiffeq import odeint_adjoint as odeint

# MM with Keq


class Torch_Rev_UniUni_MM(torch.nn.Module):
    def __init__(self,
                 vmax: float,
                 k_equilibrium: float,
                 km_substrate: float,
                 km_product: float,
                 to_be_learned):
        super(Torch_Rev_UniUni_MM, self).__init__()
        # self.substrate=concentration[0]
        # self.product=concentration[1]

        # dictionary with all parameters
        params = {
            vmax: 'vmax',
            k_equilibrium: 'k_equilibrium',
            km_substrate: 'km_substrate',
            km_product: 'km_product'
        }

        # make parameters learnable/treat as a given
        for idx, (value, param_name) in enumerate(params.items()):
            if to_be_learned[idx]:
                self.__setattr__(
                    param_name, torch.nn.Parameter(torch.Tensor([value])))
            else:
                self.__setattr__(param_name, value)

    def calculate(self, substrate,product):
        nominator = self.vmax*(substrate/self.km_substrate) * \
            (1-(1/self.k_equilibrium)*(product/substrate))
        denominator = (1+(substrate/self.km_substrate) +
                       (product/self.km_product))
        return nominator/denominator


# separate for sink for simplicity
class Torch_MM_Sink(torch.nn.Module):
    def __init__(self,
                 v_sink: float,
                 km_sink: float,
                 to_be_learned):
        super(Torch_MM_Sink, self).__init__()

        if to_be_learned[0]:
            self.v_sink = torch.nn.Parameter(torch.Tensor([v_sink]))
        else:
            self.v_sink = v_sink

        if to_be_learned[1]:
            self.km_sink = torch.nn.Parameter(torch.Tensor([km_sink]))
        else:
            self.km_sink = km_sink

    def calculate(self, substrate):
        return self.v_sink * substrate / (substrate + self.km_sink)


class Torch_MM(torch.nn.Module):
    def __init__(self,
                 vmax: float,
                 km: float,
                 to_be_learned):
        super(Torch_MM, self).__init__()

        if to_be_learned[0]:
            self.vmax = torch.nn.Parameter(torch.Tensor([vmax]))
        else:
            self.vmax = vmax

        if to_be_learned[1]:
            self.km= torch.nn.Parameter(torch.Tensor([km]))
        else:
            self.km = km

    def calculate(self, substrate):
        return self.vmax * substrate / (substrate + self.km)
    
# the following two can be merged potentially:
class Torch_Irrev_MM_Uni(torch.nn.Module):
    def __init__(self,
                 vmax: float,
                 km_substrate: float,
                 to_be_learned):
        super(Torch_Irrev_MM_Uni, self).__init__()

        if to_be_learned[0]:
            # make mu a learnable parameter
            self.vmax = torch.nn.Parameter(torch.tensor([vmax]))
        else:
            self.vmax = vmax

        if to_be_learned[1]:
            self.km_substrate = torch.nn.Parameter(torch.Tensor([km_substrate]))
        else:
            self.km_substrate = km_substrate

    def calculate(self, substrate):
        nominator = (self.vmax)*(substrate/self.km_substrate)
        denominator = (1+(substrate/self.km_substrate))
        return nominator/denominator


class Torch_Irrev_MM_Bi(torch.nn.Module):
    def __init__(self,
                 vmax: float,
                 km_substrate1: float,
                 km_substrate2: float,
                 to_be_learned):
        super(Torch_Irrev_MM_Bi, self).__init__()
        
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
      
    def calculate(self, substrate):
        substrate1 = substrate[0]
        substrate2 = substrate[1]

        nominator = self.vmax*(substrate1/self.km_substrate1) * \
            (substrate2/self.km_substrate2)
        denominator = (1+(substrate1/self.km_substrate1)) * \
            (1+(substrate2/self.km_substrate2))
        return nominator/denominator

#two noncompeting substrate-product couples
class Torch_Rev_BiBi_MM(torch.nn.Module):
    def __init__(self,
                 vmax: float,
                 k_equilibrium: float,
                 km_substrate1: float,
                 km_substrate2: float,
                 km_product1: float,
                 km_product2: float,
                 to_be_learned
                 ):
        super(Torch_Rev_BiBi_MM, self).__init__()
       

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

    def calculate(self, substrate, product):
        # common_denominator = 1 + s1/kis1 + s2/kis2 + p1/kip1 + p2/kip2 + \
        #           s1*s2/(kis1*kms2) + p1*p2/(kip2*kmp1)

        # denominator = (1+(self.substrate[0]/self.km_substrate1) +
        #                (self.substrate[1]/self.km_substrate2) +
        #                (self.product[0]/self.km_product1) +
        #                (self.product[1]/self.km_product2) +
        #                ((self.substrate[0]*self.substrate[1])/(self.km_substrate1*self.ki_substrate2)) +
        #                ((self.product[0]*self.product[1])/(self.km_product2*self.ki_product1)))

        denominator = (1 + substrate[0]/ self.km_substrate1 + product[0]/ self.km_product1)*\
            (1 + substrate[1]/ self.km_substrate2 + product[1]/ self.km_product2)
        
        nominator = self.vmax*(substrate[0]*substrate[1]/self.km_substrate1/self.km_substrate2)*(
            1-1/self.k_equilibrium*(product[0]*product[1]/substrate[0]/substrate[1]))
        v = nominator/denominator

        return v


class Torch_MA_Irrev(torch.nn.Module):
    def __init__(self,
                 k_fwd: float,
                 to_be_learned):
        super(Torch_MA_Irrev, self).__init__()

        # self.substrate=substrate

        if to_be_learned[0]:
            # make k_fwd a learnable parameter
            self.k_fwd = torch.nn.Parameter(torch.Tensor([k_fwd]))
        else:
            self.k_fwd = k_fwd

    def calculate(self, substrate):
        return self.k_fwd * substrate

        

class Torch_Hill_Irreversible(torch.nn.Module):
    def __init__(self,
                 vmax: float,
                 hill: float,
                 k_half_substrate: float,
                 to_be_learned):
        super(Torch_Hill_Irreversible, self).__init__()

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

    def calculate(self, substrate):
        numerator = self.vmax * \
            ((substrate/self.k_half_substrate) ** self.hill)
        denominator = 1 + ((substrate/self.k_half_substrate) ** self.hill)
        return numerator/denominator


class Torch_Diffusion(torch.nn.Module):
    def __init__(self,
                 enzyme: float,
                 transport_coef: float,
                 to_be_learned):
        super(Torch_Diffusion, self).__init__()

        # self.substrate=substrate
        self.enzyme = enzyme
        if to_be_learned[0]:
            self.transport_coef = torch.nn.Parameter(
                torch.Tensor([transport_coef]))
        else:
            self.transport_coef = transport_coef

    def calculate(self, substrate):
        return self.transport_coef * (substrate - self.enzyme)


if __name__ == '__main__':
    # test = Torch_Hill_Irreversible()
    print()
