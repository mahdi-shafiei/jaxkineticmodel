import torch
class SimpleActivator(torch.nn.Module):
    def __init__(self,
                 k_A:float,
                 to_be_learned):
        super(SimpleActivator, self).__init__()
        

        if to_be_learned[0]:
            self.k_A = torch.nn.Parameter(torch.Tensor([k_A]))
        else: 
            self.k_A = k_A

    def add_modifier(self, activator):
        return 1 + activator/self.k_A
    

class SimpleInhibitor(torch.nn.Module):
    def __init__(self,
                 k_I:float,
                 to_be_learned):
        super(SimpleInhibitor, self).__init__()
        

        if to_be_learned[0]:
            self.k_I = torch.nn.Parameter(torch.Tensor([k_I]))
        else: 
            self.k_I = k_I

    def add_modifier(self, inhibitor):
        return 1/ (1+ inhibitor/self.k_I) 