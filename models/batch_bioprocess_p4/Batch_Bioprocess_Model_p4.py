
import torch 
import torch.nn as nn

    
 

# class Bioprocess(nn.Module):
#     """ 
#     Batch Bioprocess. # parameter_dictionary
#     """
#     def __init__(self, 
#                  parameter_dict
#                  ):
#         super().__init__() 
#         self.qsmax = torch.nn.Parameter(torch.tensor(parameter_dict['qsmax'])) # make mu a learnable parameter
#         self.Ks = torch.nn.Parameter(torch.tensor(parameter_dict['Ks']))
#         self.a = torch.nn.Parameter(torch.tensor(parameter_dict['a']))
#         self.ms = torch.nn.Parameter(torch.tensor(parameter_dict['ms']))
        
#     def forward(self, 
#                 t: float, # time index
#                 state:torch.TensorType, # state of the system first dimension is the batch size
#                 ) -> torch.Tensor: # return the derivative of the state
#         """ 
#             Define the right hand side of the VDP oscillator.
#         """
#         Cs=state[:,0]
#         Cx=state[:,1]
        
#         if Cs<0:
#             Cs=0
#         #Kinetic equations
#         qs=self.qsmax*(Cs/(self.Ks+Cs))
#         qx=(qs-self.ms)/self.a

#         #Rate equations
#         Rs=qs*Cx*1
#         Rx=qx*Cx*1
#         #Rc=-(Rs+Rx)  #Rco2 and Ro2
        
#         dfunc = torch.zeros_like(state)
#         dfunc[:,0]=Rs
#         dfunc[:,1]=Rx
#         print(dfunc)
#         return dfunc
    
#     def __repr__(self):
#         """Print the parameters of the model."""
#         return f" qsmax: {self.qsmax.item()}, a:{self.a.item()}, ms:{self.ms.item()}, Ks:{self.Ks.item()}"

    
class Bioprocess(nn.Module):
    """ 
    Batch Bioprocess. # parameter_dictionary
    """
    def __init__(self, 
                 parameter_dict
                 ):
        super().__init__() 
        self.qsmax = torch.nn.Parameter(torch.tensor(parameter_dict['qsmax'])) # make mu a learnable parameter
        self.Ks = torch.nn.Parameter(torch.tensor(parameter_dict['Ks']))
        self.a = torch.nn.Parameter(torch.tensor(parameter_dict['a']))
        self.ms = torch.nn.Parameter(torch.tensor(parameter_dict['ms']))
        
    def forward(self, 
                t: float, # time index
                state:torch.TensorType, # state of the system first dimension is the batch size
                ) -> torch.Tensor: # return the derivative of the state

        Cs=state[0]
        Cx=state[1]
        
        if Cs<0:
            Cs=0
        #Kinetic equations
        qs=self.qsmax*(Cs/(self.Ks+Cs))
        qx=(qs-self.ms)/self.a
        #Rate equations
        
        Rs=qs*Cx
        Rx=qx*Cx
        dfunc = torch.zeros_like(state)
        dfunc[0]=Rs
        dfunc[1]=Rx
        
        return dfunc
    
    def __repr__(self):
        """Print the parameters of the model."""
        return f" qsmax: {self.qsmax.item()}, a:{self.a.item()}, ms:{self.ms.item()}, Ks:{self.Ks.item()}"