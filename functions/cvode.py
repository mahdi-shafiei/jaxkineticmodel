import abc
import torch
from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem

from .misc import _handle_unused_kwargs
import numpy as np


### torchdiffeq cvode
class CVodeWrapperODESolver(metaclass=abc.ABCMeta):
    def __init__(self, func, y0, rtol, atol, min_step=0, max_step=float('inf'), **unused_kwargs):
        unused_kwargs.pop('norm', None)
        unused_kwargs.pop('grid_points', None)
        unused_kwargs.pop('eps', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs
	
        self.dtype = y0.dtype
        self.device = y0.device
        self.shape = y0.shape
        self.y0 = y0.detach().cpu().numpy().reshape(-1)
        self.rtol = rtol
        self.atol = atol
        self.min_step = min_step
        self.max_step = max_step
        self.solver = "cvode"
        self.func = convert_func_to_numpy(func, self.shape, self.device, self.dtype)


    def integrate(self, t):
        if t.numel() == 1:
            return torch.tensor(self.y0)[None].to(self.device, self.dtype)    
        t = t.detach().cpu().numpy()


        mod = Explicit_Problem(self.func, self.y0, np.min(t))
        sim=CVode(mod)

        sim.rtol=self.rtol
        sim.atol=self.atol
        sim.verbosity=50
        sim.maxsteps=10000
        sim.time_limit=600
        sim.linear_solver = 'SPGMR'
        t,y=sim.simulate(tfinal=np.max(t),ncp=0,ncp_list=list(t))  
        sol = torch.tensor(y).to(self.device, self.dtype)
        sol = sol.reshape(-1, *self.shape)
        return sol
    
    @classmethod
    def valid_callbacks(cls):
        return set()


##required for conversion
def convert_func_to_numpy(func, shape, device, dtype):
    def np_func(t, y):
        t = torch.tensor(t).to(device, dtype)
        y = torch.reshape(torch.tensor(y).to(device, dtype), shape)
        with torch.no_grad():
            f = func(t, y)
        return f.detach().cpu().numpy().reshape(-1)
    return np_func
