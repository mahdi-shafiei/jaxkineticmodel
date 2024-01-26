## A smaller system
## Added some boundaries into the training process. 

import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import inspect
import os
import time
from scipy.integrate import OdeSolver,DenseOutput
from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem
from assimulo.solvers import ExplicitEuler
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.integrate import OdeSolver,DenseOutput

from scipy.interpolate import CubicSpline
from torchdiffeq import odeint_adjoint 
from torch import optim

import torch 
import torch.nn as nn

sys.path.append('glycolysis/')

from fluxes import create_fluxes
fluxes=create_fluxes()


timeseries=pd.read_csv("data/FF2_timeseries_format.csv",index_col=0)
#we will start by a simple part #PGM, ENO, PYK
P3G=timeseries.index.to_list().index("IC3PG")
P2G=timeseries.index.to_list().index("IC2PG")
PEP=timeseries.index.to_list().index("ICPEP")
ADP=timeseries.index.to_list().index("ICADP")
ATP=timeseries.index.to_list().index("ICATP")
FBP=timeseries.index.to_list().index("ICFBP")
indices=[P3G,P2G,PEP,ADP,ATP,FBP]
#for odeint y0 must match
timeseries=timeseries.iloc[indices,:]
indices=[0,1,2,3,4,5]

# time points
time_points=timeseries.columns.to_list()
# time_points=timeseries.columns.to_list()
time_points=[float(i) for i in time_points]
# time_points=[int(i) for i in time_points]


#metabolite names
metabolites=["P3G","P2G","PEP","ADP","ATP","F16BP"]
metabolites=dict(zip(metabolites,indices))




#choose fluxes for this simpler problem
fluxes_dict={"v_PGM":fluxes['v_PGM'],"v_ENO":fluxes['v_ENO'],"v_PYK":fluxes['v_PYK']}


#tensor_c0
tensor_c0=torch.tensor(timeseries.iloc[:,0] ,dtype=torch.float64,requires_grad=False)
# metabolites=["P3G","P2G","PEP","ADP","ATP","F16BP"]
# for i in indices:
#     plt.plot(time_points,timeseries.iloc[i,:],label=metabolites[i])
# plt.legend()
# plt.show()


from scipy.integrate import OdeSolver,DenseOutput
from scipy.integrate import OdeSolver,DenseOutput
class CVODE(OdeSolver):
    def __init__(self,fun,t0,y0,t_bound,vectorized=False,rtol=1e-6,atol=1e-10,**extraneous):
        self.t=t0
        self.y0=y0
        self.t_bound=t_bound
        self.fun=fun
        #predefine the problem
        mod = Explicit_Problem(self.fun, self.y0, self.t)
        sim = CVode(mod)
        sim.rtol=rtol
        sim.atol=atol
        sim.verbosity=50
        sim.maxsteps=10000
        self.sim=sim

        ts,ys=self.sim.simulate(t_bound) #simulate until t_bound


        self.ts=ts
        self.ys=ys
        self.index=0 #determines which value to retrieve from ts,ys
        #get the 
        self.nfev=self.sim.statistics['nfcns']
        self.status="running"
        self.n=len(y0)
        self.direction=1
        self.njev=None
        self.nlu=None
    def _step_impl(self):
        #     #A solver must implement a private method _step_impl(self) which propagates a solver one step further. 
         #     #It must return tuple (success, message), 
        #     # where success is a boolean indicating whether a step was successful, 
        #     # and message is a string containing description of a failure if a step failed or None otherwise.
        index=self.index
        ts=self.ts
        ys=self.ys

        if index != len(ts)-1:
            t_new=ts[index+1]
            new_y=ys[index+1]
            self.y=new_y
            self.t=t_new
            # print("_step_impl",self.t)
            index+=1
            self.index=index
            return True, "worked"
        elif index==len(ts)-1:
            return False, "finished"
    
    def _dense_output_impl(self): # this seems to work now
        #A solver must implement a private method _dense_output_impl(self), 
        # which returns a DenseOutput object covering the last successful step.
        t=self.t
        t_old=self.t_old
        sim=self.sim
        y=self.y
        n=self.n
        if type(t)==list():
            t=t[0]
        

        return cvodeDenseOutput(sim,t_old,t,y,n)
        
        
class cvodeDenseOutput(DenseOutput):
    #what we need is the t's and y's, and the order 
    def __init__(self,sim,t_old,t,y,n):
        super().__init__(t_old, t)
        self.t_old=t_old
        self.sim=sim
        self.n=n
        self.y=y


    def _call_impl(self,t):

        #THIS IS NOT CORRECT. CVODE has an interpolation function that can be called, but I have not implemented it properly.
        # If the timesteps chosen by cvode are too big, t_eval becomes problematic as self.t contains multiple values
        #I do not expect that this is a problem for our data, as the the evaluated timepoints are so far apart that the scipy class does not pass multiple evaluated
        #time points
        
        self.sim.re_init(t0=self.t_old,y0=self.y)
        
        if len(t)>1:
            #If the number of timepoints that need to be interpolated is more than 1, we have simulate to each timepoint. 
            # This is probably not the best way to do it, but a workaround.
            ys=np.zeros((self.n,len(t)))
            for i in range(len(t)):
                t_i,y_i=self.sim.simulate(t[i])
                t_i=t_i[-1]
                y_i=y_i[-1,:]
                ys[:,i]=y_i
            y=np.reshape(ys,newshape=(self.n,len(t)))
                
            return y
        else:
            ts,y=self.sim.simulate(t)
            self.t=ts
            y=y[-1,:]
            y=np.reshape(y,newshape=(self.n,1))
            return y


class Glycolysis(torch.nn.Module):
    def __init__(self,
                 fluxes,
                 metabolites,
                 time_points,
                 data): 
        super(Glycolysis, self).__init__()
        self.fluxes = nn.ParameterDict(fluxes) # dict with fluxes
        self.metabolites = metabolites # dict with indices of each metabolite
        self.time_points=torch.tensor(time_points,dtype=torch.float64,requires_grad=False)
        self.data=torch.tensor(np.array(data),dtype=torch.float64,requires_grad=False) #convert dataset to torch tensor with gradient


        
    def calculate_fluxes(self,concentrations):
        
        self.fluxes['v_PGM'].value=self.fluxes['v_PGM'].calculate(concentrations[self.metabolites['P3G']], concentrations[self.metabolites['P2G']])
        self.fluxes['v_ENO'].value=self.fluxes['v_ENO'].calculate(concentrations[self.metabolites['P2G']], concentrations[self.metabolites['PEP']])
        self.fluxes['v_PYK'].value =self.fluxes['v_PYK'].calculate([concentrations[self.metabolites['PEP']], concentrations[self.metabolites['ADP']]],
                                      concentrations[self.metabolites['ATP']], 
                                      concentrations[self.metabolites['F16BP']])

    # def interpolate(self,t,ind): 
    # #finds the closests values to _ called by the solver, and return the interpolated value between y1 and y0
    # #ind is the index which we dont want to learn
    #     subset_data=self.data[ind,:] #subsets the index we do not want to train on
    #     # print("sub",subset_data.requires_grad)
    #     # print(self.time_points.requires_grad)
        
    #     distances=torch.abs(t-self.time_points)

    #     min_distance_indices = torch.argsort(distances)
    #     closest_indices=min_distance_indices[:2]
    #     closest_indices_sorted=torch.sort(closest_indices).values
    #     closest_values = self.time_points[closest_indices_sorted]

    #     #Perform a linear interpolation, a=(y1-y0)/t1-t0,new_y=ax+y0
    #     t_diff=torch.abs(closest_values[1]-closest_values[0])
    #     # print("t_diff",t_diff.requires_grad)
    #     y_diff=subset_data[closest_indices_sorted[1]]-subset_data[closest_indices_sorted[0]]
    #     a=y_diff/t_diff
    #     # print("a",a.requires_grad)
    #     t_change=t-closest_values[0]
    #     new_y=(a*t_change)+subset_data[closest_indices_sorted[0]]
    #     # print("new y:",new_y.requires_grad)
    #     # print("subset_data",subset_data.requires_grad)
    #     # print("time_point",self.time_points.requires_grad)
    #     return new_y
    
    def interpolate(self,t,ind):
        subset_data=self.data[ind,:] #subsets the index we do not want to train on
        subset_data=subset_data.detach().numpy()
        # value=np.interp(t,self.time_points,subset_data)
        values=CubicSpline(self.time_points,subset_data)
        value=values(t)
        # print(value)
        value=torch.tensor(value,dtype=torch.float64,requires_grad=False)
        # print(value)
        return value

    
    def forward(self,_,conc_in):
        #Define some boundary conditions,i.e., these metabolites we assume to be measured and used for the calculation
        with torch.no_grad():
            conc_in[self.metabolites['F16BP']]=self.interpolate(_,5)
            conc_in[self.metabolites['ADP']]=self.interpolate(_,3)
            conc_in[self.metabolites['ATP']]=self.interpolate(_,4)

        self.calculate_fluxes(conc_in)
        P3G= -self.fluxes['v_PGM'].value
        P2G=+ self.fluxes['v_PGM'].value - self.fluxes['v_ENO'].value
        PEP=self.fluxes['v_ENO'].value - self.fluxes['v_PYK'].value 
        ADP=-self.fluxes['v_PYK'].value 
        ATP=+self.fluxes['v_PYK'].value
        F16BP=torch.Tensor([0])
        
        P3G=P3G*0.001452
        P2G=P2G*0.000164
        PEP=PEP*0.000902
        ADP=ADP*0.000421
        ATP=ATP*0.001099
        F16BP=F16BP*0.001045

        return torch.cat(([P3G,P2G,PEP,ADP,ATP,F16BP]),dim=0)




glycolysis=Glycolysis(fluxes=fluxes_dict, metabolites=metabolites,time_points=time_points,data=timeseries)

tensor_concentrations=torch.tensor(np.array(timeseries.T),dtype=torch.float64,requires_grad=False)
tensor_timepoints=torch.tensor(time_points,dtype=torch.float64,requires_grad=False)

print("number of timepoints",len(tensor_timepoints))
#lets try a smaller problem

print("is running")
def loss_func():
    scaling=1/(torch.max(tensor_concentrations,0)[0]-torch.min(tensor_concentrations,0)[0])
    tensor_c0=tensor_concentrations[0]#[0,:]
    indices=[0,1,2] #the other ones are basically given  in the forward pass and shouldnt be taken in the loss function
    target=tensor_concentrations
        # print(target)
    predicted_c =odeint_adjoint(func=glycolysis, y0=tensor_c0, t=tensor_timepoints,method="scipy_solver",options={"solver":CVODE},atol=1e-10,rtol=1e-6)
    try:
        # print(predicted_c)
        target=scaling*target[None,:][0]
        
        predicted_c=predicted_c

        predicted_c=scaling*predicted_c[None,:][0]
        predicted_c=predicted_c[:,indices]
        target=target[:,indices]

        ls =torch.mean(torch.square((predicted_c - target)))
    
        if ls.isnan():
            print("not normal")

        # ls=torch.Tensor(ls, requires_grad = False)
    except:
        print("end")
        pass
        
    return ls

scaling=1/(torch.max(tensor_concentrations,0)[0]-torch.min(tensor_concentrations,0)[0])
tensor_c0=tensor_concentrations[0]#[0,:]
indices=[0,1,2] #the other ones are basically given  in the forward pass and shouldnt be taken in the loss function
target=tensor_concentrations
        # print(target)


optimizer=optim.AdamW(glycolysis.parameters(), lr=1e-2)
# print(len(optimizer.param_groups[0]['params']))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20)

get_loss_per_iteration=[]
loss=0
# # # batches=3

for i in range(0,15,1):
    print(i)

    scheduler_step=True
    optimizer.zero_grad()
    try:
        loss = loss_func()
    except:
        print("continue")
        pass
    get_loss_per_iteration.append(loss.detach().numpy())
    
    print(loss)

    loss.backward()
    optimizer.step()
    
    # print(loss.backward())
    if scheduler_step:
        scheduler.step(loss)

plt.plot(get_loss_per_iteration)
plt.show()

tensor_timepoints=torch.linspace(0,2000,3000)
predicted_c =odeint_adjoint(func=glycolysis, y0=tensor_c0, t=tensor_timepoints,method="scipy_solver",options={"solver":"LSODA"},atol=1e-6,rtol=1e-3)
predicted_c=predicted_c.detach().numpy()
for i in indices:
    plt.plot(tensor_timepoints,predicted_c[:,i])
plt.legend()
plt.show()
