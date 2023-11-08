from scipy.integrate import OdeSolver,DenseOutput



from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem
from assimulo.solvers import ExplicitEuler
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.integrate import OdeSolver,DenseOutput


def lorenz(t,X):
    sigma=10 
    beta=2.6667
    rho=28
    """The Lorenz equations."""
    xdot=np.zeros(3)
    u, v, w = X[0],X[1],X[2]
    xdot[0] = -sigma*(u - v)
    xdot[1] = rho*u - v - u*w
    xdot[2] = -beta*w + u*v
    return xdot

y0 =[0, 1, 1.05]
t0= 2

from scipy.integrate import OdeSolver,DenseOutput
class CVODE(OdeSolver):
    def __init__(self,fun,t0,y0,t_bound,vectorized=False,rtol=1e-3,atol=1e-6,**extraneous):
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
        sim.maxsteps=20000
        sim.report_continuously=True
        sim.time_limit=40


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

        
        





#from scipy.integrate import solve_ivp
#solution=solve_ivp(lorenz,t_span=[0,200],t_eval=np.arange(0.0,200,0.001),y0=y0,method=CVODE) 
# solution=solve_ivp(lorenz,t_span=[0,200],y0=y0,method=CVODE) 
#plt.plot(solution.y[0,:],solution.y[2,:])
#plt.show()

## Tests with torchdiffeq works, but interpolation required for t_eval is not finished yet. 
#t_eval=np.arange(0.0,200,0.001)
#print(len(t_eval))
#print(len(solution.y[0,:]))
