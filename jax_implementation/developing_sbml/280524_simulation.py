import sympy as sp
import jax.numpy as jnp
from sympy.utilities.lambdify import lambdify
import jax
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt
import diffrax
from diffrax import diffeqsolve, ODETerm, Dopri5,SaveAt
import time
from functools import partial

jax.config.update("jax_enable_x64", True)

#some test equations
A,Ks,Vmax=sp.symbols('A Ks Vmax')
test_equation1="(Vmax*A)/(A+Ks)"
local_dict1={"A":A,"Ks":Ks,"Vmax":Vmax}

B,Ks,Vmax=sp.symbols('B Ks Vmax') #bug
test_equation2="(Vmax*B)/(B+Ks)"
local_dict2={"B":B,"Ks":Ks,"Vmax":Vmax}

Stoich=jnp.array([[-1,1,0],[0,-1,1]]).T



def sympify_lambidify_and_jit_equation(equation,local_dict):
    """Sympifies, lambdifies, and then jits a string rate law"""  ## add the jitting later, see how it affects time
    equation=sp.sympify(equation,locals=local_dict)
    equation=sp.lambdify((local_dict.values()),equation,"jax")
    equation=jax.jit(equation)
    return equation



equation1=sympify_lambidify_and_jit_equation(test_equation1,local_dict1)
equation2=sympify_lambidify_and_jit_equation(test_equation2,local_dict2)
equations=[equation1,equation2]

class TorchKineticModel():
    def __init__(self,v,S,flux_point_dict):#params,
        """Initialize given the following arguments:
        v: the flux functions given as lambidified jax functions,
        S: a stoichiometric matrix. For now only support dense matrices, but later perhaps add for sparse
        params: kinetic parameters
        flux_point_dict: a dictionary for each vi that tells what the corresponding metabolites should be in y. Should be matched to S.
        ##A pointer dictionary?
        """
        super().__init__()
        self.func=v
        self.stoichiometry=S
        # self.params=params
        self.flux_point_dict=flux_point_dict #this is ugly but wouldnt know how to do it another wa


    def __call__(self,t,y,params):
        """I explicitly add params to call for gradient calculations. Find out whether this is actually necessary"""
        def apply_func(i,y):
            vi=self.func[i](y[self.flux_point_dict[i]],**params)
            return vi
        # Vectorize the application of the functions
        indices = np.arange(len(self.func))

        v=jnp.stack([apply_func(i,y) for i in indices]) #perhaps there is a way to vectorize this in a better way
        
        dY = jnp.matmul(self.stoichiometry, v)[:, 0] #dMdt=S*v(t)
        return dY        

class NeuralODE():
    func: TorchKineticModel
    def __init__(self,v,S,flux_point_dict,params):
        super().__init__()
        self.func=TorchKineticModel(v,S,flux_point_dict)
        self.params=params

    def __call__(self,ts,y0,params):
        solution = diffrax.diffeqsolve(
        diffrax.ODETerm(self.func),
        diffrax.Kvaerno5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0,
        args=params,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys



pars={'Vmax':1.0,'Ks':3.0}
flux_point_dict={0:jnp.array([0]),1:jnp.array([1])}
model = NeuralODE(v=equations, S=Stoich, flux_point_dict=flux_point_dict,params=pars)
model=jax.jit(model)

y0=jnp.array([3,0,1])
ts=jnp.arange(0,10,0.1)
ys=model(ts,y0,pars)


def test_speed_param_perturbation(model,pars):
    ts=jnp.arange(0,10,0.1)
    ys=model(ts,y0,pars)

    values=np.arange(0,100,0.1)

    start=time.time()
    for i in values:
            pars['Vmax']=i
            ys=model(ts,y0,pars)
            plt.plot(ts,ys)
    plt.show()
    end=time.time()
    return (start-end)/len(values)


def loss_func(params,ts,ys):
    """A very simple loss function, later we need to add functionality for missing data"""
    y_pred=model(ts,y0,params)
    return jnp.mean((ys - y_pred) ** 2)

pars['Vmax']=2.0

loss=loss_func(pars,ts,ys)
print(loss)
grads=jax.jit(jax.grad(loss_func,0))



values=np.arange(0,100,0.1)
for i in values:

    pars['Vmax']=i
    print(pars)
    print(grads(pars,ts,ys))


test_speed_param_perturbation(model,pars)