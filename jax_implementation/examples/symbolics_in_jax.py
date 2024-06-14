import sympy as sp
import jax.numpy as jnp
from sympy.utilities.lambdify import lambdify
import jax
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt
import diffrax
from diffrax import diffeqsolve, ODETerm, Dopri5,SaveAt,Kvaerno5

jax.config.update("jax_enable_x64", True)

f0_str = "-k1 * y0 + k3 * y1 * y2"
f1_str = "k1 * y0 - k2 * y1**2 - k3 * y1 * y2"
f2_str = "k2 * y1**2"

f0=sp.sympify(f0_str)
f1=sp.sympify(f1_str)
f2=sp.sympify(f2_str)
y0 ,y1 ,y2, k1, k2, k3=sp.symbols('y0 y1 y2 k1 k2 k3')

expected_vars=[y0 ,y1 ,y2, k1, k2, k3]
f0=sp.lambdify(expected_vars, f0,"jax")
f1=sp.lambdify(expected_vars, f1,"jax")
f2=sp.lambdify(expected_vars,f2,"jax")

f0_jit=jax.jit(f0)
f1_jit=jax.jit(f1)
f2_jit=jax.jit(f2)
f_jitted_list=[f0_jit,f1_jit,f2_jit]



class torch_kinetic_model():
    def __init__(self,f_jitted,pars):
        #add a dictionary of what to evaluate self.fun on given y
        self.func=f_jitted
        self.pars=pars

    def __call__(self,t,y,*args):

        # Create the dictionary with scalar values
        # met1 = {'y0': y.get[0], 'y1': y[1], 'y2': y[2]}
        met1 = {'y0': y[0], 'y1': y[1], 'y2': y[2]}
        dy0=self.func[0](**met1,**self.pars)

        dy1=self.func[1](**met1,**self.pars)
        dy2=self.func[2](**met1,**self.pars)

        return jnp.stack([dy0,dy1,dy2])
    
#this works amazingly! We can simply jit compile once, and then change parameters
class NeuralODE():
    func: torch_kinetic_model

    def __init__(self,f_jitted,pars, **kwargs):
        super().__init__(**kwargs)
        self.func = torch_kinetic_model(f_jitted,pars)

    def __call__(self, ts, y0,**kwargs):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Kvaerno5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys


ys = jnp.array([[4.0, 5.0, 6.0]])
pars={'k1':0.04,'k2':3e7,'k3':1e4}
mymodel=jax.jit(NeuralODE(f_jitted_list,pars=pars))
ts=jnp.arange(0,3,0.1)
y0=jnp.array([1.0,0.0,0.0])


dataset=mymodel(ts,y0)
dataset=jnp.reshape(dataset,(1,jnp.shape(dataset)[0],jnp.shape(dataset)[1]))
print(np.shape(dataset))

@eqx.filter_value_and_grad
def grad_loss(model, ti, yi):

    y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
    return jnp.mean((yi - y_pred) ** 2)

loss,grad=grad_loss(mymodel,ts,dataset)
print(loss,grad)