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
import optax


###
from sbml_load import sympify_lambidify_and_jit_equation
from jax_kinetic_model import JaxKineticModel, NeuralODE


jax.config.update("jax_enable_x64", True)

#some test equations
A,Ks,Vmax1=sp.symbols('A Ks Vmax1')
test_equation1="(Vmax*A)/(A+Ks)"
local_dict1={"A":A,"Ks":Ks,"Vmax":Vmax1}

B,Ks,Vmax=sp.symbols('B Ks Vmax') #bug
test_equation2="(Vmax*B)/(B+Ks)"
local_dict2={"B":B,"Ks":Ks,"Vmax":Vmax}

Stoich=jnp.array([[-1,1,0],[0,-1,1]]).T
print(jnp.shape(Stoich))

equation1=sympify_lambidify_and_jit_equation(test_equation1,local_dict1)
equation2=sympify_lambidify_and_jit_equation(test_equation2,local_dict2)
equations=[equation1,equation2]



params={'Vmax':1.0,'Ks':3.0,"Vmax1":2} #would be local and global params
flux_point_dict={0:jnp.array([0]),1:jnp.array([1])}
model = NeuralODE(v=equations, S=Stoich, flux_point_dict=flux_point_dict,params=params)


y0=jnp.array([3,0,1])
ts=jnp.arange(0,10,0.1)
ys=model(ts,y0,params)


## timing of parameter perturbations
def test_speed_param_perturbation(model,params):
    ts=jnp.arange(0,10,0.1)
    ys=model(ts,y0,params)

    values=np.arange(0,100,0.1)

    start=time.time()
    for i in values:
            params['Vmax']=i
            ys=model(ts,y0,params)
            plt.plot(ts,ys)
    plt.show()
    end=time.time()
    return (start-end)/len(values)


# ### simple learning system to test whether grads works as expected
def loss_func(params,ts,ys):
    """A very simple loss function, later we need to add functionality for missing data"""
    y_pred=model(ts,y0,params)
    return jnp.mean((ys - y_pred) ** 2)

params['Vmax']=0.5

ys_off=model(ts,y0,params)



lr=1e-3
optimizer = optax.adam(lr)
opt_state = optimizer.init(params)


@jax.jit
def update(opt_state,params,ts,ys):
    """Update rule for the gradients for parameters"""
    loss=loss_func(params,ts,ys)
    grads=jax.jit(jax.grad(loss_func,0))(params,ts,ys) #loss w.r.t. parameters
    updates,opt_state=optimizer.update(grads,opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state,params,loss,grads





epochs=199
grads_list=[]
for _ in range(epochs):
    opt_state,params,loss,grads=update(opt_state,params,ts,ys)
    grads_list.append(float(grads['Vmax']))
    print(_,params,loss)

ys=model(ts,y0,params)

# plt.plot(np.arange(epochs),grads_list,label="gradient vmax")
# plt.legend()
# plt.show()