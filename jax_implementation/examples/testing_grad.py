import time

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp
import optax

class Robertson(eqx.Module):
    k1: float
    k2: float
    k3: float

    def __call__(self, t, y, args):
        f0 = -self.k1 * y[0] + self.k3 * y[1] * y[2]
        f1 = self.k1 * y[0] - self.k2 * y[1] ** 2 - self.k3 * y[1] * y[2]
        f2 = self.k2 * y[1] ** 2
        return jnp.stack([f0, f1, f2])

@jax.jit
def main(k1, k2, k3):
    robertson = Robertson(k1, k2, k3)
    terms = diffrax.ODETerm(robertson)
    t0 = 0.0
    t1 = 100.0
    y0 = jnp.array([1.0, 0.0, 0.0])
    dt0 = 0.0002
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=jnp.array([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]))
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6)
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    return sol

class NeuralODE(eqx.Module):
    func: Robertson
    def __init__(self,k1,k2,k3):
        self.func=Robertson(k1,k2,k3)
    
    def __call__(self,ts,y0):
        solution=diffrax.diffeqsolve(terms=diffrax.ODETerm(self.func),
                                     solver=diffrax.Kvaerno5(),
                                     t0=ts[0],
                                     t1=ts[-1],
                                     dt0=ts[1]-ts[0],
                                     y0=y0,
                                     saveat = diffrax.SaveAt(ts=ts),
                                     stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6))
        return solution.ys
    


model=NeuralODE(0.04, 3e7, 1e4)

print(model)

ts=jnp.arange(0,10,0.1)
y0 = jnp.array([1.0, 0.0, 0.0])

solution=model(ts,y0)
dataset=jnp.reshape(solution,(1,jnp.shape(solution)[0],jnp.shape(solution)[1]))
# print(dataset)

# start = time.time()
# sol = main(0.04, 3e7, 1e4)
# end = time.time()


# print("Results:")
# for ti, yi in zip(sol.ts, sol.ys):
    # print(f"t={ti.item()}, y={yi.tolist()}")
# print(f"Took {sol.stats['num_steps']} steps in {end - start} seconds.")

print(jnp.shape(dataset))
@eqx.filter_value_and_grad
def grad_loss(model, ti, yi):

    y_pred = jax.vmap(model,in_axes=(None,0))(ti,yi[:,0])

    return jnp.mean((yi - y_pred) ** 2)

loss,grads=grad_loss(model,ts,dataset)

print(loss,grads)


optim=optax.adabelief(1e-3)
opt_state=optim.init(eqx.filter(model,eqx.is_inexact_array))


updates,opt_state=optim.update(grads,opt_state)

print(updates)
