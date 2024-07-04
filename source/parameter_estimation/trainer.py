

import sys, os
import jax
import diffrax
sys.path.append('/home/plent/Documenten/Gitlab/NeuralODEs/jax_neural_odes')
jax.config.update("jax_enable_x64", True)
import optax
import jax.numpy as jnp 


class Trainer:
    def __init__(self,model,dataset,lr,max_iter):
        super(Trainer,self).__init__()

        self.model=model
        self.ys=jnp.array(dataset)
        self.y0=self.ys[0,:]
        self.max_iter=max_iter

        self.ts=jnp.array(list(dataset.index))
        self.optimizer=optax.adabelief(lr)

    
    def loss_func(self,params,ts):
        """mean squared error (with mask for nans)"""
        
        mask=~jnp.isnan(jnp.array(self.ys))
        yscale=jnp.nanmax(self.ys,axis=0)-jnp.nanmin(self.ys,axis=0)
        y_pred=self.model(ts,self.y0,params)*(1/yscale)
        ys=self.ys*(1/yscale)
        ys = jnp.where(mask, ys, 0)
        y_pred = jnp.where(mask, y_pred, 0)
        non_nan_count = jnp.sum(mask)
        loss = jnp.sum((y_pred - ys) ** 2) / non_nan_count
        return loss
    
    @jax.jit
    def update(self,opt_state,params,ts):
        """Update rule for the gradients for parameters"""
        loss=self.loss_func(params,ts)
        grads=jax.jit(jax.grad(self.loss_func,0))(params,ts) #loss w.r.t. parameters
        updates,opt_state=self.optimizer.update(grads,opt_state)
        params = optax.apply_updates(params, updates)
        return opt_state,params,loss,grads
    
    def train(self,params):
        # Initialize parameters and optimizer state
        opt_state = self.optimizer.init(params)
        
        for iteration in range(self.max_iter):
            opt_state, params, loss, grads = self.update(opt_state, params, self.ts)
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss}")