

import sys, os
import jax
import diffrax
sys.path.append('/home/plent/Documenten/Gitlab/NeuralODEs/jax_neural_odes')
jax.config.update("jax_enable_x64", True)
import optax
import jax.numpy as jnp 



def log_transform_parameters(params):
    params_dict={}
    for key in params.keys():
        values=jnp.log2(params[key])
        params_dict[key]=values
    return params_dict

def exponentiate_parameters(params):
    params_dict={}
    for key in params.keys():
        values=2**params[key]
        params_dict[key]=values
    return params_dict

    

def create_loss_func(model):
    def loss_func(params,ts,ys):
        mask=~jnp.isnan(jnp.array(ys))
        ys=jnp.atleast_2d(ys)
        y0=ys[0,:]
        y_pred=model(ts,y0,params)
        ys = jnp.where(mask, ys, 0)
        y_pred = jnp.where(mask, y_pred, 0)
        # print(ys,y_pred)
        non_nan_count = jnp.sum(mask)
        
        loss = jnp.sum((y_pred - ys) ** 2) / non_nan_count
        return loss
    return loss_func

def create_log_params_loss_func(model):
    """Loss function for log transformed parameters """
    def loss_func(params,ts,ys):
        params=exponentiate_parameters(params)
        mask=~jnp.isnan(jnp.array(ys))
        ys=jnp.atleast_2d(ys)
        y0=ys[0,:]
        y_pred=model(ts,y0,params)
        ys = jnp.where(mask, ys, 0)
        y_pred = jnp.where(mask, y_pred, 0)
        # print(ys,y_pred)
        non_nan_count = jnp.sum(mask)
        
        loss = jnp.sum((y_pred - ys) ** 2) / non_nan_count
        return loss
    return loss_func

def create_log_params_log_loss_func(model):
    """Loss function for log transformed parameters """
    def loss_func(params,ts,ys):
        params=exponentiate_parameters(params)
        mask=~jnp.isnan(jnp.array(ys))
        ys=jnp.atleast_2d(ys)
        y0=ys[0,:]
        y_pred=model(ts,y0,params)

        y_pred=jnp.log2(y_pred+1)
        ys=jnp.log2(ys+1)


        ys = jnp.where(mask, ys, 0)
        y_pred = jnp.where(mask, y_pred, 0)
        # print(ys,y_pred)
        non_nan_count = jnp.sum(mask)
        
        loss = jnp.sum((y_pred - ys) ** 2) / non_nan_count
        return loss
    return loss_func

def create_log_params_means_centered_loss_func(model):
    """Loss function for log transformed parameters. 
    We do a simple input scaling using the mean per state variable (we add 1 everywhere to prevent division by zero) """
    def loss_func(params,ts,ys):

        params=exponentiate_parameters(params)
        mask=~jnp.isnan(jnp.array(ys))
        ys=jnp.atleast_2d(ys)
        y0=ys[0,:]
        y_pred=model(ts,y0,params)
        ys = jnp.where(mask, ys, 0)

        ys=ys+1
        y_pred=y_pred+1
        scale=jnp.mean(ys,axis=0)

        ys=ys/scale
        y_pred=y_pred/scale

        y_pred = jnp.where(mask, y_pred, 0)
        # print(ys,y_pred)
        non_nan_count = jnp.sum(mask)

        loss = jnp.sum((y_pred - ys) ** 2) / non_nan_count
        return loss
    return loss_func


def create_log_params_means_centered_loss_func2(model,to_include:list):
    """Loss function for log transformed parameters. 
    We do a simple input scaling using the mean per state variable (we add 1 everywhere to prevent division by zero). Furthermore, we allow for not every state variable to be learned (sometimes it is not in the model for example)"""
    def loss_func(params,ts,ys):

        params=exponentiate_parameters(params)
        mask=~jnp.isnan(jnp.array(ys))
        ys=jnp.atleast_2d(ys)
        y0=ys[0,:]
        y_pred=model(ts,y0,params)
        ys = jnp.where(mask, ys, 0)

        ys=ys+1
        y_pred=y_pred+1
        scale=jnp.mean(ys,axis=0)

        ys=ys/scale
        y_pred=y_pred/scale

        y_pred = jnp.where(mask, y_pred, 0)


            
        ys=ys[:,to_include]
        y_pred=y_pred[:,to_include]
        # print(ys,y_pred)
        non_nan_count = jnp.sum(mask)

        loss = jnp.sum((y_pred - ys) ** 2) / non_nan_count
        return loss
    return loss_func


def create_update_rule(optimizer,loss_func):
    def update(opt_state,params,ts,ys):
        """Update rule for the gradients for parameters"""
        loss=loss_func(params,ts,ys)
        grads=jax.jit(jax.grad(loss_func,0))(params,ts,ys) #loss w.r.t. parameters
        updates,opt_state=optimizer.update(grads,opt_state)
        params = optax.apply_updates(params, updates)
        return opt_state,params,loss,grads
    return update



@jax.jit
def create_log_update_rule(optimizer,log_loss_func):
    def update_log(opt_state,params,ts,ys):
        """Update rule for the gradients for log-transformed parameters. Can only be applied
        to nonnegative parameters"""
        log_params=log_transform_parameters(params)

        loss=log_loss_func(log_params,ts,ys)

        grads=jax.jit(jax.grad(log_loss_func,0))(log_params,ts,ys) #loss w.r.t. parameters
        updates,opt_state=optimizer.update(grads,opt_state)

        #we perform updates in log space, but only return params in lin space
        log_params = optax.apply_updates(log_params, updates)
        lin_params = exponentiate_parameters(log_params) 
        return opt_state,lin_params,loss,grads
    return update_log
        

def global_norm(grads):
    """Calculate the global norm of a list of gradient arrays."""
    global_norm=[]
    for key in grads.keys():

        value=float(grads[key])**2
        global_norm.append(value)
    global_norm=jnp.sqrt(jnp.sum(jnp.array(global_norm)))
    return global_norm
