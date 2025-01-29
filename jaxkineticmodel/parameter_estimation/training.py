import jax
import optax
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
from jaxkineticmodel.load_sbml.sbml_load import get_global_parameters
from scipy.stats import qmc
import logging
from jaxkineticmodel.building_models import JaxKineticModelBuild as jkm
import numpy as np
import jax.numpy as jnp
import pandas as pd


jax.config.update("jax_enable_x64", True)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class that fits data using a set of parameter initializations,
    Input: a model that is a JaxKineticModel class to fit, and a dataset"""

    def __init__(self, model, data: pd.DataFrame, learning_rate=1e-3, loss_threshold=1e-4, clip=4, n_iter=100):

        if isinstance(model, SBMLModel):
            self.model = jax.jit(model.get_kinetic_model())
            self.parameters = list(model.parameters.keys())

        if isinstance(model, jkm.NeuralODEBuild):
            logger.info("type not tested yet")
            self.parameters = model.parameter_names
            self.model = jax.jit(model)

        self.ts = jnp.array(list(data.index))

        self.lr = learning_rate
        self.optimizer = self._add_optimizer(clip=clip)
        self.loss_threshold = loss_threshold
        self.n_iter = n_iter
        self.log_loss_func = jax.jit(create_log_params_means_centered_loss_func(self.model))
        self.dataset = data
        self.parameter_sets = None

    def _generate_bounds(self, parameters_base: dict, lower_bound: float, upper_bound: float):
        """Generates bounds given an estimate of the parameters
        Input:
         Base parameter: a rough estimate of the parameter values
          Lower bound: Defines the lower bound given the parameter base values
           Upper bound: defines upper bound given the parameter base values"""
        lbs, ubs, names = [], [], []

        for key in parameters_base.keys():
            if parameters_base[key] != 0:
                lb = parameters_base[key] * lower_bound
                ub = parameters_base[key] * upper_bound
            else:
                lb = 0
                ub = 0.00001
            lbs.append(lb)
            ubs.append(ub)
            names.append(key)
        bounds = pd.DataFrame({"lb": lbs, "ub": ubs}, index=names)

        return bounds

    def latinhypercube_sampling(self, parameters_base, lower_bound, upper_bound, N):
        """Performs latin hypercube sampling"""
        bounds = self._generate_bounds(parameters_base=parameters_base, lower_bound=lower_bound, upper_bound=upper_bound)

        sampler = qmc.LatinHypercube(d=len(bounds.index))
        samples = sampler.random(N)

        lb = bounds["lb"]
        ub = bounds["ub"]

        sample_scaled = qmc.scale(samples, lb, ub)
        names = list(bounds.index)
        parameter_sets = []
        for i in range(np.shape(sample_scaled)[0]):
            parameter_sets.append(dict(zip(names, sample_scaled[i, :])))
        parameter_sets = pd.DataFrame(parameter_sets)
        self.parameter_sets = parameter_sets
        return parameter_sets

    def _add_optimizer(self, clip):
        optimizer = optax.adabelief(self.lr)
        clip_by_global = optax.clip_by_global_norm(np.log(clip))
        optimizer = optax.chain(optimizer, clip_by_global)
        self.optimizer = optimizer
        return optimizer

    # @jax.jit
    def _update_log(self, opt_state, params, ts, ys):
        """Update rule for the gradients for log-transformed parameters. Can only be applied
        to non-negative parameters"""

        log_params = log_transform_parameters(params)

        loss = self.log_loss_func(log_params, ts, ys)

        grads = jax.grad(self.log_loss_func, 0)(log_params, ts, ys)  # loss w.r.t. parameters
        updates, opt_state = self.optimizer.update(grads, opt_state)

        # we perform updates in log space, but only return params in lin space
        log_params = optax.apply_updates(log_params, updates)
        lin_params = exponentiate_parameters(log_params)
        return opt_state, lin_params, loss, grads

    def train(self):
        """Train model given the initializations"""
        loss_per_iteration_dict = {}
        optimized_parameters_dict = {}
        global_norm_dict = {}

        # loop over parameter sets
        for init in range(np.shape(self.parameter_sets)[0]):
            params_init = self.parameter_sets.iloc[init, :].to_dict()
            opt_state = self.optimizer.init(params_init)

            loss_per_iter = []
            gradient_norms = []
            try:
                for step in range(self.n_iter):  # loop over number of iterations
                    opt_state, params_init, loss, grads = self._update_log(
                        opt_state, params_init, self.ts, jnp.array(self.dataset)
                    )

                    gradient_norms.append(global_norm(grads))
                    loss_per_iter.append(loss)
                    if loss < self.loss_threshold:
                        logger.info("loss threshold reached")
                        loss_per_iteration_dict[init] = loss_per_iter
                        global_norm_dict[init] = gradient_norms
                        optimized_parameters_dict[init] = params_init
                        break

                    if step % 50 == 0:
                        print(f"Step {step}, Loss {loss}")

                    loss_per_iteration_dict[init] = loss_per_iter
                    optimized_parameters_dict[init] = params_init
                    global_norm_dict[init] = gradient_norms

            except:
                logger.error(f"init {init} could not be optimized")
                loss_per_iteration_dict[init] = loss_per_iter
                loss_per_iteration_dict[init].append(-1)
        return optimized_parameters_dict, loss_per_iteration_dict, global_norm_dict


def log_transform_parameters(params):
    params_dict = {}
    for key in params.keys():
        values = jnp.log2(params[key])
        params_dict[key] = values
    return params_dict


def exponentiate_parameters(params):
    params_dict = {}
    for key in params.keys():
        values = 2 ** params[key]
        params_dict[key] = values
    return params_dict


def create_loss_func(model):
    def loss_func(params, ts, ys):
        mask = ~jnp.isnan(jnp.array(ys))
        ys = jnp.atleast_2d(ys)
        y0 = ys[0, :]
        y_pred = model(ts, y0, params)
        ys = jnp.where(mask, ys, 0)
        y_pred = jnp.where(mask, y_pred, 0)
        # print(ys,y_pred)
        non_nan_count = jnp.sum(mask)

        loss = jnp.sum((y_pred - ys) ** 2) / non_nan_count
        return loss

    return loss_func


def create_log_params_loss_func(model):
    """Loss function for log transformed parameters"""

    def loss_func(params, ts, ys):
        params = exponentiate_parameters(params)
        mask = ~jnp.isnan(jnp.array(ys))
        ys = jnp.atleast_2d(ys)
        y0 = ys[0, :]
        y_pred = model(ts, y0, params)
        ys = jnp.where(mask, ys, 0)
        y_pred = jnp.where(mask, y_pred, 0)
        # print(ys,y_pred)
        non_nan_count = jnp.sum(mask)

        loss = jnp.sum((y_pred - ys) ** 2) / non_nan_count
        return loss

    return loss_func


def create_log_params_log_loss_func(model):
    """Loss function for log transformed parameters"""

    def loss_func(params, ts, ys):
        params = exponentiate_parameters(params)
        mask = ~jnp.isnan(jnp.array(ys))
        ys = jnp.atleast_2d(ys)
        y0 = ys[0, :]
        y_pred = model(ts, y0, params)

        y_pred = jnp.log2(y_pred + 1)
        ys = jnp.log2(ys + 1)

        ys = jnp.where(mask, ys, 0)
        y_pred = jnp.where(mask, y_pred, 0)
        # print(ys,y_pred)
        non_nan_count = jnp.sum(mask)

        loss = jnp.sum((y_pred - ys) ** 2) / non_nan_count
        return loss

    return loss_func


def create_log_params_means_centered_loss_func(model):
    """Loss function for log transformed parameters.
    We do a simple input scaling using the mean per state variable (we add 1 everywhere to prevent division by zero)"""

    def loss_func(params, ts, ys):
        params = exponentiate_parameters(params)
        mask = ~jnp.isnan(jnp.array(ys))
        ys = jnp.atleast_2d(ys)
        y0 = ys[0, :]
        y_pred = model(ts, y0, params)
        ys = jnp.where(mask, ys, 0)

        ys += 1
        y_pred += 1
        scale = jnp.mean(ys, axis=0)

        ys /= scale
        y_pred /= scale

        y_pred = jnp.where(mask, y_pred, 0)
        # print(ys,y_pred)
        non_nan_count = jnp.sum(mask)

        loss = jnp.sum((y_pred - ys) ** 2) / non_nan_count
        return loss

    return loss_func


def create_log_params_means_centered_loss_func2(model, to_include: list):
    """Loss function for log transformed parameters.
    We do a simple input scaling using the mean per state variable (we add 1 everywhere to prevent division by zero).
    Furthermore, we allow for not every state variable to be learned (sometimes it is not in the model for example)"""

    def loss_func(params, ts, ys):
        params = exponentiate_parameters(params)
        mask = ~jnp.isnan(jnp.array(ys))
        ys = jnp.atleast_2d(ys)
        y0 = ys[0, :]
        y_pred = model(ts, y0, params)
        ys = jnp.where(mask, ys, 0)

        ys += 1
        y_pred += 1
        scale = jnp.mean(ys, axis=0)

        ys /= scale
        y_pred /= scale

        y_pred = jnp.where(mask, y_pred, 0)

        ys = ys[:, to_include]
        y_pred = y_pred[:, to_include]
        # print(ys,y_pred)
        non_nan_count = jnp.sum(mask)

        loss = jnp.sum((y_pred - ys) ** 2) / non_nan_count
        return loss

    return loss_func


def create_update_rule(optimizer, loss_func):
    def update(opt_state, params, ts, ys):
        """Update rule for the gradients for parameters"""
        loss = loss_func(params, ts, ys)
        grads = jax.jit(jax.grad(loss_func, 0))(params, ts, ys)  # loss w.r.t. parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss, grads

    return update


@jax.jit
def create_log_update_rule(optimizer, log_loss_func):
    def update_log(opt_state, params, ts, ys):
        """Update rule for the gradients for log-transformed parameters. Can only be applied
        to non-negative parameters"""
        log_params = log_transform_parameters(params)

        loss = log_loss_func(log_params, ts, ys)

        grads = jax.jit(jax.grad(log_loss_func, 0))(log_params, ts, ys)  # loss w.r.t. parameters
        updates, opt_state = optimizer.update(grads, opt_state)

        # we perform updates in log space, but only return params in lin space
        log_params = optax.apply_updates(log_params, updates)
        lin_params = exponentiate_parameters(log_params)
        return opt_state, lin_params, loss, grads

    return update_log


def global_norm(grads):
    """Calculate the global norm of a list of gradient arrays."""
    global_norm = []
    for key in grads.keys():
        value = float(grads[key]) ** 2
        global_norm.append(value)
    global_norm = jnp.sqrt(jnp.sum(jnp.array(global_norm)))
    return global_norm
