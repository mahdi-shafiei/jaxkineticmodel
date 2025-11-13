import inspect
import time
from dataclasses import dataclass

import jax
import optax
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel
from scipy.stats import qmc
import logging
from jaxkineticmodel.building_models import JaxKineticModelBuild as jkm
import numpy as np
import jax.numpy as jnp
import equinox
import pandas as pd
import traceback

from jaxkineticmodel.load_sbml.jax_kinetic_model import NeuralODE

jax.config.update("jax_enable_x64", True)
logger = logging.getLogger(__name__)


@dataclass
class TrainingData:
    """Training data class that makes dataset ready for training of a kinetic model"""
    ts: list[float]
    dataset: np.ndarray  #(n_samples, n_ts,n_species)
    species: list[list[str]]  # it could be that you need to have to specify species per sample, as they may differ
    model: SBMLModel

    def __init__(self, ts, dataset, species, model):
        assert len(dataset.shape) == 3, """Shape of dataset should be (n_samples, n_timepoints, n_species)"""
        assert isinstance(model, SBMLModel), """Model should be SBMLModel object"""
        assert dataset.shape[1] == len(ts), """Dataset shape does not match length timepoints"""
        assert isinstance(species, list), "Species should be of type [[]]"
        assert all(isinstance(inner, list) for inner in species), """Species list is defined per sample [[]]"""
        self.model = model
        self.ts = ts
        self.species = species
        self.dataset = dataset
        self.dataset = self._preprocess_data_set()

    def _preprocess_data_set(self):
        """Processes incomplete data to ensure that it matches"""
        model_species = self.model.species_names
        n_samples, n_timepoints, n_species = self.dataset.shape
        processed_dataset = np.empty((n_samples, n_timepoints, len(model_species)))  #change the dataset
        processed_dataset[:] = np.nan
        processed_dataset[:, 0, :] = self.model.y0

        for k, specimen_list in enumerate(self.species):
            indices = [model_species.index(x) for x in specimen_list]

            # this assigns the right species to the dataset with the right initial condition
            for m, index in enumerate(indices):
                processed_dataset[:, :, index] = self.dataset[k, :, m]

        return processed_dataset

    def add_datapoints(self,
                       sample_index: int,
                       species: str,
                       ts: list,
                       values: list):
        """It should be possible to add a datapoint later to the dataset, e.g., when you are having additional
        measurement or whatever"""

        assert species in self.model.species_names, "species should be in the kinetic model (and be of type 'str')"
        assert len(values) == len(ts), "value length should be the same length as ts"
        species_index = self.model.species_names.index(species)
        self.dataset[sample_index, :, species_index] = values
        return self.dataset










class Trainer:
    """Trainer class that fits data using a set of parameter initializations,
    Input: a model that is a JaxKineticModel class to fit, and a dataset"""

    def __init__(self,
                 model: [NeuralODE, jkm.NeuralODEBuild],
                 data: TrainingData,
                 n_iter: int,
                 initial_conditions=None,
                 learning_rate=1e-3,
                 loss_threshold=1e-4,
                 optimizer=None,
                 optim_space="log",
                 clip=4,
                 ):
        assert isinstance(data, TrainingData), """Data is not of type TrainingData"""
        if isinstance(model, NeuralODE):
            # initial conditions need to be retrieved to ensure that the dataset matches in the loss function.
            # this makes sure that we can evaluate, also for points where there is no data.
            self.initial_conditions = dict(zip(model.species_names, model.y0))

            self.kin_model = model
            self.species_names = self.kin_model.species_names

            self.parameters = list(model.parameters.keys())
        elif isinstance(model, jkm.NeuralODEBuild):
            logger.info("NeuralODEBuild object is not tested yet")
            # To do: add initial conditions to object
            if initial_conditions is not None:
                self.initial_conditions = dict(zip(model.species_names, initial_conditions))
            else:
                logger.error("NeuralODEBuild class requires y0 input in Trainer object.")
            self.species_names = model.species_names
            self.parameters = model.parameter_names
            self.kin_model = model
        else:
            logger.error(f"{model} is not a JaxKineticModel class")

        self.ts = data.ts
        self.lr = learning_rate
        self.optim_space = optim_space

        if optimizer is None:
            self.optimizer = self._add_optimizer(clip=clip)
        elif isinstance(optimizer, optax.GradientTransformation):
            self.optimizer = optimizer
        else:
            logger.error(f"optimizer args {optimizer} is not an optax.GradientTransformation object.")

        self.make_step, self.loss_func = self._choose_optimization_space()

        self.loss_threshold = loss_threshold
        self.n_iter = n_iter

        self.dataset = data.dataset
        self.parameter_sets = None

    def _choose_optimization_space(self):
        """Choose between linear or log optimization space. The loss function
        in the case of logarithmic space exponentiates the parameters"""
        if self.optim_space == "log":
            loss_func = self._create_loss_func(log_mean_centered_loss_func)
            self.loss_func = loss_func
            update_rule = jax.jit(self.update_log_wrap)

        elif self.optim_space == "linear":
            loss_func = self._create_loss_func(mse_loss_func)
            self.loss_func = loss_func
            update_rule = self.update_lin_wrap

        return update_rule, loss_func

    def update_log_wrap(self, opt_state, params, ts, ys):
        return update_log(opt_state, params, ts, ys, self.loss_func, self.optimizer)

    def update_lin_wrap(self, opt_state, params, ts, ys):
        return update_linear(opt_state, params, ts, ys, self.loss_func, self.optimizer)

    def _create_loss_func(self, loss_func, **kwargs):
        """Function that helps to implement custom loss functions.
        It will be up to the user to ensure proper usage for log or linspace optimization"""
        model = self.kin_model
        if not callable(loss_func):
            logger.error("loss_func is not callable")

        arguments = inspect.signature(loss_func).parameters
        required_args = {'params', 'ts', 'ys'}
        if required_args.issubset(arguments.keys()):
            def wrapped_loss(params, ts, ys):
                return loss_func(params, ts, ys, model, **kwargs)
        else:
            logger.error(f"required arguments {required_args} are not in loss_function")

        self.loss_func = wrapped_loss
        return wrapped_loss

    # def _process_data(self):
    #     """Ensuring that dataset species that are not in model are not passed as part of the dataset"""
    #     # we start by adding an empty dataset with the shape of [y0]*ts
    #     processed_dataset = np.empty((len(self.ts), len(self.species_names)))
    #     processed_dataset[:] = np.nan
    #     processed_dataset = pd.DataFrame(processed_dataset, index=self.ts, columns=self.species_names)
    #
    #     # we map the processed data on the right column
    #     for species_name in self.dataset.columns:
    #         if species_name in self.species_names:
    #             processed_dataset[species_name] = self.dataset[species_name].values
    #
    #         else:
    #             logger.info(f"species {species_name} not in model, excluded from data for parameter estimation")
    #
    #     #initial conditions need to be in the data for the loss function to be calculated
    #     # thin
    #     inits_with_nans = processed_dataset.iloc[0, :].isna().to_dict()
    #     for (name, initial_cond) in inits_with_nans.items():
    #         if initial_cond:
    #             processed_dataset.loc[0, name] = self.initial_conditions[name]
    #
    #     return processed_dataset

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
        bounds = self._generate_bounds(parameters_base=parameters_base, lower_bound=lower_bound,
                                       upper_bound=upper_bound)

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

    def train(self):
        """Train model given the initializations"""
        loss_per_iteration_dict = {}
        optimized_parameters_dict = {}
        global_norm_dict = {}
        dataset = jnp.array(self.dataset)
        ts = self.ts
        # loop over parameter sets
        for init in range(np.shape(self.parameter_sets)[0]):
            params_init = self.parameter_sets.iloc[init, :].to_dict()
            opt_state = self.optimizer.init(params_init)

            loss_per_iter = []
            gradient_norms = []
            try:
                for step in range(self.n_iter):  # loop over number of iterations

                    opt_state, params_init, loss, grads = self.make_step(opt_state=opt_state,
                                                                         params=params_init,
                                                                         ts=ts,
                                                                         ys=dataset)
                    loss_per_iter.append(loss)

                    if loss < self.loss_threshold:
                        logger.info("loss threshold reached")

                        # global_norm_dict[init] = gradient_norms

                        break

                    if step % 20 == 0:
                        print(f"Step {step}, Loss {loss}")

                loss_per_iteration_dict[init] = loss_per_iter
                optimized_parameters_dict[init] = params_init
                global_norm_dict[init] = gradient_norms



            except Exception as e:
                print(f"Encountered an error: {e}")

                logger.error(f"init {init} could not be optimized")
                traceback.print_exc()
                loss_per_iteration_dict[init] = loss_per_iter
                loss_per_iteration_dict[init].append(-1)

        return optimized_parameters_dict, loss_per_iteration_dict


def update_linear(opt_state,
                  params,
                  ts,
                  ys,
                  loss_func,
                  optimizer):
    """Update rule for the gradients for log-transformed parameters. Can only be applied
    to non-negative parameters"""

    loss = loss_func(params, ts, ys)

    grads = jax.grad(loss_func, 0)(params, ts, ys)  # loss w.r.t. parameters
    updates, opt_state = optimizer.update(grads, opt_state)

    # we perform updates in log space, but only return params in lin space
    params = optax.apply_updates(params, updates)

    return opt_state, params, loss, grads


def update_log(opt_state,
               params,
               ts,
               ys,
               loss_func,
               optimizer):
    """Update rule for the gradients for log-transformed parameters. Can only be applied
    to non-negative parameters"""

    log_params = jax.tree.map(lambda x: jnp.log2(x), params)

    loss, grads = jax.value_and_grad(loss_func, 0)(log_params, ts, ys)

    updates, opt_state = optimizer.update(grads, opt_state)

    # we perform updates in log space, but only return params in lin space
    log_params = optax.apply_updates(log_params, updates)

    lin_params = jax.tree.map(lambda x: jnp.exp2(x), log_params)

    return opt_state, lin_params, loss, grads




def log_mean_centered_loss_func(params, ts, ys, model):
    """A log mean centered loss function. Typically works well on systems biology models
    due to their exponential parameter distributions"""
    params = jax.tree.map(lambda x: jnp.exp2(x), params)
    mask = ~jnp.isnan(jnp.array(ys))
    ys = jnp.atleast_2d(ys)
    y0 = ys[:, 0, :]
    vmapped_model = jax.vmap(model, in_axes=(None,0,None))
    y_pred = vmapped_model(ts, y0, params)

    ys = jnp.where(mask, ys, 0)

    ys += 1
    y_pred += 1
    scale = jnp.mean(ys, axis=0)

    ys /= scale
    y_pred /= scale

    y_pred = jnp.where(mask, y_pred, 0)
    non_nan_count = jnp.sum(mask)

    loss = jnp.sum((y_pred - ys) ** 2) / non_nan_count
    return loss


def mse_loss_func(params, ts, ys, model):
    """A typical mean squared error loss function"""
    mask = ~jnp.isnan(jnp.array(ys))
    ys = jnp.atleast_2d(ys)
    y0 = ys[:, 0, :]
    vmapped_model = jax.vmap(model, in_axes=(None,0,None))
    y_pred = vmapped_model(ts, y0, params)
    ys = jnp.where(mask, ys, 0)
    y_pred = jnp.where(mask, y_pred, 0)
    # print(ys,y_pred)
    non_nan_count = jnp.sum(mask)

    loss = jnp.sum((y_pred - ys) ** 2) / non_nan_count
    return loss



#
# def log_transform_parameters(params):
#     params_dict = {}
#     for key in params.keys():
#         values = jnp.log2(params[key])
#         params_dict[key] = values
#     return params_dict


# def exponentiate_parameters(params):
#     params_dict = {}
#     for key in params.keys():
#         values = jnp.exp2(params[key])
#         params_dict[key] = values
#     return params_dict



# @jax.jit
# def create_log_params_log_loss_func(model):
#     """Loss function for log transformed parameters"""
#
#     def loss_func(params, ts, ys):
#         params = exponentiate_parameters(params)
#         mask = ~jnp.isnan(jnp.array(ys))
#         ys = jnp.atleast_2d(ys)
#         y0 = ys[0, :]
#         y_pred = model(ts, y0, params)
#
#         y_pred = jnp.log2(y_pred + 1)
#         ys = jnp.log2(ys + 1)
#
#         ys = jnp.where(mask, ys, 0)
#         y_pred = jnp.where(mask, y_pred, 0)
#         # print(ys,y_pred)
#         non_nan_count = jnp.sum(mask)
#
#         loss = jnp.sum((y_pred - ys) ** 2) / non_nan_count
#         return loss
#
#     return loss_func


# def create_log_params_means_centered_loss_func(model):
#     """Loss function for log transformed parameters.
#     We do a simple input scaling using the mean per state variable (we add 1 everywhere to prevent division by zero)"""
#
#     def log_mean_centered_loss_func(params, ts, ys):
#         params = exponentiate_parameters(params)
#         mask = ~jnp.isnan(jnp.array(ys))
#         ys = jnp.atleast_2d(ys)
#         y0 = ys[0, :]
#         y_pred = model(ts, y0, params)
#         ys = jnp.where(mask, ys, 0)
#
#         ys += 1
#         y_pred += 1
#         scale = jnp.mean(ys, axis=0)
#
#         ys /= scale
#         y_pred /= scale
#
#         y_pred = jnp.where(mask, y_pred, 0)
#         non_nan_count = jnp.sum(mask)
#
#         loss = jnp.sum((y_pred - ys) ** 2) / non_nan_count
#         return loss
#
#     return log_mean_centered_loss_func


# def create_log_params_means_centered_loss_func2(model, to_include: list):
#     """Loss function for log transformed parameters.
#     We do a simple input scaling using the mean per state variable (we add 1 everywhere to prevent division by zero).
#     Furthermore, we allow for not every state variable to be learned (sometimes it is not in the model for example)"""
#
#     def loss_func(params, ts, ys):
#         params = exponentiate_parameters(params)
#         mask = ~jnp.isnan(jnp.array(ys))
#         ys = jnp.atleast_2d(ys)
#         y0 = ys[0, :]
#         y_pred = model(ts, y0, params)
#         ys = jnp.where(mask, ys, 0)
#
#         ys += 1
#         y_pred += 1
#         scale = jnp.mean(ys, axis=0)
#
#         ys /= scale
#         y_pred /= scale
#
#         y_pred = jnp.where(mask, y_pred, 0)
#         ys = ys[:, to_include]
#         y_pred = y_pred[:, to_include]
#         non_nan_count = jnp.sum(mask)
#         loss = jnp.sum((y_pred - ys) ** 2) / non_nan_count
#         return loss
#
#     return loss_func

# @jax.jit
# def global_norm(grads):
#     """Calculate the global norm of a list of gradient arrays."""
#     global_norm = []
#     for key in grads.keys():
#         value = jnp.array(grads[key],float) ** 2
#         global_norm.append(value)
#     global_norm = jnp.sqrt(jnp.sum(jnp.array(global_norm)))
#     return global_norm
