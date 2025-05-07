

import tempfile
from collections import namedtuple
from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import libsbml
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd


from jaxkineticmodel.load_sbml.jax_kinetic_model import (
    JaxKineticModel, construct_param_point_dictionary, separate_params)
from jaxkineticmodel.load_sbml.sbml_model import SBMLModel




def mask_reaction_in_sbml(sbml_path: Path | str, reaction_to_remove: str) -> str:
    """
    Load an SBML file, remove a specified reaction, and return the modified model as string.

    Args:
        sbml_path: Path to SBML file
        reaction_to_remove: ID of the reaction to remove

    Returns:
        str: Modified SBML model as string, or empty string if there were errors
    """
    # Read SBML from string
    reader = libsbml.SBMLReader()
    document = reader.readSBMLFromFile(str(sbml_path))

    # Check for read errors
    if document.getNumErrors() > 0:
        raise ValueError("Error: SBML string is not valid")

    # Get the model
    model = document.getModel()
    if model is None:
        raise ValueError("Error: No model found in SBML string")

    # Remove specified reaction
    reaction = model.removeReaction(reaction_to_remove)
    if reaction is None:
        raise ValueError(f"Error: Reaction '{reaction_to_remove}' not found")

    # Write the modified model to string
    writer = libsbml.SBMLWriter()
    output_string = writer.writeSBMLToString(document)

    if not output_string:
        raise ValueError("Error writing SBML to string")

    return output_string


def random_batch_dataloader(arrays, batch_size, *, key):
    """
    Creates an infinite generator that yields random batches from the provided arrays.

    Args:
        arrays: A sequence of arrays, all with the same first dimension (dataset size)
        batch_size: The size of each batch to yield
        key: JAX PRNG key for random permutation of indices

    Yields:
        Tuple of batched arrays, each containing batch_size elements
    """
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def simulate_sbml(
    model: SBMLModel, t_span: jnp.ndarray, y0s: jnp.ndarray = None
) -> jnp.ndarray:
    jax_kmodel = model.get_kinetic_model(compile=True)
    initial_state = y0s if y0s is not None else model.y0
    parameters = model.parameters
    result = jax_kmodel(ts=t_span, y0=initial_state, params=parameters)
    return result


SyntheticDataset = namedtuple(
    "SyntheticDataset",
    [
        "ts",  # (t_cropped,) - Time points after cropping
        "ys",  # (n_samples * n_trajectories, t_cropped, n_species) - Noisy measurements
        "y_ground_truth",  # (n_trajectories, t_cropped, n_species) - Clean trajectories
        "species_names",  # List of species names from the model
    ],
)


def create_dataset_from_sbml(
    *,
    t_span: jnp.ndarray,
    key: jr.PRNGKey,
    sbml_path: Path | str,
    n_trajectories: int = 10,
    n_samples: int = 10,
    noise_level: float = 0.05,
    y0_noise_level: float = 0.05,
    crop_start: int = 0,
) -> SyntheticDataset:
    """
    Generate a synthetic dataset from an SBML model.

    Args:
        t_span: Time points to simulate
        key: JAX PRNG key for reproducibility
        sbml_path: Path to SBML model
        n_trajectories: Number of different initial conditions to simulate
        n_samples: Number of noisy samples to generate per trajectory
        noise_level: Standard deviation of measurement noise as fraction of species standard deviation
        y0_noise_level: Standard deviation of initial condition perturbations as fraction of species standard deviation
        crop_start: Number of initial time points to remove from the simulation

    Returns: dataset
    """
    keys = jr.split(key, 2)

    sbml_model = SBMLModel(str(sbml_path))
    data_base = simulate_sbml(sbml_model, t_span)  # (t_span, n_species)
    data_base = data_base[crop_start:]  # (t_cropped, n_species)
    t_span_cropped = t_span[crop_start:]  # (t_cropped,)

    species_stds = []
    for species_idx in range(len(sbml_model.species_names)):
        species_data = data_base[:, species_idx]
        species_std = jnp.std(species_data)
        species_stds.append(species_std)
    species_stds = jnp.array(species_stds)  # (n_species,)

    # Add measurement noise to the data
    noise = (
        noise_level
        * species_stds[None, None, :]
        * jr.normal(
            keys[1],
            shape=(
                n_samples * n_trajectories,
                len(t_span_cropped),
                len(sbml_model.species_names),
            ),
        )
    )  # (n_samples * n_trajectories, t_cropped, n_species)

    noisy_samples = jnp.tile(data_base, (n_samples * n_trajectories, 1, 1)) + noise
    noisy_samples = jnp.maximum(noisy_samples, 0.0)

    return SyntheticDataset(
        ts=t_span_cropped,
        ys=noisy_samples,
        y_ground_truth=data_base,
        species_names=sbml_model.species_names,
    )


# ## Model

# In[7]:


class NeuralODEFunc(eqx.Module):
    """Neural network defining the vector field f_θ(t,y) using eqx.MLP."""

    mlp: eqx.nn.MLP

    def __init__(
        self,
        data_size: int,
        width_size: int,
        depth: int,
        *,
        key,
    ):
        super().__init__()
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            final_activation=lambda x: x,  # Identity
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y)


class HybridODEFunc(eqx.Module):
    """Hybrid ODE function."""

    node_func: NeuralODEFunc
    kinetic_func: JaxKineticModel

    def __init__(self, node_func: NeuralODEFunc, kinetic_func: JaxKineticModel):
        self.node_func = node_func
        self.kinetic_func = kinetic_func

    def __call__(self, t, y, args):
        node_deriv = self.node_func(t, y, args)
        kinetic_deriv = self.kinetic_func(t, y, args)
        return node_deriv + kinetic_deriv


class HybridModel(eqx.Module):
    """Hybrid model dY/dt = f_θ(t,y) + g_ϕ(t,y)."""

    func: HybridODEFunc
    model_parameters: tuple
    v_symbols: dict
    reaction_names: tuple
    solver_params: dict

    def __init__(
        self,
        sbml_string: str,
        width_size: int,
        depth: int,
        *,
        key,
    ):
        self.solver_params = {
            "solver": diffrax.Tsit5(),
            "stepsize_controller": diffrax.PIDController(rtol=1e-3, atol=1e-6),
            "dt0": 1e-8,
            "event": diffrax.Event(
                cond_fn=lambda t, y, args, **kwargs: jnp.logical_or(
                    jnp.any(y < -10.0),
                    jnp.any(y > 1e7),
                )
            ),
            "max_steps": 65536,
        }

        node_key, kinetic_key = jr.split(key)
        with tempfile.NamedTemporaryFile(
            delete=True, suffix=".xml", mode="w"
        ) as temp_file:
            temp_file.write(sbml_string)
            temp_file_path = temp_file.name

            sbml_model = SBMLModel(temp_file_path)
            kinetic_model = sbml_model.get_kinetic_model(compile=True)
            kinetic_func = kinetic_model.func
            self.model_parameters = sbml_model.parameters
            self.v_symbols = kinetic_model.v_symbols
            self.reaction_names = tuple(kinetic_model.reaction_names)

        data_size = len(sbml_model.species_names)
        node_func = NeuralODEFunc(
            data_size,
            width_size,
            depth,
            key=node_key,
        )

        self.func = HybridODEFunc(node_func, kinetic_func)

    def __call__(self, ts, y0):
        global_params, local_params = separate_params(self.model_parameters)
        global_params = construct_param_point_dictionary(
            self.v_symbols, self.reaction_names, global_params
        )
        flux_array = jnp.zeros(len(self.reaction_names))

        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self.func),
            y0=y0,
            t0=ts[0],
            t1=ts[-1],
            args=(global_params, local_params, flux_array),
            saveat=diffrax.SaveAt(ts=ts),
            solver=self.solver_params["solver"],
            dt0=self.solver_params["dt0"],
            stepsize_controller=self.solver_params["stepsize_controller"],
            event=self.solver_params["event"],
            max_steps=self.solver_params["max_steps"],
            throw=False,
        )
        return solution.ys


def train_hybrid_model(
    kinetic_sbml: str,
    ts: jnp.ndarray,
    ys: jnp.ndarray,
    seed: int = 42,
    width_size: int = 32,
    depth: int = 2,
    batch_size: int = 32,
    lr: float = 1e-3,
    steps: int = 1000,
    negativity_penalty_scale: float = 1.0,
) -> HybridModel:
    """Initializes and trains a HybridModel."""

    key = jr.PRNGKey(seed)
    model_key, loader_key, train_key = jr.split(key, 3)

    loss_per_iteration = []

    model = HybridModel(
        kinetic_sbml,
        width_size,
        depth,
        key=model_key,
    )

    @eqx.filter_value_and_grad
    def grad_loss(current_model, ti, yi):
        y_pred = jax.vmap(current_model, in_axes=(None, 0))(ti, yi[:, 0])
        # Mask out inf/nan values which can occur during training
        mask = jnp.isfinite(y_pred) & jnp.isfinite(yi)
        safe_y_pred = jnp.where(mask, y_pred, 0.0)
        safe_yi = jnp.where(mask, yi, 0.0)

        errors = safe_yi - safe_y_pred
        num_valid = jnp.sum(mask)
        mse_loss = jnp.sum(errors**2) / jnp.maximum(
            1, num_valid
        )  # Avoid division by zero

        # Penalize negativity only on valid predictions
        neg_penalty = jnp.sum(jnp.maximum(0.0, -safe_y_pred) ** 2 * mask) / jnp.maximum(
            1, num_valid
        )

        loss = mse_loss + negativity_penalty_scale * neg_penalty
        return loss

    @eqx.filter_jit
    def make_step(ti, yi, current_model, opt_state, optim):
        loss, grads = grad_loss(current_model, ti, yi)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(current_model, eqx.is_inexact_array)
        )
        new_model = eqx.apply_updates(current_model, updates)
        return loss, new_model, opt_state

    # Initialize optimizer
    model_params = eqx.filter(model, eqx.is_inexact_array)
    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )
    opt_state = optim.init(model_params)

    # Training loop
    train_loader = random_batch_dataloader((ys,), batch_size, key=loader_key)
    for step in range(steps):
        (yi,) = next(train_loader)
        loss, model, opt_state = make_step(ts, yi, model, opt_state, optim)
        loss_per_iteration.append(loss)
        if step % 100 == 0 or step == steps - 1:
            print(f"Step: {step}/{steps}, Loss: {loss.item():.4f}")

    print(f"Training finished. Final Loss: {loss.item():.4f}")
    return model,loss_per_iteration


# ## Training data

# In[11]:


seed = 2323
np.random.seed(seed)
key = jr.PRNGKey(seed)

dur = 10
dt = 0.01
t_span = jnp.arange(0, dur, dt)
model_path = "models/sbml_models/working_models/Garde2020.xml"
dataset = create_dataset_from_sbml(
    t_span=t_span,
    key=key,
    sbml_path=model_path,
    noise_level=0.05,
    y0_noise_level=0.00,
)

for species_idx, species_name in enumerate(dataset.species_names):
    plt.plot(jnp.tile(t_span, (dataset.ys.shape[0], 1)), dataset.ys[:, :, species_idx], c="gray", alpha=0.5)
plt.plot(t_span, dataset.y_ground_truth, c="black")
plt.title("Training data")
plt.xlabel('Time')
plt.ylabel('Concentration')

plt.show()


# ## Training and results

# In[12]:
reactions = ['Positive_feedback_of_Gp_on_itself',
             'Gp_diffusion_to_the_interior_cells',
             'Biomass_production',
             'Diffusion_of_Ammonia_from_interior_to_middle_layer',
             'Production_of_Ammonia_using_Glutamate_by_inner_cells',
             'Ammonia_diffusion_from_middle_layer_to_periphery',
             'loss_of_Ammonia_du_to_diffusion_to_the_environment',
             'Diffusion_of_Glutamate_from_middle_to_interior_layer',
             'Production_of_Ammonia_in_middle_layer_using_glutamate',
             'Use_of_glutamate_in_middle_layer_for_biomass_production']

for reaction_to_remove in reactions:
    masked_sbml_str = mask_reaction_in_sbml(model_path, reaction_to_remove)
    model,loss_per_iteration = train_hybrid_model(masked_sbml_str, t_span, dataset.ys, seed=seed)





    def predict_kinetic_only(model, ts, y0):
        global_params, local_params = separate_params(model.model_parameters)
        global_params = construct_param_point_dictionary(
            model.v_symbols, model.reaction_names, global_params
        )
        flux_array = jnp.zeros(len(model.reaction_names))

        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(model.func.kinetic_func),
            y0=y0,
            t0=ts[0],
            t1=ts[-1],
            args=(global_params, local_params,flux_array),
            saveat=diffrax.SaveAt(ts=ts),
            solver=model.solver_params["solver"],
            dt0=model.solver_params["dt0"],
            stepsize_controller=model.solver_params["stepsize_controller"],
            event=model.solver_params["event"],
            max_steps=model.solver_params["max_steps"],
            throw=False,
        )
        return solution.ys


    y_pred_hybrid = model(t_span, dataset.y_ground_truth[0])
    y_pred_kinetic = predict_kinetic_only(model, t_span, dataset.y_ground_truth[0])

    y_pred_kinetic_df = pd.DataFrame(y_pred_kinetic, columns=dataset.species_names)
    y_pred_hybrid_df = pd.DataFrame(y_pred_hybrid, columns=dataset.species_names)
    y_pred_kinetic_df.to_csv(f"results/hybrid_model_example/predicted_timeseries/garde_predicted_kinetic_timeseriess_{reaction_to_remove}.csv")
    y_pred_hybrid_df.to_csv(f"results/hybrid_model_example/predicted_timeseries/garde_predicted_hybrid_timeseries_{reaction_to_remove}.csv")
    pd.Series(loss_per_iteration).to_csv(f"results/hybrid_model_example/loss_curves/loss_per_iteration_{reaction_to_remove}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    ax = axes[0]
    lines_gt_left = []
    lines_kinetic = []
    for species_idx in range(len(dataset.species_names)):
        (line_gt,) = ax.plot(
            t_span, dataset.y_ground_truth[:, species_idx], c="gray", alpha=0.5
        )
        (line_kinetic,) = ax.plot(
            t_span, y_pred_kinetic[:, species_idx], c="blue", alpha=0.8
        )
        if species_idx == 0:
            lines_gt_left.append(line_gt)
            lines_kinetic.append(line_kinetic)

    ax.set_title(f"Masked Kinetic Model\n('{reaction_to_remove}' removed)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    if lines_gt_left and lines_kinetic:
        ax.legend([lines_gt_left[0], lines_kinetic[0]], ["Ground Truth", "Masked Kinetic"])
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    lines_gt_right = []
    lines_hybrid = []
    for species_idx in range(len(dataset.species_names)):
        (line_gt,) = ax.plot(
            t_span, dataset.y_ground_truth[:, species_idx], c="gray", alpha=0.5
        )
        (line_hybrid,) = ax.plot(t_span, y_pred_hybrid[:, species_idx], c="black")
        if species_idx == 0:
            lines_gt_right.append(line_gt)
            lines_hybrid.append(line_hybrid)

    ax.set_title("Trained Hybrid Model Prediction")
    ax.set_xlabel("Time")
    if lines_gt_right and lines_hybrid:
        ax.legend([lines_gt_right[0], lines_hybrid[0]], ["Ground Truth", "Trained Hybrid"])
    ax.grid(True, alpha=0.3)

    plt.suptitle("Model Predictions vs Ground Truth", y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()

